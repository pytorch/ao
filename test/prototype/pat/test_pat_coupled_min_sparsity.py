# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import unittest
from unittest import mock

import torch
from torch.distributed.tensor import distribute_tensor, init_device_mesh
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import DistributedTestMixin
from torchao.prototype.pat.group import (
    Dim0Grouper,
    Dim1Grouper,
    ElemGrouper,
    Grouper,
    KElementGrouper,
    LayerGrouper,
)
from torchao.prototype.pat.optim import CoupledMinSparsityConstraint, PruneOptimizer
from torchao.prototype.pat.optim.prox_executor import apply_coupled_prox, grouped_view

D, INNER, HIDDEN, VOCAB = 4, 4, 6, 3
EXECUTOR = "torchao.prototype.pat.optim.prox_executor"
OPTIMIZER = "torchao.prototype.pat.optim.pruneopt"


def _coupled_group(params, target, group_type, **overrides):
    return dict(
        params=list(params),
        prox_type="CoupledMinSparsityConstraint",
        group_type=group_type,
        couple_key="resid",
        min_sparsity=target,
        **overrides,
    )


def _zero_channels(p, axis):
    view = p.detach() if axis == 0 else p.detach().transpose(0, 1)
    return set(view.eq(0).all(dim=1).nonzero(as_tuple=True)[0].tolist())


def _step(optimizer, params, steps=1, dense_gradient=False):
    for _ in range(steps):
        optimizer.zero_grad()
        for p in params:
            p.grad = torch.ones_like(p) if dense_gradient else torch.zeros_like(p)
        optimizer.step()


def _optimizer(groups, lr=0.0, healing=100, latent_weights=True, momentum=0.0):
    return PruneOptimizer(
        torch.optim.SGD(groups, lr=lr, momentum=momentum),
        healing_start_step=healing,
        latent_weights=latent_weights,
    )


class ToyBlockStack(torch.nn.Module):
    """Small reader/writer stack sharing one residual channel axis."""

    def __init__(self):
        super().__init__()
        self.tok_embeddings = torch.nn.Parameter(torch.randn(VOCAB, D))
        for name, shape in (
            ("wv", (INNER, D)),
            ("wo", (D, INNER)),
            ("w1", (HIDDEN, D)),
            ("w2", (D, HIDDEN)),
        ):
            setattr(
                self,
                name,
                torch.nn.ParameterList(
                    torch.nn.Parameter(torch.randn(*shape)) for _ in range(2)
                ),
            )

    def readers(self):
        return [self.tok_embeddings, *self.wv, *self.w1]

    def writers(self):
        return [*self.wo, *self.w2]

    def specs(self):
        return [(p, Dim1Grouper, {}) for p in self.readers()] + [
            (p, Dim0Grouper, {}) for p in self.writers()
        ]

    def param_groups(self, target, **overrides):
        return [
            _coupled_group(self.readers(), target, "Dim1Grouper", **overrides),
            _coupled_group(self.writers(), target, "Dim0Grouper", **overrides),
        ]

    def nested_groups(self, target, **overrides):
        groups = self.param_groups(target)
        for group in groups:
            group["coupled"] = {
                key: group.pop(key) for key in ("couple_key", "min_sparsity")
            }
            group["coupled"]["group_type"] = group["group_type"]
            group.update(
                prox_type="MinSparsityConstraint", min_sparsity=0.0, **overrides
            )
        return groups


class _CoupledAssertions:
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def assert_shared_mask(self, model, target):
        masks = [_zero_channels(p, 1) for p in model.readers()]
        masks += [_zero_channels(p, 0) for p in model.writers()]
        self.assertEqual(masks, [masks[0]] * len(masks))
        self.assertEqual(len(masks[0]), math.ceil(target * D))
        return masks[0]

    def assert_literal_metrics(self, optimizer, params):
        total = sum(p.numel() for p in params)
        zeros = sum(p.eq(0).sum().item() for p in params)
        self.assertAlmostEqual(optimizer.relative_sparsity, zeros / total)
        for p in params:
            self.assertAlmostEqual(
                optimizer.state[p]["sparsity_frac"], p.eq(0).sum().item() / p.numel()
            )


class TestCoupledExecutor(_CoupledAssertions, common_utils.TestCase):
    def test_combined_scores_match_concatenated_norm(self):
        a, b = torch.randn(4, 3), torch.randn(4, 5)
        joint = torch.cat((a, b), dim=1)
        squared = a.square().sum(dim=1) + b.square().sum(dim=1)
        for score_type, divisor in (
            ("rms", math.sqrt(8)),
            ("l2", 1),
            ("param_cost", 8),
        ):
            with self.subTest(score_type=score_type):
                prox = CoupledMinSparsityConstraint(0.0, 0.5, score_type)
                scores = prox.combine_scores(squared, 8)
                self.assertEqual(
                    scores, torch.linalg.vector_norm(joint, dim=1) / divisor
                )
                self.assertEqual(scores, prox.score(joint))

    @torch.no_grad()
    def test_shared_mask_and_result_contract(self):
        for target in (0.0, 0.3, 1.0):
            with self.subTest(target=target):
                model = ToyBlockStack()
                params = model.readers() + model.writers()
                result = apply_coupled_prox(
                    model.specs(), CoupledMinSparsityConstraint(0.0, target), target
                )
                self.assert_shared_mask(model, target)
                self.assertEqual(result.channels, D)
                self.assertEqual(result.zero_channels, math.ceil(target * D))
                self.assertEqual(result.numel, sum(p.numel() for p in params))
                self.assertEqual(
                    result.zero_elts, sum(p.eq(0).sum().item() for p in params)
                )
                self.assertEqual(len(result.parameters), len(params))
                for record, p in zip(result.parameters, params):
                    self.assertIs(record.parameter, p)
                    self.assertEqual(record.numel, p.numel())
                    self.assertEqual(record.zero_elts, p.eq(0).sum().item())

    def test_float64_scoring_and_noncontiguous_flatten_writeback(self):
        for dtype in (torch.float32, torch.float64):
            for cls, target in ((ElemGrouper, 0.25), (LayerGrouper, 1.0)):
                with self.subTest(dtype=dtype, grouper=cls.__name__):
                    p = torch.arange(1, 13, dtype=dtype).reshape(3, 4).t()
                    self.assertFalse(p.is_contiguous())
                    expected = p.clone()
                    if cls is LayerGrouper:
                        expected.zero_()
                    else:
                        expected[expected <= 3] = 0
                    result = apply_coupled_prox(
                        [(p, cls, {})],
                        CoupledMinSparsityConstraint(0.0, target),
                        target,
                    )
                    self.assertEqual(p, expected)
                    self.assertEqual(result.zero_elts, int(expected.eq(0).sum()))
                    self.assertEqual(result.numel, p.numel())

    def test_selects_lowest_joint_score_not_individual_scores(self):
        reader = torch.tensor([[0.01, 10.0, 0.01]])
        writer = torch.tensor([[10.0], [0.01], [0.01]])
        apply_coupled_prox(
            [(reader, Dim1Grouper, {}), (writer, Dim0Grouper, {})],
            CoupledMinSparsityConstraint(0.0, 1 / 3),
            1 / 3,
        )
        self.assertEqual(_zero_channels(reader, 1), {2})
        self.assertEqual(_zero_channels(writer, 0), {2})

    def test_counts_existing_zeros_not_only_chosen_channels(self):
        p = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
        result = apply_coupled_prox(
            [(p, Dim0Grouper, {})], CoupledMinSparsityConstraint(0.0, 0.25), 0.25
        )
        self.assertEqual(result.zero_channels, 1)
        self.assertEqual(len(_zero_channels(p, 0)), 2)
        self.assertEqual(result.zero_elts, 5)
        self.assertEqual(result.parameters[0].zero_elts, 5)
        self.assertEqual(result.numel, 8)

    def test_invalid_views_leave_originals_unchanged(self):
        cases = [
            (torch.randn(5, 2), Dim0Grouper, {}, "same leading dimension"),
            (torch.randn(2, 3, 4), Grouper, {}, "2-D"),
            (torch.empty(4, 2, device="meta"), Dim0Grouper, {}, "same device"),
            (torch.randn(1, 5), KElementGrouper, {"k": 4}, "padded"),
        ]
        for bad, grouper, kwargs, message in cases:
            with self.subTest(message=message):
                good = torch.randn(4, 2)
                originals = [p.clone() for p in (good, bad)]
                with self.assertRaisesRegex(ValueError, message):
                    apply_coupled_prox(
                        [(good, Dim0Grouper, {}), (bad, grouper, kwargs)],
                        CoupledMinSparsityConstraint(0.0, 0.5),
                        0.5,
                    )
                self.assertEqual(good, originals[0])
                if bad.device.type != "meta":
                    self.assertEqual(bad, originals[1])


class TestCoupledConfig(_CoupledAssertions, common_utils.TestCase):
    def test_cluster_mismatches_fail_at_construction(self):
        for nested in (False, True):
            for field, value in (
                ("min_sparsity", 0.25),
                ("min_sparsity_schedule", True),
                ("score_type", "l2"),
                ("prox_freq", 2),
            ):
                with self.subTest(nested=nested, field=field):
                    model = ToyBlockStack()
                    groups = (
                        model.nested_groups(0.5) if nested else model.param_groups(0.5)
                    )
                    cfg = groups[1]["coupled"] if nested else groups[1]
                    cfg[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        _optimizer(groups)

    def test_implicit_defaults_match_explicit_defaults(self):
        for nested in (False, True):
            model = ToyBlockStack()
            groups = model.nested_groups(0.5) if nested else model.param_groups(0.5)
            cfg = groups[1]["coupled"] if nested else groups[1]
            cfg.update(min_sparsity_schedule=False, score_type="rms", prox_freq=1)
            optimizer = _optimizer(groups)
            _step(optimizer, list(model.parameters()))
            self.assert_shared_mask(model, 0.5)
            self.assert_literal_metrics(optimizer, list(model.parameters()))

    def test_malformed_nested_config_fails_at_construction(self):
        for value in (None, False, [], 1, "bad", {}):
            with self.subTest(value=value):
                group = ToyBlockStack().nested_groups(0.5)[0]
                group["coupled"] = value
                with self.assertRaises(ValueError):
                    _optimizer([group])

    def test_required_fields_including_nested_target(self):
        for nested in (False, True):
            for field in ("group_type", "couple_key", "min_sparsity"):
                with self.subTest(nested=nested, field=field):
                    model = ToyBlockStack()
                    group = (
                        model.nested_groups(0.5) if nested else model.param_groups(0.5)
                    )[0]
                    if nested:
                        group["min_sparsity"] = 0.75
                    del (group["coupled"] if nested else group)[field]
                    with self.assertRaisesRegex(ValueError, field):
                        _optimizer([group])

    def test_rejects_missing_parent_svd_and_ambiguous_stages(self):
        for case in (
            "no_parent",
            "svd_parent",
            "svd_coupled",
            "packed_coupled",
            "ambiguous",
        ):
            with self.subTest(case=case):
                group = ToyBlockStack().nested_groups(0.5)[0]
                if case == "no_parent":
                    del group["prox_type"]
                elif case == "svd_parent":
                    group.update(
                        group_type="SVDGrouper",
                        prox_type="MinRankConstraint",
                        min_sparsity=0.5,
                    )
                elif case in ("svd_coupled", "packed_coupled"):
                    group["coupled"]["group_type"] = (
                        "SVDGrouper" if case == "svd_coupled" else "PackedSVDGrouper"
                    )
                    group["coupled"]["npack"] = 2
                else:
                    group.update(
                        prox_type="CoupledMinSparsityConstraint",
                        couple_key="resid",
                        min_sparsity=0.5,
                    )
                with self.assertRaises(ValueError):
                    _optimizer([group])

    def test_invalid_frequency_and_key(self):
        for nested in (False, True):
            for field, values in (
                ("prox_freq", (0, -1, 1.5, True, "2")),
                ("couple_key", ("", None, 1, False)),
            ):
                for value in values:
                    with self.subTest(nested=nested, field=field, value=value):
                        model = ToyBlockStack()
                        group = (
                            model.nested_groups(0.5)
                            if nested
                            else model.param_groups(0.5)
                        )[0]
                        (group["coupled"] if nested else group)[field] = value
                        with self.assertRaisesRegex(ValueError, field):
                            _optimizer([group])

    def test_nested_schedule_independently_requires_finite_healing(self):
        for healing in (sys.maxsize, float("inf")):
            group = ToyBlockStack().nested_groups(0.5, min_sparsity_schedule=False)[0]
            group["coupled"]["min_sparsity_schedule"] = True
            with self.assertRaisesRegex(ValueError, "finite healing_start_step"):
                _optimizer([group], healing=healing)

    def test_layout_validation_precedes_optimizer_mutation(self):
        reader = torch.nn.Parameter(torch.ones(3, 4))
        writer = torch.nn.Parameter(torch.ones(5, 2))
        groups = [
            _coupled_group([reader], 0.5, "Dim1Grouper"),
            _coupled_group([writer], 0.5, "Dim0Grouper"),
        ]
        with self.assertRaisesRegex(ValueError, "same leading dimension"):
            _optimizer(groups, lr=0.1)
        writer = torch.nn.Parameter(torch.ones(4, 2))
        groups[1]["params"] = [writer]
        optimizer = _optimizer(groups, lr=0.1)
        with torch.no_grad():
            writer.set_(torch.ones(5, 2))
        originals = [p.detach().clone() for p in (reader, writer)]
        with self.assertRaisesRegex(ValueError, "same leading dimension"):
            _step(optimizer, [reader, writer], dense_gradient=True)
        self.assertEqual([reader, writer], originals)
        self.assertEqual(optimizer.num_steps, 0)
        self.assertEqual(len(optimizer.state), 0)

    def test_dense_attention_strides_fail_before_optimizer_mutation(self):
        p = torch.nn.Parameter(torch.arange(1.0, 17.0).reshape(4, 4).t())
        group = _coupled_group([p], 0.5, "AttentionHeadGrouperDim0", num_heads=2)
        original = p.detach().clone()
        with self.assertRaisesRegex(ValueError, "unsupported tensor layout"):
            _optimizer([group], lr=0.1)
        self.assertEqual(p, original)
        p = torch.nn.Parameter(original.contiguous())
        group["params"] = [p]
        optimizer = _optimizer([group], lr=0.1)
        with torch.no_grad():
            p.set_(p.t())
        before = p.detach().clone()
        with self.assertRaisesRegex(ValueError, "unsupported tensor layout"):
            _step(optimizer, [p], dense_gradient=True)
        self.assertEqual(p, before)
        self.assertEqual(optimizer.num_steps, 0)
        self.assertEqual(len(optimizer.state), 0)

    def test_pruning_step_revalidates_changed_cluster(self):
        for field, value in (
            ("min_sparsity", 0.25),
            ("min_sparsity_schedule", True),
            ("score_type", "l2"),
            ("prox_freq", 2),
        ):
            with self.subTest(field=field):
                model = ToyBlockStack()
                optimizer = _optimizer(model.nested_groups(0.5))
                _step(optimizer, list(model.parameters()))
                optimizer.param_groups[1]["coupled"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    _step(optimizer, list(model.parameters()))


class TestCoupledOptimizer(_CoupledAssertions, common_utils.TestCase):
    def test_exact_shared_budget_with_and_without_latent_weights(self):
        for latent in (False, True):
            model = ToyBlockStack()
            optimizer = _optimizer(model.param_groups(0.5), latent_weights=latent)
            _step(optimizer, list(model.parameters()), steps=3)
            self.assert_shared_mask(model, 0.5)
            self.assert_literal_metrics(optimizer, list(model.parameters()))
            self.assertEqual(optimizer.relative_sparsity, 0.5)
            self.assertEqual(
                ["latent" in optimizer.state[p] for p in model.parameters()],
                [latent] * len(list(model.parameters())),
            )

    def test_nested_inherits_schedule_score_and_regularization(self):
        model = ToyBlockStack()
        groups = model.nested_groups(
            0.75, min_sparsity_schedule=True, score_type="param_cost", reg_lambda=0.7
        )
        optimizer = _optimizer(groups, healing=8)
        with mock.patch(
            f"{OPTIMIZER}.apply_coupled_prox", wraps=apply_coupled_prox
        ) as apply:
            _step(optimizer, list(model.parameters()), steps=2)
        self.assertEqual(apply.call_count, 2)
        prox = apply.call_args.args[1]
        self.assertEqual((prox.score_type, prox.reg_lambda), ("param_cost", 0.7))
        self.assertEqual(apply.call_args_list[0].args[2], 0.0)
        self.assertAlmostEqual(apply.call_args.args[2], 0.75 * (1 - (1 - 1 / 8) ** 3))

    def test_nested_overrides_inherited_values(self):
        model = ToyBlockStack()
        groups = model.nested_groups(
            0.5, min_sparsity_schedule=True, score_type="param_cost", reg_lambda=0.7
        )
        for group in groups:
            group["coupled"].update(
                min_sparsity_schedule=False, score_type="l2", reg_lambda=0.2
            )
        optimizer = _optimizer(groups)
        with mock.patch(
            f"{OPTIMIZER}.apply_coupled_prox", wraps=apply_coupled_prox
        ) as apply:
            _step(optimizer, list(model.parameters()))
        prox = apply.call_args.args[1]
        self.assertEqual((prox.score_type, prox.reg_lambda), ("l2", 0.2))
        self.assertEqual(apply.call_args.args[2], 0.5)
        self.assert_shared_mask(model, 0.5)

    def test_schedule_ramps_to_final_mask_at_healing_boundary(self):
        for nested in (False, True):
            model = ToyBlockStack()
            groups = (
                model.nested_groups(0.75, min_sparsity_schedule=True)
                if nested
                else model.param_groups(0.75, min_sparsity_schedule=True)
            )
            optimizer = _optimizer(groups, healing=8)
            counts = []
            for _ in range(8):
                _step(optimizer, list(model.parameters()))
                counts.append(len(_zero_channels(model.tok_embeddings, 1)))
            self.assertEqual(counts[0], 0)
            self.assertEqual(counts, sorted(counts))
            self.assertGreater(len(set(counts)), 2)
            self.assert_shared_mask(model, 0.75)

    def test_coupled_frequency_is_independent_and_skips_cache_total_metrics(self):
        for skipped in ("parent", "coupled"):
            p = torch.nn.Parameter(torch.arange(1.0, 17.0).reshape(4, 4))
            group = dict(
                params=[p],
                group_type="Dim0Grouper",
                prox_type="MinSparsityConstraint",
                min_sparsity=0.25,
                prox_freq=4 if skipped == "parent" else 1,
                coupled=dict(
                    group_type="Dim1Grouper", couple_key="resid", min_sparsity=0.5
                ),
            )
            if skipped == "coupled":
                group["coupled"]["prox_freq"] = 4
            optimizer = _optimizer([group])
            with mock.patch(
                f"{OPTIMIZER}.apply_coupled_prox", wraps=apply_coupled_prox
            ) as apply:
                _step(optimizer, [p])
                cached = (optimizer.relative_sparsity, optimizer.relative_factored_frac)
                self.assert_literal_metrics(optimizer, [p])
                _step(optimizer, [p])
            self.assertEqual(
                (optimizer.relative_sparsity, optimizer.relative_factored_frac), cached
            )
            self.assertEqual(apply.call_count, 2 if skipped == "parent" else 1)
            self.assertEqual(len(_zero_channels(p, 1)), 2 if skipped == "parent" else 0)
            self.assertEqual(len(_zero_channels(p, 0)), 0 if skipped == "parent" else 1)

    def test_final_prune_forces_coupled_frequency_before_healing(self):
        for nested in (False, True):
            for scheduled in (False, True):
                model = ToyBlockStack()
                groups = model.nested_groups(0.5) if nested else model.param_groups(0.5)
                for group in groups:
                    (group["coupled"] if nested else group).update(
                        prox_freq=4, min_sparsity_schedule=scheduled
                    )
                optimizer = _optimizer(groups, healing=3)
                _step(optimizer, list(model.parameters()), steps=3)
                self.assertEqual(optimizer.num_steps, 3)
                self.assert_shared_mask(model, 0.5)

    def test_healing_dense_gradients_and_momentum_cannot_revive_mask(self):
        for latent in (False, True):
            model = ToyBlockStack()
            params = list(model.parameters())
            optimizer = _optimizer(
                model.nested_groups(0.5),
                lr=0.1,
                healing=2,
                latent_weights=latent,
                momentum=0.9,
            )
            _step(optimizer, params, steps=2, dense_gradient=True)
            frozen = self.assert_shared_mask(model, 0.5)
            masks = [p.eq(0) for p in params]
            self.assertTrue(
                any(
                    optimizer.state[p]["momentum_buffer"][mask].ne(0).any()
                    for p, mask in zip(params, masks)
                )
            )
            _step(optimizer, params, steps=4, dense_gradient=True)
            self.assertEqual(self.assert_shared_mask(model, 0.5), frozen)
            for p, mask in zip(params, masks):
                self.assertTrue(p[mask].eq(0).all())

    def test_composition_order_and_literal_metrics_with_existing_individual_zero(self):
        # Removing row 0 changes the lowest channel from 0 to 1.
        for prox_type in (
            "MinSparsityConstraint",
            "GlobalMinSparsityConstraint",
            "ProxGroupLasso",
        ):
            for latent in (False, True):
                p = torch.nn.Parameter(
                    torch.tensor(
                        [
                            [0.01, 3.0, 0.01, 0.01],
                            [1.0, 0.1, 0.0, 3.0],
                            [1.0, 0.1, 2.0, 3.0],
                            [1.0, 0.1, 2.0, 3.0],
                        ]
                    )
                )
                other = torch.nn.Parameter(torch.ones(2, 2))
                group = dict(
                    params=[p],
                    prox_type=prox_type,
                    group_type="Dim0Grouper",
                    min_sparsity=0.25,
                    reg_lambda=1.525 if prox_type == "ProxGroupLasso" else 0.0,
                    coupled=dict(
                        group_type="Dim1Grouper", couple_key="resid", min_sparsity=0.25
                    ),
                )
                untouched = dict(
                    params=[other],
                    prox_type="MinSparsityConstraint",
                    group_type="Dim0Grouper",
                    min_sparsity=0.0,
                )
                optimizer = _optimizer(
                    [group, untouched], lr=1.0, latent_weights=latent
                )
                _step(optimizer, [p, other])
                self.assertEqual(_zero_channels(p, 0), {0})
                self.assertEqual(_zero_channels(p, 1), {1})
                self.assertEqual(p[1, 2], 0.0)
                self.assert_literal_metrics(optimizer, [p, other])
                self.assertAlmostEqual(optimizer.relative_sparsity, 8 / 20)

    def test_global_heads_with_head_dim_and_nested_residual(self):
        for latent in (False, True):
            model = ToyBlockStack()
            groups = []
            for params, axis, coupled in ((model.wv, 0, 1), (model.wo, 1, 0)):
                groups.append(
                    dict(
                        params=list(params),
                        group_type=f"AttentionHeadGrouperDim{axis}",
                        head_dim=2,
                        prox_type="GlobalMinSparsityConstraint",
                        min_sparsity=0.25,
                        coupled=dict(
                            group_type=f"Dim{coupled}Grouper",
                            couple_key="resid",
                            min_sparsity=0.5,
                        ),
                    )
                )
            groups += [
                _coupled_group([model.tok_embeddings, *model.w1], 0.5, "Dim1Grouper"),
                _coupled_group(model.w2, 0.5, "Dim0Grouper"),
            ]
            optimizer = _optimizer(groups, latent_weights=latent)
            _step(optimizer, list(model.parameters()), steps=2)
            self.assert_shared_mask(model, 0.5)
            for params, axis in ((model.wv, 0), (model.wo, 1)):
                zero_heads = sum(
                    len(
                        _zero_channels(
                            (p if axis == 0 else p.transpose(0, 1)).reshape(
                                INNER // 2, -1
                            ),
                            0,
                        )
                    )
                    for p in params
                )
                self.assertGreaterEqual(zero_heads, 1)
            self.assert_literal_metrics(optimizer, list(model.parameters()))


class TestCoupledDTensor(
    DistributedTestMixin, _CoupledAssertions, common_utils.TestCase
):
    @torch.no_grad()
    def test_world_size_one_multishard_fallback_matches_dense(self):
        model = ToyBlockStack()
        dense_specs = [
            (p.detach().clone(), cls, kwargs) for p, cls, kwargs in model.specs()
        ]
        specs = [
            (distribute_tensor(p.clone(), self.mesh, [Shard(0), Shard(0)]), cls, kwargs)
            for p, cls, kwargs in dense_specs
        ]
        prox = CoupledMinSparsityConstraint(0.0, 0.5)
        dense = apply_coupled_prox(dense_specs, prox, 0.5)
        with mock.patch(f"{EXECUTOR}.grouped_view", wraps=grouped_view) as view:
            result = apply_coupled_prox(specs, prox, 0.5)
        self.assertEqual(view.call_count, len(specs))
        self.assertEqual(
            (result.zero_elts, result.numel, result.zero_channels, result.channels),
            (dense.zero_elts, dense.numel, dense.zero_channels, dense.channels),
        )
        for record, (p, _, _), (want, _, _) in zip(
            result.parameters, specs, dense_specs
        ):
            self.assertIs(record.parameter, p)
            self.assertEqual(p.full_tensor(), want)
            self.assertEqual(record.zero_elts, want.eq(0).sum().item())

    def test_nonparticipant_skips_before_grouping(self):
        p = distribute_tensor(torch.randn(4, 2), self.mesh, [Shard(0), Shard(0)])
        original = p.to_local().clone()
        with (
            mock.patch.object(p.device_mesh, "get_coordinate", return_value=None),
            mock.patch(
                f"{EXECUTOR}.grouped_view",
                side_effect=AssertionError("unexpected grouping"),
            ),
        ):
            result = apply_coupled_prox(
                [(p, Dim0Grouper, {})], CoupledMinSparsityConstraint(0.0, 0.5), 0.5
            )
        self.assertEqual(result.parameters, ())
        self.assertEqual(
            (result.zero_elts, result.numel, result.zero_channels, result.channels),
            (0, 0, 0, 0),
        )
        self.assertEqual(p.to_local(), original)

    def test_mixed_dense_or_mesh_rejection_leaves_originals_unchanged(self):
        p = distribute_tensor(torch.randn(4, 2), self.mesh, [Shard(0), Shard(0)])
        different_mesh = init_device_mesh("cpu", (1,))
        alternatives = [
            torch.randn(4, 2),
            distribute_tensor(torch.randn(4, 2), different_mesh, [Shard(0)]),
        ]
        for other, message in zip(
            alternatives, ("cannot mix dense", "same DeviceMesh")
        ):
            original = p.to_local().clone()
            other_original = (
                other.to_local() if hasattr(other, "to_local") else other
            ).clone()
            with self.assertRaisesRegex(ValueError, message):
                apply_coupled_prox(
                    [(p, Dim0Grouper, {}), (other, Dim0Grouper, {})],
                    CoupledMinSparsityConstraint(0.0, 0.5),
                    0.5,
                )
            self.assertEqual(p.to_local(), original)
            self.assertEqual(
                other.to_local() if hasattr(other, "to_local") else other,
                other_original,
            )


if __name__ == "__main__":
    unittest.main()
