# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import unittest
from unittest.mock import patch

import torch
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import TwoLayerMLP, make_prox_kwargs, optim_step
from torchao.prototype.pat.group import PackedSVDGrouper, SVDGrouper
from torchao.prototype.pat.optim import MinRankConstraint, PruneOptimizer
from torchao.prototype.pat.optim.prox_executor import apply_prox
from torchao.prototype.pat.utils import get_param_groups


class TestMinRankConstraintApply(common_utils.TestCase):
    """Direct tests for MinRankConstraint on singular-value tensors."""

    @common_utils.parametrize(
        "k,min_sparsity,n_killed",
        [(5, 0.0, 0), (5, 0.4, 2), (5, 0.5, 3), (5, 1.0, 5)],
    )
    def test_zeros_smallest_singular_values(self, k, min_sparsity, n_killed):
        p = torch.arange(1, k + 1, dtype=torch.float32)
        prox = MinRankConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        zeros_count, group_norm = prox.apply_(p, gamma=1.0)

        self.assertEqual(zeros_count.item(), n_killed)
        self.assertTrue(p[:n_killed].eq(0).all())
        expected_survivors = torch.arange(n_killed + 1, k + 1, dtype=torch.float32)
        self.assertEqual(p[n_killed:], expected_survivors)
        if min_sparsity == 1.0:
            self.assertEqual(group_norm.item(), 0.0)

    def test_packed_singular_values_are_pruned_per_matrix(self):
        npack, k, min_sparsity = 3, 6, 0.5
        p = torch.arange(1, k + 1, dtype=torch.float32).repeat(npack, 1)
        prox = MinRankConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        zeros_count, _ = prox.apply_(p, gamma=1.0)

        n_killed = math.ceil(min_sparsity * k)
        self.assertEqual(zeros_count.item(), npack * n_killed)
        self.assertTrue(p[:, :n_killed].eq(0).all())
        expected_survivors = torch.arange(
            n_killed + 1, k + 1, dtype=torch.float32
        ).expand(npack, -1)
        self.assertEqual(p[:, n_killed:], expected_survivors)

    @common_utils.parametrize("min_sparsity", [-0.1, 1.1])
    def test_invalid_min_sparsity(self, min_sparsity):
        with self.assertRaises(AssertionError):
            MinRankConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)


class TestMinRankWithSVDGrouper(common_utils.TestCase):
    """Integration tests for MinRankConstraint with SVD groupers."""

    @staticmethod
    def _effective_rank(weight):
        singular_values = torch.linalg.svdvals(weight.to(torch.float32))
        return int((singular_values > 1e-5).sum().item())

    @common_utils.parametrize("min_sparsity", [0.25, 0.5, 0.75])
    def test_svd_grouper(self, min_sparsity):
        torch.manual_seed(0)
        in_dim, out_dim = 64, 32
        k = min(in_dim, out_dim)
        model = torch.nn.Linear(in_dim, out_dim)
        prox = MinRankConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        grouper = SVDGrouper(model.weight)
        prox_kwargs = make_prox_kwargs(
            gamma=1.0, zero_elts_are_counts=True, is_svd_grouper=True
        )
        zero_elts, _, zeros_are_summed = apply_prox(
            grouper,
            prox,
            model.weight,
            sv_count=torch.zeros(1, dtype=torch.int),
            **prox_kwargs,
        )

        n_killed = math.ceil(min_sparsity * k)
        self.assertTrue(zeros_are_summed)
        self.assertEqual(zero_elts, n_killed)
        self.assertEqual(self._effective_rank(model.weight), k - n_killed)

    def test_packed_svd_grouper(self):
        torch.manual_seed(0)
        embed_dim, npack, min_sparsity = 16, 3, 0.5
        model = torch.nn.Linear(embed_dim, embed_dim * npack)
        prox = MinRankConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        grouper = PackedSVDGrouper(model.weight, npack, pack_dim=0)
        prox_kwargs = make_prox_kwargs(
            gamma=1.0, zero_elts_are_counts=True, is_svd_grouper=True
        )
        sv_count = torch.zeros(npack, dtype=torch.int)
        zero_elts, _, _ = apply_prox(
            grouper,
            prox,
            model.weight,
            sv_count=sv_count,
            **prox_kwargs,
        )

        n_killed_per_pack = math.ceil(min_sparsity * embed_dim)
        retained_per_pack = embed_dim - n_killed_per_pack
        self.assertEqual(zero_elts, npack * n_killed_per_pack)
        self.assertEqual(sv_count, torch.full_like(sv_count, retained_per_pack))
        for packed_weight in model.weight.chunk(npack, dim=0):
            self.assertEqual(
                self._effective_rank(packed_weight), embed_dim - n_killed_per_pack
            )


class TestProxFreqGate(common_utils.TestCase):
    def test_prox_freq_gates_svd(self):
        torch.manual_seed(0)
        n_steps, prox_freq = 16, 4
        model = TwoLayerMLP(input_size=8, output_size=2)
        prune_config = {
            (torch.nn.Linear, "weight"): {
                "group_type": "SVDGrouper",
                "prox_type": "MinRankConstraint",
                "min_sparsity": 0.5,
                "prox_freq": prox_freq,
            }
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            warmup_steps=0,
            reg_lambda=0.0,
        )
        dummy_input = torch.randn(n_steps, 8)
        label = torch.randint(0, 2, (n_steps,))

        enter_count = 0
        real_enter = SVDGrouper.__enter__

        def counting_enter(grouper):
            nonlocal enter_count
            enter_count += 1
            return real_enter(grouper)

        with patch.object(SVDGrouper, "__enter__", counting_enter):
            for step in range(n_steps):
                optim_step(model, optimizer, dummy_input, label, step)

        n_params = sum(
            len(group["params"]) for group in optimizer.regularized_param_groups()
        )
        expected_per_param = n_steps // prox_freq
        self.assertEqual(enter_count, expected_per_param * n_params)

    def test_mixed_frequencies_keep_complete_metrics(self):
        torch.manual_seed(0)
        model = TwoLayerMLP(input_size=8, output_size=2)
        prune_config = {
            "fc1.weight": {
                "group_type": "SVDGrouper",
                "prox_type": "MinRankConstraint",
                "min_sparsity": 0.5,
                "prox_freq": 2,
            },
            "fc2.weight": {
                "group_type": "SVDGrouper",
                "prox_type": "MinRankConstraint",
                "min_sparsity": 0.5,
                "prox_freq": 3,
            },
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            warmup_steps=0,
            reg_lambda=0.0,
        )
        dummy_input = torch.randn(3, 8)
        label = torch.randint(0, 2, (3,))

        optim_step(model, optimizer, dummy_input, label, 0)
        relative_sparsity = optimizer.relative_sparsity
        relative_factored_frac = optimizer.relative_factored_frac

        # At step 2 only fc1 runs. The global metrics must retain the last
        # complete aggregate instead of reporting the running subset alone.
        optim_step(model, optimizer, dummy_input, label, 1)
        optim_step(model, optimizer, dummy_input, label, 2)
        self.assertEqual(optimizer.relative_sparsity, relative_sparsity)
        self.assertEqual(optimizer.relative_factored_frac, relative_factored_frac)


class TestProxThroughHeal(common_utils.TestCase):
    @staticmethod
    def _make_optimizer(group_type, prox_type, prox_through_heal=None):
        param = torch.nn.Parameter(torch.randn(4, 4))
        group = {
            "params": [param],
            "group_type": group_type,
            "prox_type": prox_type,
        }
        if prox_type == "MinRankConstraint":
            group["min_sparsity"] = 0.5
        if prox_through_heal is not None:
            group["prox_through_heal"] = prox_through_heal
        return PruneOptimizer(torch.optim.SGD([group], lr=0.1))

    def test_non_svd_opt_in_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "prox_through_heal=True requires an SVD grouper"
        ):
            self._make_optimizer("Dim0Grouper", "ProxGroupLasso", True)

    def test_non_svd_opt_in_mutation_is_rejected(self):
        optimizer = self._make_optimizer("Dim0Grouper", "ProxGroupLasso")
        group = next(optimizer.regularized_param_groups())
        group["prox_through_heal"] = True
        with self.assertRaisesRegex(
            ValueError, "prox_through_heal=True requires an SVD grouper"
        ):
            optimizer._prox_through_heal(group)

    @common_utils.parametrize(
        "prox_type,override,expected",
        [
            ("MinRankConstraint", None, True),
            ("ProxNuclearNorm", None, False),
            ("ProxNuclearNorm", True, True),
            ("MinRankConstraint", False, False),
        ],
    )
    def test_svd_policy_defaults_and_overrides(self, prox_type, override, expected):
        optimizer = self._make_optimizer("SVDGrouper", prox_type, override)
        group = next(optimizer.regularized_param_groups())
        self.assertEqual(optimizer._prox_through_heal(group), expected)

    def test_singular_values_stay_zero_during_healing(self):
        torch.manual_seed(0)
        warmup, healing_start, n_steps = 2, 6, 12
        min_sparsity = 0.5
        model = TwoLayerMLP(input_size=8, output_size=2)
        prune_config = {
            (torch.nn.Linear, "weight"): {
                "group_type": "SVDGrouper",
                "prox_type": "MinRankConstraint",
                "min_sparsity": min_sparsity,
            }
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            warmup_steps=warmup,
            healing_start_step=healing_start,
            reg_lambda=0.0,
        )
        dummy_input = torch.randn(n_steps, 8)
        label = torch.randint(0, 2, (n_steps,))

        for step in range(n_steps):
            optim_step(model, optimizer, dummy_input, label, step)

        for layer in (model.fc1, model.fc2):
            k = min(layer.weight.shape)
            singular_values = torch.linalg.svdvals(
                layer.weight.detach().to(torch.float32)
            )
            n_killed = math.ceil(min_sparsity * k)
            effective_rank = int((singular_values > 1e-5).sum().item())
            self.assertLessEqual(effective_rank, k - n_killed)


class TestPackedFactorizationMetrics(common_utils.TestCase):
    @common_utils.parametrize("pack_dim", [0, 1])
    def test_full_step_uses_total_packed_storage(self, pack_dim):
        torch.manual_seed(0)
        embed_dim, npack, min_sparsity = 8, 3, 0.75
        shape = (
            (embed_dim * npack, embed_dim)
            if pack_dim == 0
            else (embed_dim, embed_dim * npack)
        )
        param = torch.nn.Parameter(torch.randn(shape))
        group = {
            "params": [param],
            "group_type": "PackedSVDGrouper",
            "prox_type": "MinRankConstraint",
            "min_sparsity": min_sparsity,
            "npack": npack,
            "pack_dim": pack_dim,
        }
        optimizer = PruneOptimizer(torch.optim.SGD([group], lr=0.1))

        param.grad = torch.zeros_like(param)
        optimizer.step()

        retained_per_pack = embed_dim - math.ceil(min_sparsity * embed_dim)
        retained_total = npack * retained_per_pack
        factored_size = (embed_dim + embed_dim) * retained_total
        expected_frac = factored_size / param.numel()
        self.assertEqual(optimizer.param_groups[0]["factored_frac"], expected_frac)
        self.assertEqual(optimizer.relative_factored_frac, expected_frac)


common_utils.instantiate_parametrized_tests(TestMinRankConstraintApply)
common_utils.instantiate_parametrized_tests(TestMinRankWithSVDGrouper)
common_utils.instantiate_parametrized_tests(TestProxFreqGate)
common_utils.instantiate_parametrized_tests(TestProxThroughHeal)
common_utils.instantiate_parametrized_tests(TestPackedFactorizationMetrics)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
