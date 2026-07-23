# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import unittest
from unittest import mock

import torch
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import DistributedTestMixin, optim_step
from torchao.prototype.pat.group import Dim0Grouper, Grouper
from torchao.prototype.pat.optim import GlobalMinSparsityConstraint, PruneOptimizer
from torchao.prototype.pat.optim.prox_executor import apply_global_prox, grouped_view
from torchao.prototype.pat.utils import get_param_groups


def _global_group(params, min_sparsity, score_type="rms", group_type="Dim0Grouper"):
    return {
        "params": list(params),
        "group_type": group_type,
        "prox_type": "GlobalMinSparsityConstraint",
        "min_sparsity": min_sparsity,
        "score_type": score_type,
    }


def _count_zero_groups(p, axis=0):
    flat = p.data if axis == 0 else p.data.transpose(0, 1)
    return sum(flat[i].eq(0).all().item() for i in range(flat.size(0)))


class TestGlobalMinSparsityScore(common_utils.TestCase):
    def test_score_types(self):
        p = torch.randn(6, 4)
        norm = torch.linalg.vector_norm(p, dim=1)
        self.assertTrue(
            torch.allclose(
                GlobalMinSparsityConstraint(0.0, 0.5, "rms").score(p),
                norm / math.sqrt(4),
            )
        )
        self.assertTrue(
            torch.allclose(GlobalMinSparsityConstraint(0.0, 0.5, "l2").score(p), norm)
        )
        self.assertTrue(
            torch.allclose(
                GlobalMinSparsityConstraint(0.0, 0.5, "param_cost").score(p),
                norm / 4,
            )
        )

    def test_invalid_score_type(self):
        with self.assertRaises(AssertionError):
            GlobalMinSparsityConstraint(0.0, 0.5, "bogus")

    def test_zero_groups(self):
        p = torch.arange(24.0).reshape(6, 4).clone()
        idx = torch.tensor([1, 3, 4])
        zeros = GlobalMinSparsityConstraint.zero_groups_(p, idx)
        self.assertEqual(int(zeros), 3 * 4)
        for i in range(6):
            if i in (1, 3, 4):
                self.assertTrue(p[i].eq(0).all())
            else:
                self.assertTrue(p[i].ne(0).any())

    def test_zero_groups_empty(self):
        p = torch.randn(5, 4).clone()
        original = p.clone()
        zeros = GlobalMinSparsityConstraint.zero_groups_(
            p, torch.tensor([], dtype=torch.long)
        )
        self.assertEqual(int(zeros), 0)
        self.assertTrue(torch.equal(p, original))

    def test_rejects_mixed_score_devices(self):
        params = [torch.randn(2, 2), torch.empty(2, 2, device="meta")]
        prox = GlobalMinSparsityConstraint(0.0, 0.5)
        with self.assertRaisesRegex(ValueError, "scores on the same device"):
            apply_global_prox(params, prox, Dim0Grouper, {}, min_sparsity=0.5)

    def test_rejects_non_2d_grouped_view(self):
        param = torch.randn(2, 3, 4)
        prox = GlobalMinSparsityConstraint(0.0, 0.5)
        with self.assertRaisesRegex(
            ValueError, "requires a grouper that produces a 2-D"
        ):
            apply_global_prox([param], prox, Grouper, {}, min_sparsity=0.5)


class TestGlobalMinSparsityOptimizer(common_utils.TestCase):
    def _run(self, params, min_sparsity, score_type="rms", steps=3, lr=0.1):
        base = torch.optim.SGD([_global_group(params, min_sparsity, score_type)], lr=lr)
        optimizer = PruneOptimizer(base, warmup_steps=0, healing_start_step=100)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = sum((p * p).sum() for p in params)
            loss.backward()
            optimizer.step()
        return optimizer

    def test_nonuniform_allocation(self):
        torch.manual_seed(0)
        a = torch.nn.Parameter(torch.randn(8, 4) * 10.0)
        b = torch.nn.Parameter(torch.randn(8, 4) * 0.1)
        self._run([a, b], 0.5, lr=0.0, steps=1)
        zero_a = _count_zero_groups(a)
        zero_b = _count_zero_groups(b)
        self.assertEqual(zero_a + zero_b, 8)
        self.assertEqual(zero_a, 0)
        self.assertEqual(zero_b, 8)

    def test_effective_min_sparsity_is_evaluated_once(self):
        a = torch.nn.Parameter(torch.randn(8, 4))
        b = torch.nn.Parameter(torch.randn(8, 4))
        group = _global_group([a, b], 0.5)
        group["min_sparsity_schedule"] = True
        optimizer = PruneOptimizer(
            torch.optim.SGD([group], lr=0.0), healing_start_step=10
        )
        for param in (a, b):
            param.grad = torch.zeros_like(param)

        with mock.patch.object(
            optimizer,
            "_effective_min_sparsity",
            wraps=optimizer._effective_min_sparsity,
        ) as effective_min_sparsity:
            optimizer.step()

        self.assertEqual(effective_min_sparsity.call_count, 1)

    def test_builds_each_grouped_view_once(self):
        a = torch.nn.Parameter(torch.arange(32.0).reshape(8, 4) + 1)
        b = torch.nn.Parameter(torch.arange(32.0).reshape(8, 4) + 33)
        group = _global_group([a, b], 0.5)
        optimizer = PruneOptimizer(torch.optim.SGD([group], lr=0.0))
        for param in (a, b):
            param.grad = torch.zeros_like(param)

        with mock.patch(
            "torchao.prototype.pat.optim.prox_executor.grouped_view",
            wraps=grouped_view,
        ) as grouped_view_mock:
            optimizer.step()

        self.assertEqual(grouped_view_mock.call_count, 2)
        grouped_params = [
            call.args[0]._param for call in grouped_view_mock.call_args_list
        ]
        self.assertEqual(grouped_params, [a, b])

    def test_healing_freezes_nonuniform_mask(self):
        torch.manual_seed(0)
        a = torch.nn.Parameter(torch.randn(8, 4) * 10.0)
        b = torch.nn.Parameter(torch.randn(8, 4) * 0.1)
        base = torch.optim.SGD([_global_group([a, b], 0.5)], lr=0.1)
        optimizer = PruneOptimizer(base, warmup_steps=0, healing_start_step=2)
        for _ in range(6):
            optimizer.zero_grad()
            (a.pow(2).sum() + b.pow(2).sum()).backward()
            optimizer.step()
        self.assertEqual(_count_zero_groups(a) + _count_zero_groups(b), 8)

    @common_utils.parametrize("scheduled", [False, True])
    def test_healing_boundary_overrides_prox_freq(self, scheduled):
        torch.manual_seed(0)
        a = torch.nn.Parameter(torch.randn(8, 4) * 10.0)
        b = torch.nn.Parameter(torch.randn(8, 4) * 0.1)
        group = _global_group([a, b], 0.5)
        group["prox_freq"] = 2
        group["min_sparsity_schedule"] = scheduled
        optimizer = PruneOptimizer(
            torch.optim.SGD([group], lr=0.1),
            warmup_steps=0,
            healing_start_step=2,
        )
        for _ in range(5):
            optimizer.zero_grad()
            (a.pow(2).sum() + b.pow(2).sum()).backward()
            optimizer.step()
        self.assertEqual(_count_zero_groups(a) + _count_zero_groups(b), 8)

    def test_rejects_padded_k_element_groups(self):
        param = torch.nn.Parameter(torch.randn(1, 5))
        group = _global_group([param], 0.5, group_type="KElementGrouper")
        group["k"] = 4
        optimizer = PruneOptimizer(torch.optim.SGD([group], lr=0.0))
        param.grad = torch.zeros_like(param)
        with self.assertRaisesRegex(
            ValueError, "does not support padded KElementGrouper groups"
        ):
            optimizer.step()

    def test_elem_grouper_allocates_element_budget(self):
        a = torch.nn.Parameter(torch.tensor([[10.0, 9.0], [8.0, 7.0]]))
        b = torch.nn.Parameter(torch.tensor([[0.4, 0.3], [0.2, 0.1]]))
        group = _global_group([a, b], 0.5, group_type="ElemGrouper")
        optimizer = PruneOptimizer(torch.optim.SGD([group], lr=0.0))
        for param in (a, b):
            param.grad = torch.zeros_like(param)
        optimizer.step()
        self.assertEqual(torch.count_nonzero(a), 4)
        self.assertEqual(torch.count_nonzero(b), 0)
        self.assertEqual(optimizer.relative_sparsity, 0.5)

    def test_layer_grouper_allocates_layer_budget(self):
        a = torch.nn.Parameter(torch.full((2, 2), 10.0))
        b = torch.nn.Parameter(torch.full((2, 2), 0.1))
        group = _global_group([a, b], 0.5, group_type="LayerGrouper")
        optimizer = PruneOptimizer(torch.optim.SGD([group], lr=0.0))
        for param in (a, b):
            param.grad = torch.zeros_like(param)
        optimizer.step()
        self.assertTrue(a.ne(0).all())
        self.assertTrue(b.eq(0).all())
        self.assertEqual(optimizer.relative_sparsity, 0.5)

    def test_via_get_param_groups(self):
        torch.manual_seed(0)

        class TinyStack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(8, 8, bias=False) for _ in range(3)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = TinyStack()
        prune_config = {
            (torch.nn.Linear, "weight"): {
                "group_type": "Dim0Grouper",
                "prox_type": "GlobalMinSparsityConstraint",
                "min_sparsity": 0.5,
                "score_type": "rms",
            }
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1), reg_lambda=0.0
        )
        dummy = torch.randn(4, 8)
        label = torch.randint(0, 8, (4,))
        for step in range(4):
            optim_step(model, optimizer, dummy, label, step)
        total_groups = sum(layer.weight.size(0) for layer in model.layers)
        total_zero = sum(_count_zero_groups(layer.weight) for layer in model.layers)
        self.assertEqual(total_zero, math.ceil(0.5 * total_groups))
        self.assertAlmostEqual(optimizer.relative_sparsity, 0.5, places=6)


class TestGlobalMinSparsityDTensor(DistributedTestMixin, common_utils.TestCase):
    """PAT CI exercises the DTensor API at world size one, not true multi-rank."""

    def test_rejects_mixed_dense_and_dtensor_group(self):
        dense = torch.randn(8, 4)
        dtensor = distribute_tensor(
            torch.randn(8, 4), self.mesh, [Shard(0)] * self.mesh.ndim
        )
        prox = GlobalMinSparsityConstraint(0.0, 0.5)
        with self.assertRaisesRegex(ValueError, "cannot mix dense tensors"):
            apply_global_prox([dense, dtensor], prox, Dim0Grouper, {}, min_sparsity=0.5)

    def test_nonparticipant_skips_before_grouping(self):
        dtensor = distribute_tensor(
            torch.randn(8, 4), self.mesh, [Shard(0)] * self.mesh.ndim
        )
        prox = GlobalMinSparsityConstraint(0.0, 0.5)
        with (
            mock.patch.object(dtensor.device_mesh, "get_coordinate", return_value=None),
            mock.patch(
                "torchao.prototype.pat.optim.prox_executor.grouped_view",
                side_effect=AssertionError("unexpected grouped view"),
            ),
        ):
            result = apply_global_prox(
                [dtensor], prox, Dim0Grouper, {}, min_sparsity=0.5
            )
        self.assertEqual(result.parameters, ())
        self.assertEqual(result.zero_elts, 0)
        self.assertEqual(result.numel, 0)

    def test_two_dtensors_match_dense_result(self):
        torch.manual_seed(0)
        a = torch.randn(8, 4) * 10.0
        b = torch.randn(8, 4) * 0.1
        a_dense = torch.nn.Parameter(a.clone())
        b_dense = torch.nn.Parameter(b.clone())
        dense_optimizer = PruneOptimizer(
            torch.optim.SGD([_global_group([a_dense, b_dense], 0.5)], lr=0.0),
            warmup_steps=0,
            healing_start_step=100,
        )
        for param in (a_dense, b_dense):
            param.grad = torch.zeros_like(param)
        dense_optimizer.step()

        a_dt = torch.nn.Parameter(
            distribute_tensor(a, self.mesh, [Shard(0)] * self.mesh.ndim)
        )
        b_dt = torch.nn.Parameter(
            distribute_tensor(b, self.mesh, [Shard(0)] * self.mesh.ndim)
        )
        optimizer = PruneOptimizer(
            torch.optim.SGD([_global_group([a_dt, b_dt], 0.5)], lr=0.0),
            warmup_steps=0,
            healing_start_step=100,
        )
        optimizer.zero_grad()
        (a_dt.to_local().pow(2).sum() + b_dt.to_local().pow(2).sum()).backward()
        optimizer.step()

        full_a = a_dt.full_tensor()
        full_b = b_dt.full_tensor()
        self.assertEqual(full_a, a_dense)
        self.assertEqual(full_b, b_dense)
        self.assertEqual(_count_zero_groups(full_a) + _count_zero_groups(full_b), 8)


common_utils.instantiate_parametrized_tests(TestGlobalMinSparsityScore)
common_utils.instantiate_parametrized_tests(TestGlobalMinSparsityOptimizer)
common_utils.instantiate_parametrized_tests(TestGlobalMinSparsityDTensor)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
