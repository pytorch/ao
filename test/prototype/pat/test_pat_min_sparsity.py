# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import sys
import unittest

import torch
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import (
    DistributedTestMixin,
    TwoLayerMLP,
    make_prox_kwargs,
    optim_step,
)
from torchao.prototype.pat.group import Dim0Grouper, Dim1Grouper, KElementGrouper
from torchao.prototype.pat.optim import MinSparsityConstraint, PruneOptimizer
from torchao.prototype.pat.utils import get_param_groups


class TestMinSparsityConstraint(common_utils.TestCase):
    """Direct unit tests for MinSparsityConstraint.apply_ on a 2-D view."""

    @common_utils.parametrize(
        "M,min_sparsity",
        [(4, 0.0), (4, 0.5), (5, 0.6), (4, 1.0)],
    )
    def test_ranking_by_l2_norm(self, M, min_sparsity):
        """Rows with strictly increasing per-row L2 norms; the smallest
        ceil(min_sparsity * M) rows must be zeroed and survivors intact.
        Covers boundary values (sp=0 leaves p untouched, sp=1 zeros all)."""
        N = 4
        p = torch.zeros(M, N)
        for i in range(M):
            p[i] = float(i + 1)
        original = p.clone()
        prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        zeros_count, group_norm = prox.apply_(p, gamma=1.0)

        n_killed = math.ceil(min_sparsity * M)
        self.assertEqual(zeros_count.item(), n_killed * N)
        for i in range(n_killed):
            self.assertTrue(p[i].eq(0).all(), f"row {i} should be zero")
        for i in range(n_killed, M):
            self.assertTrue(torch.equal(p[i], original[i]), f"row {i} should be intact")
        if min_sparsity == 1.0:
            self.assertEqual(group_norm.item(), 0.0)

    def test_assert_2d_view(self):
        """Non-2-D input raises a clear assertion."""
        p = torch.randn(4, 5, 3)
        prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=0.5)
        with self.assertRaises(AssertionError):
            prox.apply_(p, gamma=1.0)

    @common_utils.parametrize("min_sparsity", [-0.1, 1.1])
    def test_invalid_min_sparsity(self, min_sparsity):
        """Out-of-range min_sparsity values are rejected at construction."""
        with self.assertRaises(AssertionError):
            MinSparsityConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)


class TestMinSparsityWithGrouper(common_utils.TestCase):
    """Integration: MinSparsityConstraint + grouper via _apply_prox."""

    @common_utils.parametrize("grouper_cls,axis", [(Dim0Grouper, 0), (Dim1Grouper, 1)])
    def test_dim_grouper(self, grouper_cls, axis):
        """Dim{0,1}Grouper kills whole rows/columns. Dim1 exercises the
        whole_tensor transpose branch in PruneOptimizer._apply_prox."""
        torch.manual_seed(0)
        M, N, min_sparsity = 8, 4, 0.5
        n_total = M if axis == 0 else N
        p = torch.zeros(M, N)
        for i in range(n_total):
            if axis == 0:
                p[i] = float(i + 1)
            else:
                p[:, i] = float(i + 1)
        original = p.clone()

        prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        grouper = grouper_cls(p)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        zero_elts, _, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox, p, **prox_kwargs
        )

        n_killed = math.ceil(min_sparsity * n_total)
        view = p if axis == 0 else p.transpose(0, 1)
        orig_view = original if axis == 0 else original.transpose(0, 1)
        for i in range(n_killed):
            self.assertTrue(view[i].eq(0).all(), f"slice {i} should be zero")
        for i in range(n_killed, n_total):
            self.assertTrue(
                torch.equal(view[i], orig_view[i]), f"slice {i} should be intact"
            )
        self.assertTrue(zeros_are_summed)
        group_size = N if axis == 0 else M
        self.assertEqual(zero_elts, n_killed * group_size)

    def test_with_layer_grouper_global_magnitude(self):
        """KElementGrouper(k=1) reshapes to (numel, 1) so per-row L2 collapses
        to per-element |x| -- global magnitude pruning over the whole tensor."""
        torch.manual_seed(0)
        M, N, min_sparsity = 4, 5, 0.4
        p = torch.randn(M, N)
        original = p.clone()
        prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)

        grouper = KElementGrouper(p, k=1)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        zero_elts, _, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox, p, **prox_kwargs
        )

        n_zero = math.ceil(min_sparsity * p.numel())
        self.assertEqual(p.eq(0).sum().item(), n_zero)
        self.assertEqual(zero_elts, n_zero)
        # Survivors are exactly the top |x| from the original tensor.
        sorted_abs = original.abs().flatten().sort().values
        threshold = sorted_abs[n_zero - 1].item()
        survivors = p[p.ne(0)].abs()
        self.assertTrue((survivors > threshold).all())
        self.assertTrue(zeros_are_summed)


class TestMinSparsityDTensor(DistributedTestMixin, common_utils.TestCase):
    """Whole-tensor DTensor branch in PruneOptimizer._apply_prox: gather,
    mutate, scatter. Covers both Dim0Grouper (no transpose) and Dim1Grouper
    (transpose round-trip)."""

    @common_utils.parametrize("grouper_cls,axis", [(Dim0Grouper, 0), (Dim1Grouper, 1)])
    def test_dim_grouper_dtensor(self, grouper_cls, axis):
        torch.manual_seed(0)
        M, N, min_sparsity = 8, 4, 0.5
        n_total = M if axis == 0 else N

        p = torch.zeros(M, N)
        for i in range(n_total):
            if axis == 0:
                p[i] = float(i + 1)
            else:
                p[:, i] = float(i + 1)
        original = p.clone()

        p_dt = distribute_tensor(
            p, device_mesh=self.mesh, placements=[Shard(0)] * self.mesh.ndim
        )
        prox = MinSparsityConstraint(reg_lambda=0.0, min_sparsity=min_sparsity)
        grouper = grouper_cls(p_dt)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        _, _, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox, p_dt, **prox_kwargs
        )

        result = p_dt.full_tensor()
        n_killed = math.ceil(min_sparsity * n_total)
        view = result if axis == 0 else result.transpose(0, 1)
        orig_view = original if axis == 0 else original.transpose(0, 1)
        for i in range(n_killed):
            self.assertTrue(view[i].eq(0).all(), f"slice {i} should be zero")
        for i in range(n_killed, n_total):
            self.assertTrue(
                torch.equal(view[i], orig_view[i]), f"slice {i} should be intact"
            )
        self.assertTrue(zeros_are_summed)


class TestMinSparsityOptimizer(common_utils.TestCase):
    """End-to-end PruneOptimizer tests against the shipped configs."""

    def test_with_dim0_grouper_e2e(self):
        torch.manual_seed(42)
        min_sparsity, total_steps = 0.5, 5
        model = TwoLayerMLP(input_size=8, output_size=2)
        prune_config = {
            (torch.nn.Linear, "weight"): {
                "group_type": "Dim0Grouper",
                "prox_type": "MinSparsityConstraint",
                "min_sparsity": min_sparsity,
            }
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=0.0,
        )

        dummy_input = torch.randn(total_steps, 8)
        label = torch.randint(0, 2, (total_steps,))
        for step in range(total_steps):
            optim_step(model, optimizer, dummy_input, label, step)

        for group in optimizer.regularized_param_groups():
            for p in group["params"]:
                flat = p.data.view(p.size(0), -1)
                M = flat.size(0)
                expected_killed = math.ceil(min_sparsity * M)
                row_is_zero = sum(flat[i].eq(0).all().item() for i in range(M))
                self.assertEqual(
                    row_is_zero,
                    expected_killed,
                    f"Expected {expected_killed} rows zeroed, got {row_is_zero}",
                )

        self.assertGreaterEqual(optimizer.relative_sparsity, min_sparsity)

    def test_with_conv_filter_grouper(self):
        """ConvFilterGrouper + MinSparsityConstraint zeroes whole filter slices."""
        torch.manual_seed(42)
        min_sparsity, total_steps = 0.5, 3

        class TinyConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 6, kernel_size=3)

            def forward(self, x):
                return self.conv(x).flatten(1)

        model = TinyConv()
        prune_config = {
            (torch.nn.Conv2d, "weight"): {
                "group_type": "ConvFilterGrouper",
                "prox_type": "MinSparsityConstraint",
                "min_sparsity": min_sparsity,
            }
        }
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=0.0,
        )

        for step in range(total_steps):
            x = torch.randn(1, 4, 5, 5)
            output = model(x)
            label = torch.randint(0, output.size(1), (1,))
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        w = model.conv.weight.data  # (C_out, C_in, kH, kW)
        c_out, c_in, kH, kW = w.shape
        flat = w.view(c_out * c_in, kH * kW)
        n_slices = c_out * c_in
        expected_killed = math.ceil(min_sparsity * n_slices)
        slice_is_zero = sum(flat[i].eq(0).all().item() for i in range(n_slices))
        self.assertEqual(slice_is_zero, expected_killed)


class TestMinSparsitySchedule(common_utils.TestCase):
    """Cubic min_sparsity ramp from 0 -> target over (warmup, healing_start)."""

    def _make_optimizer(self, schedule, warmup, healing, target=0.8):
        torch.manual_seed(0)
        model = TwoLayerMLP(input_size=8, output_size=2)
        cfg = {
            (torch.nn.Linear, "weight"): {
                "group_type": "Dim0Grouper",
                "prox_type": "MinSparsityConstraint",
                "min_sparsity": target,
            }
        }
        if schedule:
            cfg[(torch.nn.Linear, "weight")]["min_sparsity_schedule"] = True
        param_groups = get_param_groups(model, cfg, verbose=False)
        opt = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            warmup_steps=warmup,
            healing_start_step=healing,
            reg_lambda=0.0,
        )
        return model, opt

    def test_disabled_returns_target(self):
        _, opt = self._make_optimizer(schedule=False, warmup=2, healing=10, target=0.8)
        opt.num_steps = 5
        for g in opt.regularized_param_groups():
            self.assertEqual(opt._effective_min_sparsity(g), 0.8)

    def test_cubic_endpoints_and_midpoint(self):
        warmup, healing, target = 4, 12, 0.8
        _, opt = self._make_optimizer(True, warmup, healing, target)
        g = next(opt.regularized_param_groups())
        # Pre-warmup -> 0
        opt.num_steps = 0
        self.assertEqual(opt._effective_min_sparsity(g), 0.0)
        opt.num_steps = warmup
        self.assertEqual(opt._effective_min_sparsity(g), 0.0)
        # Post-healing -> target
        opt.num_steps = healing
        self.assertEqual(opt._effective_min_sparsity(g), target)
        opt.num_steps = healing + 5
        self.assertEqual(opt._effective_min_sparsity(g), target)
        # Midpoint t=0.5 -> target * (1 - 0.5^3) = target * 0.875
        opt.num_steps = warmup + (healing - warmup) // 2
        self.assertAlmostEqual(
            opt._effective_min_sparsity(g), target * (1 - 0.5**3), places=6
        )

    def test_schedule_requires_finite_healing(self):
        with self.assertRaises(AssertionError):
            self._make_optimizer(True, warmup=2, healing=sys.maxsize, target=0.5)

    def test_resume_via_patch_state_dict(self):
        """Drive opt_a forward with real step() calls, snapshot via state_dict(),
        then load_state_dict() + patch_state_dict() into a fresh opt_b and
        verify the schedule state survives the checkpoint round-trip."""
        warmup, healing, target = 2, 10, 0.6
        model_a, opt_a = self._make_optimizer(True, warmup, healing, target)
        # 5 real step() calls land us mid-ramp (warmup < n < healing).
        n_steps = 5
        dummy_input = torch.randn(n_steps + 3, 8)
        label = torch.randint(0, 2, (n_steps + 3,))
        for s in range(n_steps):
            optim_step(model_a, opt_a, dummy_input, label, s)
        self.assertGreater(opt_a.num_steps, warmup)
        self.assertLess(opt_a.num_steps, healing)

        # Snapshot, then load + patch into a fresh optimizer.
        sd = opt_a.state_dict()
        model_b, opt_b = self._make_optimizer(True, warmup, healing, target)
        opt_b.load_state_dict(sd)
        opt_b.patch_state_dict(sd)

        self.assertEqual(opt_b.num_steps, opt_a.num_steps)
        g_a = next(opt_a.regularized_param_groups())
        g_b = next(opt_b.regularized_param_groups())
        self.assertEqual(
            opt_a._effective_min_sparsity(g_a),
            opt_b._effective_min_sparsity(g_b),
        )

        # Continue training both. Schedule state must stay in sync, including
        # across the healing boundary where the ramp clamps to target.
        for s in range(n_steps, n_steps + 3):
            optim_step(model_a, opt_a, dummy_input, label, s)
            optim_step(model_b, opt_b, dummy_input, label, s)
            self.assertEqual(opt_b.num_steps, opt_a.num_steps)
            self.assertEqual(
                opt_a._effective_min_sparsity(g_a),
                opt_b._effective_min_sparsity(g_b),
            )


common_utils.instantiate_parametrized_tests(TestMinSparsityConstraint)
common_utils.instantiate_parametrized_tests(TestMinSparsityWithGrouper)
common_utils.instantiate_parametrized_tests(TestMinSparsityDTensor)
common_utils.instantiate_parametrized_tests(TestMinSparsityOptimizer)
common_utils.instantiate_parametrized_tests(TestMinSparsitySchedule)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
