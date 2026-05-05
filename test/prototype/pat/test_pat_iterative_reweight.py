# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import unittest

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import (
    DistributedTestMixin,
    TwoLayerMLP,
    make_prox_kwargs,
    optim_step,
)
from torchao.prototype.pat.group import Dim0Grouper, Dim1Grouper
from torchao.prototype.pat.group.grouper import ElemGrouper
from torchao.prototype.pat.optim import ProxGroupLasso, ProxLasso, PruneOptimizer
from torchao.prototype.pat.optim.iterative_reweight import IterativeReweight
from torchao.prototype.pat.utils import get_param_groups


class TestIterativeReweight(common_utils.TestCase):
    """Unit tests for the IterativeReweight class directly."""

    def test_reweight_formula(self):
        """1 / (group_norm / sigma + eps) produces correct values."""
        eps = 1e-3
        reweight = IterativeReweight(reweight_freq=1, eps=eps)
        group_norm = torch.tensor([1.0, 2.0, 4.0])
        sigma = torch.tensor([2.0, 2.0, 2.0])

        result = reweight(group_norm.clone(), sigma.clone())
        expected = 1.0 / (torch.tensor([1.0, 2.0, 4.0]) / (sigma + eps) + eps)
        self.assertEqual(result, expected)

    def test_small_norm_high_reweight(self):
        """Groups with norm << sigma get large tau_reweight."""
        reweight = IterativeReweight(reweight_freq=1, eps=1e-3)
        sigma = torch.tensor([1.0, 1.0])
        small_norm = torch.tensor([0.001, 0.001])
        large_norm = torch.tensor([10.0, 10.0])

        rw_small = reweight(small_norm.clone(), sigma.clone())
        rw_large = reweight(large_norm.clone(), sigma.clone())
        self.assertTrue((rw_small > rw_large).all())

    def test_eps_prevents_division_by_zero(self):
        """When group_norm == 0, result is 1/eps."""
        eps = 1e-3
        reweight = IterativeReweight(reweight_freq=1, eps=eps)
        group_norm = torch.tensor([0.0])
        sigma = torch.tensor([1.0])

        result = reweight(group_norm.clone(), sigma.clone())
        self.assertAlmostEqual(result.item(), 1.0 / eps, places=1)

    def test_should_update_at_end_step(self):
        """True when step == end_step and on-frequency; False one step later."""
        rw = IterativeReweight(reweight_freq=2, reweight_end_step=6)
        self.assertTrue(rw.should_update(6))
        self.assertFalse(rw.should_update(7))

    def test_should_update_past_end_step(self):
        """Updates at steps 0..end_step, stops after."""
        rw = IterativeReweight(reweight_freq=1, reweight_end_step=3)
        for step in range(4):
            self.assertTrue(rw.should_update(step), f"step={step}")
        for step in range(4, 7):
            self.assertFalse(rw.should_update(step), f"step={step}")

    def test_should_update_step_zero_with_freq_gt_one(self):
        """Step 0 is always on-frequency (0 % freq == 0)."""
        rw = IterativeReweight(reweight_freq=3, reweight_end_step=100)
        self.assertTrue(rw.should_update(0))
        self.assertFalse(rw.should_update(1))
        self.assertTrue(rw.should_update(3))


class TestApplyProxReweight(common_utils.TestCase):
    """Tests _apply_prox with tau_reweight != 1.0 across branches."""

    @common_utils.parametrize(
        "grouper_cls,prox_cls,tau_reweight,disable_vmap",
        [
            (ElemGrouper, ProxLasso, 2.0, True),
            (Dim0Grouper, ProxLasso, 3.0, False),
            (Dim0Grouper, ProxGroupLasso, 2.0, False),
        ],
    )
    def test_tau_reweight_scales_threshold(
        self, grouper_cls, prox_cls, tau_reweight, disable_vmap
    ):
        """tau_reweight multiplies into the pruning threshold correctly."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 2.0

        p = torch.randn(4, 6)
        p_ref = p.clone()

        # Compute reference manually
        if prox_cls is ProxGroupLasso:
            tau = math.sqrt(p.numel() // p.size(0))  # group_size for Dim0Grouper
            threshold = reg_lambda * tau * tau_reweight * gamma
            for i in range(4):
                row = p_ref[i]
                norm = torch.linalg.vector_norm(row)
                row.mul_(max(1 - threshold / norm.item(), 0))
        else:
            threshold = reg_lambda * gamma * tau_reweight
            mult_ref = (1 - threshold / p_ref.abs()).clamp(min=0)
            p_ref.mul_(mult_ref)

        grouper = grouper_cls(p)
        prox_kwargs = make_prox_kwargs(gamma, disable_vmap=disable_vmap)
        PruneOptimizer._apply_prox(
            grouper, prox_cls(reg_lambda), p, tau_reweight=tau_reweight, **prox_kwargs
        )

        self.assertEqual(p, p_ref)

    def test_reweight_monotonicity(self):
        """Higher tau_reweight zeros more elements; lower zeros fewer."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 1.0

        data = torch.randn(4, 6)
        zeros = {}
        for tw in [0.1, 1.0, 5.0]:
            p = data.clone()
            grouper = Dim0Grouper(p)
            prox_kwargs = make_prox_kwargs(gamma)
            z, _, _ = PruneOptimizer._apply_prox(
                grouper, ProxGroupLasso(reg_lambda), p, tau_reweight=tw, **prox_kwargs
            )
            zeros[tw] = z

        self.assertLessEqual(zeros[0.1], zeros[1.0])
        self.assertGreaterEqual(zeros[5.0], zeros[1.0])


@unittest.skipUnless(dist.is_available(), "torch.distributed not available")
class TestApplyProxReweightDTensor(DistributedTestMixin, common_utils.TestCase):
    """DTensor tests for _apply_prox with tau_reweight."""

    @common_utils.parametrize(
        "GrouperCls,placements,prox_cls",
        [
            (Dim0Grouper, (Shard(0), Replicate()), ProxGroupLasso),
            (Dim1Grouper, (Shard(1), Replicate()), ProxGroupLasso),
            (Dim0Grouper, (Shard(0), Replicate()), ProxLasso),
            (Dim1Grouper, (Shard(1), Replicate()), ProxLasso),
        ],
    )
    def test_dtensor_matches_regular_with_reweight(
        self, GrouperCls, placements, prox_cls
    ):
        """DTensor vs regular tensor equivalence with tau_reweight."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 2.0
        tau_reweight = 2.5

        p_regular = torch.randn(4, 6)
        p_dtensor = distribute_tensor(
            p_regular.clone(), device_mesh=self.mesh, placements=placements
        )

        prox_kwargs = make_prox_kwargs(gamma)

        grouper_reg = GrouperCls(p_regular)
        z_reg, _, _ = PruneOptimizer._apply_prox(
            grouper_reg,
            prox_cls(reg_lambda),
            p_regular,
            tau_reweight=tau_reweight,
            **prox_kwargs,
        )

        grouper_dt = GrouperCls(p_dtensor)
        z_dt, _, _ = PruneOptimizer._apply_prox(
            grouper_dt,
            prox_cls(reg_lambda),
            p_dtensor,
            tau_reweight=tau_reweight,
            **prox_kwargs,
        )

        self.assertEqual(z_reg, z_dt)
        self.assertEqual(p_regular, p_dtensor.full_tensor())

    @common_utils.parametrize(
        "GrouperCls,placements,prox_cls",
        [
            (Dim0Grouper, (Shard(0), Replicate()), ProxGroupLasso),
            (Dim1Grouper, (Shard(1), Replicate()), ProxGroupLasso),
            (Dim0Grouper, (Shard(0), Replicate()), ProxLasso),
            (Dim1Grouper, (Shard(1), Replicate()), ProxLasso),
        ],
    )
    def test_dtensor_gamma_index_slope_with_tensor_reweight(
        self, GrouperCls, placements, prox_cls
    ):
        """DTensor with gamma_index_slope > 0 and tensor tau_reweight."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 2.0

        p_regular = torch.randn(4, 6)
        n_groups = p_regular.size(0) if GrouperCls == Dim0Grouper else p_regular.size(1)
        tau_reweight = torch.rand(n_groups) + 0.5  # [0.5, 1.5)

        p_dtensor = distribute_tensor(
            p_regular.clone(), device_mesh=self.mesh, placements=placements
        )

        prox_kwargs = make_prox_kwargs(gamma, gamma_index_slope=0.5)

        grouper_reg = GrouperCls(p_regular)
        z_reg, _, _ = PruneOptimizer._apply_prox(
            grouper_reg,
            prox_cls(reg_lambda),
            p_regular,
            tau_reweight=tau_reweight.clone(),
            **prox_kwargs,
        )

        grouper_dt = GrouperCls(p_dtensor)
        z_dt, _, _ = PruneOptimizer._apply_prox(
            grouper_dt,
            prox_cls(reg_lambda),
            p_dtensor,
            tau_reweight=tau_reweight.clone(),
            **prox_kwargs,
        )

        self.assertEqual(z_reg, z_dt)
        self.assertEqual(p_regular, p_dtensor.full_tensor())


class TestPruneOptimizerReweight(common_utils.TestCase):
    """End-to-end tests using PruneOptimizer with reweight_tau_freq > 0."""

    def test_sigma_initialized_at_warmup_end(self):
        """After warmup, state['sigma'] exists for regularized params."""
        torch.manual_seed(42)
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        warmup = 2
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=1.0,
            warmup_steps=warmup,
            reweight_tau_freq=1,
        )

        dummy_input = torch.randn(10, 10)
        label = torch.randint(0, 2, (10,))
        for step in range(5):
            optim_step(model, optimizer, dummy_input, label, step)
            if step < warmup:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        self.assertNotIn("sigma", optimizer.state[p])
            elif step == warmup:
                for group in optimizer.regularized_param_groups():
                    for p in group["params"]:
                        self.assertIn("sigma", optimizer.state[p])

    def test_tau_reweight_updated_at_freq(self):
        """state['tau_reweight'] is updated every reweight_tau_freq steps."""
        torch.manual_seed(42)
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=1.0,
            warmup_steps=0,
            reweight_tau_freq=3,
        )

        dummy_input = torch.randn(10, 10)
        label = torch.randint(0, 2, (10,))
        for step in range(10):
            optim_step(model, optimizer, dummy_input, label, step)

        has_tau_reweight = any(
            "tau_reweight" in optimizer.state[p]
            for group in optimizer.regularized_param_groups()
            for p in group["params"]
        )
        self.assertTrue(has_tau_reweight)

    def test_no_reweight_when_freq_zero(self):
        """With reweight_tau_freq=0, no sigma/tau_reweight in state."""
        torch.manual_seed(42)
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=1.0,
            warmup_steps=0,
            reweight_tau_freq=0,
        )

        dummy_input = torch.randn(10, 10)
        label = torch.randint(0, 2, (10,))
        for step in range(10):
            optim_step(model, optimizer, dummy_input, label, step)

        for group in optimizer.regularized_param_groups():
            for p in group["params"]:
                self.assertNotIn("sigma", optimizer.state[p])
                self.assertNotIn("tau_reweight", optimizer.state[p])

    def test_tau_reweight_frozen_after_end_step(self):
        """tau_reweight stops updating after reweight_tau_end_step."""
        torch.manual_seed(42)
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._linear_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        end_step = 5
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=1.0,
            warmup_steps=0,
            reweight_tau_freq=1,
            reweight_tau_end_step=end_step,
        )

        dummy_input = torch.randn(20, 10)
        label = torch.randint(0, 2, (20,))
        for step in range(10):
            optim_step(model, optimizer, dummy_input, label, step)

        # Capture tau_reweight after it should have frozen
        frozen = {
            id(p): optimizer.state[p]["tau_reweight"].clone()
            for group in optimizer.regularized_param_groups()
            for p in group["params"]
            if "tau_reweight" in optimizer.state[p]
        }
        self.assertTrue(len(frozen) > 0, "tau_reweight should exist")

        for step in range(10, 15):
            optim_step(model, optimizer, dummy_input, label, step)

        for group in optimizer.regularized_param_groups():
            for p in group["params"]:
                self.assertEqual(optimizer.state[p]["tau_reweight"], frozen[id(p)])

    def test_reweight_with_group_lasso(self):
        """End-to-end with Dim0Grouper + ProxGroupLasso (hits vmap branch)."""
        torch.manual_seed(42)
        model = TwoLayerMLP(input_size=10, output_size=2)
        prune_config = model._group_lasso_prune_config()
        param_groups = get_param_groups(model, prune_config, verbose=False)
        optimizer = PruneOptimizer(
            torch.optim.SGD(param_groups, lr=0.1),
            reg_lambda=1.0,
            warmup_steps=0,
            reweight_tau_freq=2,
        )

        dummy_input = torch.randn(10, 10)
        label = torch.randint(0, 2, (10,))
        for step in range(10):
            optim_step(model, optimizer, dummy_input, label, step)

        for group in optimizer.regularized_param_groups():
            for p in group["params"]:
                state = optimizer.state[p]
                self.assertIn("sigma", state)
                self.assertIn("tau_reweight", state)
                n_groups = p.size(0)  # Dim0Grouper groups along dim 0
                self.assertEqual(state["tau_reweight"].numel(), n_groups)


common_utils.instantiate_parametrized_tests(TestApplyProxReweight)
common_utils.instantiate_parametrized_tests(TestApplyProxReweightDTensor)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
