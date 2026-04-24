# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import DistributedTestMixin, make_prox_kwargs
from torchao.prototype.pat.group import Dim0Grouper, Dim1Grouper
from torchao.prototype.pat.optim import (
    ProxGroupLasso,
    ProxGroupLassoVectorized,
    ProxLasso,
    PruneOptimizer,
)
from torchao.prototype.pat.utils import get_index_linspace


class TestApplyProx(common_utils.TestCase):
    """Tests that _apply_prox correctly vmaps the prox map across groups."""

    @common_utils.parametrize("GrouperCls", [Dim0Grouper, Dim1Grouper])
    def test_vmap_matches_manual_per_group(self, GrouperCls):
        """Verify vmap-based _apply_prox produces correct tensor values."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 2.0
        prox_map = ProxLasso(reg_lambda)
        # Threshold = reg_lambda * tau * gamma = 0.5 * 1.0 * 2.0 = 1.0
        # Elements with |x| <= 1.0 should be zeroed via soft thresholding.

        p = torch.randn(4, 6)
        p_ref = p.clone()

        # Manual element-wise soft thresholding (reference)
        threshold = reg_lambda * gamma  # tau=1 for ProxLasso
        mult_ref = (1 - threshold / p_ref.abs()).clamp(min=0)
        p_ref.mul_(mult_ref)

        # _apply_prox with vmap
        grouper = GrouperCls(p)
        prox_kwargs = make_prox_kwargs(gamma)
        zero_elts, group_norm, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper,
            prox_map,
            p,
            **prox_kwargs,
        )

        self.assertTrue(zeros_are_summed)
        self.assertEqual(p, p_ref)
        self.assertGreater(zero_elts, 0)

    def test_per_group_independence(self):
        """Verify each group is processed independently via vmap.

        With group lasso, vmap should compute the norm per-group (per-row for
        Dim0Grouper). Without vmap, the norm would be computed over the entire
        tensor, producing different results.
        """
        reg_lambda = 0.1
        gamma = 1.0
        prox_map = ProxGroupLasso(reg_lambda)

        # Row 0: small values (norm < threshold) -> should be fully zeroed
        # Row 1: large values (norm >> threshold) -> should survive
        p = torch.tensor(
            [
                [0.01, 0.02, -0.01, 0.005],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
        # Per-group (per-row) norms:
        # Row 0 norm ~ 0.025, threshold = 0.1 * sqrt(4) * 1.0 = 0.2 -> zeroed
        # Row 1 norm ~ 13.19, threshold = 0.2 -> survives

        grouper = Dim0Grouper(p)
        prox_kwargs = make_prox_kwargs(gamma)
        zero_elts, group_norm, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox_map, p, **prox_kwargs
        )

        # Row 0 should be fully zeroed
        self.assertTrue(p[0].eq(0).all())
        # Row 1 should be non-zero (slightly shrunk)
        self.assertTrue(p[1].ne(0).all())

    def test_gamma_index_slope(self):
        """_apply_prox with gamma_index_slope applies per-group gamma scaling."""
        torch.manual_seed(0)
        reg_lambda = 1.0
        gamma = 4.0
        slope = 1.0  # linspace from 0 to 1 (after div 2, clamp)

        n_groups = 4
        p = torch.randn(n_groups, 5)
        p_ref = p.clone()

        grouper = Dim0Grouper(p)
        prox_kwargs = make_prox_kwargs(gamma, gamma_index_slope=slope)
        zero_elts, group_norm, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper,
            ProxLasso(reg_lambda),
            p,
            **prox_kwargs,
        )

        # First group gets smallest gamma multiplier (~0), last gets largest (~1)
        # so first group should be least pruned and last group most pruned
        first_zeros = p[0].eq(0).sum().item()
        last_zeros = p[-1].eq(0).sum().item()
        self.assertGreaterEqual(last_zeros, first_zeros)

        # Verify tensor values match manual computation
        gamma_multiplier = (
            torch.linspace(1 - slope, 1 + slope, n_groups).div_(2.0).clamp_(min=0.0)
        )
        scaled_gamma = gamma * gamma_multiplier
        for i in range(n_groups):
            threshold = reg_lambda * scaled_gamma[i]
            row = p_ref[i]
            mult = (1 - threshold / row.abs()).clamp(min=0)
            row.mul_(mult)
        self.assertEqual(p, p_ref)


class TestProxGroupLassoVectorized(common_utils.TestCase):
    """Tests that ProxGroupLassoVectorized matches ProxGroupLasso + vmap."""

    @common_utils.parametrize("reduce_dim", [0, 1])
    def test_vectorized_matches_vmap(self, reduce_dim):
        """Vectorized apply_ produces same results as vmap over ProxGroupLasso."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 2.0

        p = torch.randn(4, 6)
        p_vec = p.clone()
        p_vmap = p.clone()

        # Vectorized path
        prox_vec = ProxGroupLassoVectorized(reg_lambda, reduce_dim=reduce_dim)
        zero_vec, group_norm_vec = prox_vec.apply_(p_vec, gamma, 1.0)

        # vmap path
        prox_map = ProxGroupLasso(reg_lambda)
        in_dims = int(not reduce_dim)  # vmap iterates over groups dimension
        zero_vmap, group_norm_vmap = torch.vmap(
            prox_map.apply_, in_dims=(in_dims, None, None), out_dims=(0, 0)
        )(p_vmap, gamma, 1.0)

        self.assertEqual(p_vec, p_vmap)
        self.assertEqual(zero_vec, zero_vmap.sum())


@unittest.skipUnless(dist.is_available(), "torch.distributed not available")
class TestApplyProxVmap(DistributedTestMixin, common_utils.TestCase):
    """Tests that _apply_prox handles DTensor inputs via local_map."""

    @common_utils.parametrize(
        "GrouperCls,placements,prox_cls",
        [
            (Dim0Grouper, (Shard(0), Replicate()), ProxLasso),
            (Dim1Grouper, (Shard(1), Replicate()), ProxLasso),
            (Dim0Grouper, (Shard(0), Replicate()), ProxGroupLasso),
            (Dim1Grouper, (Shard(1), Replicate()), ProxGroupLasso),
        ],
    )
    def test_dtensor_matches_regular(self, GrouperCls, placements, prox_cls):
        """DTensor _apply_prox produces same results as regular tensor path."""
        torch.manual_seed(42)
        reg_lambda = 0.5
        gamma = 2.0
        prox_map = prox_cls(reg_lambda)

        p_regular = torch.randn(4, 6)
        p_dtensor = distribute_tensor(
            p_regular.clone(), device_mesh=self.mesh, placements=placements
        )

        # Run regular tensor path
        grouper_reg = GrouperCls(p_regular)
        prox_kwargs = make_prox_kwargs(gamma)
        zero_reg, group_norm_reg, summed_reg = PruneOptimizer._apply_prox(
            grouper_reg, prox_map, p_regular, **prox_kwargs
        )

        # Run DTensor path
        grouper_dt = GrouperCls(p_dtensor)
        zero_dt, group_norm_dt, summed_dt = PruneOptimizer._apply_prox(
            grouper_dt, prox_map, p_dtensor, **prox_kwargs
        )

        self.assertTrue(summed_reg)
        self.assertTrue(summed_dt)
        self.assertEqual(zero_reg, zero_dt)
        self.assertEqual(p_regular, p_dtensor.full_tensor())

    @common_utils.parametrize(
        "GrouperCls,placements,prox_cls,slope",
        [
            # Dim0Grouper combinations
            (Dim0Grouper, (Shard(0), Replicate()), ProxLasso, 1.0),
            (Dim0Grouper, (Shard(0), Replicate()), ProxGroupLasso, 1.0),
            # Dim1Grouper combinations
            (Dim1Grouper, (Shard(1), Replicate()), ProxLasso, 1.0),
            (Dim1Grouper, (Shard(1), Replicate()), ProxGroupLasso, 1.0),
        ],
    )
    def test_dtensor_gamma_index_slope(self, GrouperCls, placements, prox_cls, slope):
        """DTensor path with gamma_index_slope distributes gamma correctly."""
        torch.manual_seed(0)
        reg_lambda = 1.0
        gamma = 4.0

        n_groups = 4
        p_data = torch.randn(n_groups, 5)
        p_regular = p_data.clone()
        p_dtensor = distribute_tensor(
            p_data.clone(), device_mesh=self.mesh, placements=placements
        )

        prox_kwargs = make_prox_kwargs(gamma, gamma_index_slope=slope)
        prox_map = prox_cls(reg_lambda)

        grouper_reg = GrouperCls(p_regular)
        PruneOptimizer._apply_prox(grouper_reg, prox_map, p_regular, **prox_kwargs)

        prox_map_dt = prox_cls(reg_lambda)
        grouper_dt = GrouperCls(p_dtensor)
        PruneOptimizer._apply_prox(grouper_dt, prox_map_dt, p_dtensor, **prox_kwargs)

        result_dt = p_dtensor.full_tensor()

        # Verify ordering: last group more pruned than first
        first_zeros = result_dt[0].eq(0).sum().item()
        last_zeros = result_dt[-1].eq(0).sum().item()
        self.assertGreaterEqual(last_zeros, first_zeros)

        # Verify values match regular tensor path
        self.assertEqual(p_regular, result_dt)

    @common_utils.parametrize(
        "reduce_dim,placements",
        [
            (0, (Shard(1), Replicate())),
            (1, (Shard(0), Replicate())),
        ],
    )
    def test_vectorized_tensor_gamma(self, reduce_dim, placements):
        """Vectorized apply_ handles 1D tensor gamma with DTensor inputs."""
        torch.manual_seed(42)
        reg_lambda = 0.5

        p = torch.randn(4, 6)
        n_groups = p.size(1 - reduce_dim)
        gamma = 4.0 * get_index_linspace(1.0, n_groups, device=p.device)

        # --- DTensor path (vectorized) ---
        # Use placements matching p_in_placements = Shard(int(not reduce_dim))
        # so local_map doesn't need to redistribute p.
        p_dt = distribute_tensor(
            p.clone(), device_mesh=self.mesh, placements=placements
        )
        gamma_dt = distribute_tensor(
            gamma.unsqueeze(reduce_dim),
            device_mesh=self.mesh,
            placements=(Shard(int(not reduce_dim)),) * self.mesh.ndim,
        )

        prox_vec = ProxGroupLassoVectorized(reg_lambda, reduce_dim=reduce_dim)

        p_in_placements = tuple(
            Shard(int(not reduce_dim)) if plc.is_shard() else plc
            for plc in p_dt.placements
        )

        zero_dt, group_norm_dt = local_map(
            prox_vec.apply_,
            out_placements=(
                (Partial(),) * self.mesh.ndim,
                (Shard(0),) * self.mesh.ndim,
            ),
            in_placements=(p_in_placements, gamma_dt.placements, None),
            redistribute_inputs=True,
        )(p_dt, gamma_dt, 1.0)

        # --- Reference path (vmap on regular tensor) ---
        p_ref = p.clone()
        prox_ref = ProxGroupLasso(reg_lambda)
        in_dims = int(not reduce_dim)
        zero_ref, group_norm_ref = torch.vmap(
            prox_ref.apply_, in_dims=(in_dims, 0, None), out_dims=(0, 0)
        )(p_ref, gamma, 1.0)

        self.assertEqual(p_dt.full_tensor(), p_ref)
        self.assertEqual(zero_dt.full_tensor().sum(), zero_ref.sum())


common_utils.instantiate_parametrized_tests(TestApplyProx)
common_utils.instantiate_parametrized_tests(TestApplyProxVmap)
common_utils.instantiate_parametrized_tests(TestProxGroupLassoVectorized)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
