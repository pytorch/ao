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
    """_apply_prox correctly vmaps the prox map across groups."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    @common_utils.parametrize("GrouperCls", [Dim0Grouper, Dim1Grouper])
    def test_vmap_matches_manual_per_group(self, GrouperCls):
        reg_lambda = 0.5
        gamma = 2.0
        prox_map = ProxLasso(reg_lambda)

        p = torch.randn(4, 6)
        p_ref = p.clone()

        # reference: soft thresholding (tau=1 for ProxLasso)
        threshold = reg_lambda * gamma
        mult_ref = (1 - threshold / p_ref.abs()).clamp(min=0)
        p_ref.mul_(mult_ref)

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
        """Group lasso must compute the norm per-row, not over the whole tensor."""
        reg_lambda = 0.1
        gamma = 1.0
        prox_map = ProxGroupLasso(reg_lambda)

        # Row 0 norm < threshold -> zeroed; Row 1 norm >> threshold -> survives.
        p = torch.tensor(
            [
                [0.01, 0.02, -0.01, 0.005],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )

        grouper = Dim0Grouper(p)
        prox_kwargs = make_prox_kwargs(gamma)
        zero_elts, group_norm, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox_map, p, **prox_kwargs
        )

        self.assertTrue(p[0].eq(0).all())
        self.assertTrue(p[1].ne(0).all())

    def test_gamma_index_slope(self):
        """gamma_index_slope applies per-group gamma scaling."""
        reg_lambda = 1.0
        gamma = 4.0
        slope = 1.0

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

        # gamma multiplier increases with group index, so prune count should too.
        first_zeros = p[0].eq(0).sum().item()
        last_zeros = p[-1].eq(0).sum().item()
        self.assertGreaterEqual(last_zeros, first_zeros)

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
    """ProxGroupLassoVectorized matches ProxGroupLasso + vmap."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)

    @common_utils.parametrize("reduce_dim", [0, 1])
    def test_vectorized_matches_vmap(self, reduce_dim):
        reg_lambda = 0.5
        gamma = 2.0

        p = torch.randn(4, 6)
        p_vec = p.clone()
        p_vmap = p.clone()

        prox_vec = ProxGroupLassoVectorized(reg_lambda, reduce_dim=reduce_dim)
        zero_vec, group_norm_vec = prox_vec.apply_(p_vec, gamma, 1.0)

        prox_map = ProxGroupLasso(reg_lambda)
        in_dims = int(not reduce_dim)
        zero_vmap, group_norm_vmap = torch.vmap(
            prox_map.apply_, in_dims=(in_dims, None, None), out_dims=(0, 0)
        )(p_vmap, gamma, 1.0)

        self.assertEqual(p_vec, p_vmap)
        self.assertEqual(zero_vec, zero_vmap.sum())


@unittest.skipUnless(dist.is_available(), "torch.distributed not available")
class TestApplyProxVmap(DistributedTestMixin, common_utils.TestCase):
    """_apply_prox handles DTensor inputs via local_map."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)

    @common_utils.parametrize(
        "GrouperCls,placements,prox_cls",
        [
            # Placement-mismatch cases: DTensor sharded on a dim that does not
            # match grouper.in_dims, exercising the redistribute + write-back
            # path in _apply_prox_dtensor (e.g. FSDP2 Shard(0) + Dim1Grouper).
            (Dim1Grouper, (Shard(0), Replicate()), ProxLasso),
            (Dim1Grouper, (Shard(0), Replicate()), ProxGroupLasso),
            (Dim0Grouper, (Shard(1), Replicate()), ProxLasso),
            (Dim0Grouper, (Shard(1), Replicate()), ProxGroupLasso),
        ],
    )
    def test_dtensor_placement_mismatch_writes_back(
        self, GrouperCls, placements, prox_cls
    ):
        """DTensor _apply_prox propagates in-place sparsity updates when
        parameter placements differ from grouper's required p_in_placements.
        Regression test for silent sparsity drop on local_map redistribute.
        """
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

        # Run DTensor path with mismatched placements. Without write-back,
        # p_dtensor.full_tensor() would still hold the pre-prox values.
        grouper_dt = GrouperCls(p_dtensor)
        zero_dt, group_norm_dt, summed_dt = PruneOptimizer._apply_prox(
            grouper_dt, prox_map, p_dtensor, **prox_kwargs
        )

        self.assertTrue(summed_reg)
        self.assertTrue(summed_dt)
        self.assertEqual(zero_reg, zero_dt)
        # Mutation must be visible on the original DTensor.
        self.assertEqual(p_regular, p_dtensor.full_tensor())

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
        reg_lambda = 0.5
        gamma = 2.0
        prox_map = prox_cls(reg_lambda)

        p_regular = torch.randn(4, 6)
        p_dtensor = distribute_tensor(
            p_regular.clone(), device_mesh=self.mesh, placements=placements
        )

        grouper_reg = GrouperCls(p_regular)
        prox_kwargs = make_prox_kwargs(gamma)
        zero_reg, group_norm_reg, summed_reg = PruneOptimizer._apply_prox(
            grouper_reg, prox_map, p_regular, **prox_kwargs
        )

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
            (Dim0Grouper, (Shard(0), Replicate()), ProxLasso, 1.0),
            (Dim0Grouper, (Shard(0), Replicate()), ProxGroupLasso, 1.0),
            (Dim1Grouper, (Shard(1), Replicate()), ProxLasso, 1.0),
            (Dim1Grouper, (Shard(1), Replicate()), ProxGroupLasso, 1.0),
        ],
    )
    def test_dtensor_gamma_index_slope(self, GrouperCls, placements, prox_cls, slope):
        """DTensor path distributes gamma_index_slope correctly."""
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

        first_zeros = result_dt[0].eq(0).sum().item()
        last_zeros = result_dt[-1].eq(0).sum().item()
        self.assertGreaterEqual(last_zeros, first_zeros)

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
        reg_lambda = 0.5

        p = torch.randn(4, 6)
        n_groups = p.size(1 - reduce_dim)
        gamma = 4.0 * get_index_linspace(1.0, n_groups, device=p.device)

        # placements match p_in_placements so local_map skips redistribution.
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
