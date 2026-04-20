# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.prototype.pat.group import (
    AttentionHeadGrouperDim0,
    AttentionHeadGrouperDim1,
    PackedSVDGrouper,
    SVDGrouper,
)
from torchao.prototype.pat.optim import (
    ProxGroupLasso,
    ProxNuclearNorm,
)


class TestAttentionHeadGrouper(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestAttentionHeadGrouper, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxGroupLasso(self.reg_lambda)

    @staticmethod
    def _get_view_shape_reduce_dim(dim, num_heads, head_pack_dim):
        if head_pack_dim == 0:
            view_shape = (num_heads, -1, dim)
            reduce_dim = (1, 2)
        else:
            view_shape = (dim, num_heads, -1)
            reduce_dim = (0, 2)
        return view_shape, reduce_dim

    def _test_post_prune(self, p, p_orig, head_pack_dim, view_shape, reduce_dim, gamma):
        nz_mask = p.view(*view_shape).sum(dim=reduce_dim).ne(0)
        self.assertTrue(nz_mask.eq(0).any(), "No groups of p were pruned")

        # original groups that are <= gamma are pruned
        expect_nz_mask = p_orig.view(*view_shape).gt(gamma).all(dim=reduce_dim)
        torch.testing.assert_close(nz_mask > 1, expect_nz_mask, atol=0, rtol=0)

    def get_gamma(self, p, head_pack_dim, view_shape):
        """Heuristic that uses the mean of the group to set gamma."""
        p = p.view(*view_shape)
        p_group = p[0] if head_pack_dim == 0 else p[:, 0]
        gamma = (1 - p_group.mean()) * torch.linalg.vector_norm(p_group)
        gamma.div_(self.prox_map.tau(p_group))
        return gamma

    @common_utils.parametrize("dim", [64, 128])
    @common_utils.parametrize("head_pack_dim", [0, 1])
    def test_head_grouper(self, dim=16, head_pack_dim=0, head_dim_ratio=8):
        assert dim % head_dim_ratio == 0, (
            f"{dim=} must be divisible by {head_dim_ratio=}"
        )
        num_heads = dim // 8
        packed_dim = dim * num_heads
        shape = (dim, packed_dim) if head_pack_dim == 0 else (packed_dim, dim)
        model = nn.Linear(*shape, bias=False)
        p = model.weight.detach()
        p_orig = p.clone()
        view_shape, reduce_dim = self._get_view_shape_reduce_dim(
            dim, num_heads, head_pack_dim
        )
        grouper_cls = (
            AttentionHeadGrouperDim0 if head_pack_dim == 0 else AttentionHeadGrouperDim1
        )
        with grouper_cls(p, num_heads) as grouper:
            gamma = self.get_gamma(grouper.p, head_pack_dim, view_shape)
            _ = torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p, gamma)
            self.assertEqual(grouper.p.size(head_pack_dim), num_heads)
        self._test_post_prune(p, p_orig, head_pack_dim, view_shape, reduce_dim, gamma)


class TestSVDGrouper(common_utils.TestCase):
    def __init__(self, methodName):
        super(TestSVDGrouper, self).__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxNuclearNorm(self.reg_lambda)

    @common_utils.parametrize("embed_dim", (16, 64))
    def test_grouper(self, embed_dim=16):
        model = torch.nn.Linear(embed_dim, embed_dim)
        p = model.weight
        with SVDGrouper(p) as grouper:
            gamma = grouper.p.mean()
            p_orig = grouper.p.clone()
            torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p, gamma)
            expect_nz_mask = p_orig.gt(gamma)
            torch.testing.assert_close(grouper.p.ne(0), expect_nz_mask, atol=0, rtol=0)

    @common_utils.parametrize("embed_dim", (16, 64))
    @common_utils.parametrize("pack_dim", (0, 1))
    def test_packed_grouper(self, embed_dim=16, npack=3, pack_dim=0):
        shape = [embed_dim, embed_dim]
        shape[int(not pack_dim)] *= npack
        model = torch.nn.Linear(*shape)
        p = model.weight
        with PackedSVDGrouper(p, npack, pack_dim=pack_dim) as grouper:
            gamma = grouper.p.mean(0).mean()
            p_orig = grouper.p.clone()
            torch.vmap(
                self.prox_map.apply_, in_dims=(grouper.in_dims, None), out_dims=0
            )(grouper.p.flatten(), gamma)
            torch.testing.assert_close(
                grouper.p.ne(0), p_orig.gt(gamma), atol=0, rtol=0
            )
            self.assertEqual(p.data_ptr(), grouper._p.data_ptr())


common_utils.instantiate_parametrized_tests(TestAttentionHeadGrouper)
common_utils.instantiate_parametrized_tests(TestSVDGrouper)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
