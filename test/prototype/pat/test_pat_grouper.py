# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import unittest

import torch
from torch import nn
from torch.testing._internal import common_utils

from torchao.prototype.pat.group import (
    AttentionHeadGrouperDim0,
    AttentionHeadGrouperDim1,
    Dim0Grouper,
    PackedSVDGrouper,
    SVDGrouper,
)
from torchao.prototype.pat.optim import (
    ProxGroupLasso,
    ProxNuclearNorm,
    PruneOptimizer,
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


class TestFixedHeadWidth(common_utils.TestCase):
    def test_equivalence_and_writeback(self):
        for cls, axis in ((AttentionHeadGrouperDim0, 0), (AttentionHeadGrouperDim1, 1)):
            for heads in (2, 4):
                with self.subTest(grouper=cls.__name__, heads=heads):
                    shape = (heads * 3, 5) if axis == 0 else (5, heads * 3)
                    original = torch.arange(1.0, 1 + math.prod(shape)).reshape(shape)
                    by_count, by_width = original.clone(), original.clone()
                    with (
                        cls(by_count, num_heads=heads) as a,
                        cls(by_width, head_dim=3) as b,
                    ):
                        self.assertEqual(a.p, b.p)
                        self.assertEqual(b.num_heads, heads)
                        if axis == 0:
                            a.p[0].zero_()
                            b.p[0].zero_()
                        else:
                            a.p[:, 0].zero_()
                            b.p[:, 0].zero_()
                    expected = original.clone()
                    if axis == 0:
                        expected[:3].zero_()
                    else:
                        expected[:, :3].zero_()
                    self.assertEqual(by_count, expected)
                    self.assertEqual(by_width, expected)

    def test_invalid_arguments(self):
        for cls in (AttentionHeadGrouperDim0, AttentionHeadGrouperDim1):
            for kwargs, message in (
                ({}, "exactly one"),
                ({"num_heads": 2, "head_dim": 3}, "exactly one"),
                ({"head_dim": 0}, "positive integer"),
                ({"num_heads": -1}, "positive integer"),
                ({"head_dim": 1.5}, "positive integer"),
                ({"num_heads": True}, "positive integer"),
                ({"head_dim": 4}, "not divisible"),
                ({"num_heads": 4}, "not divisible"),
            ):
                with self.subTest(grouper=cls.__name__, kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        cls(torch.ones(6, 6), **kwargs)

    def test_optimizer_config(self):
        for axis in (0, 1):
            params = [nn.Parameter(torch.arange(1.0, 25.0).reshape(6, 4))]
            if axis == 1:
                params = [nn.Parameter(params[0].detach().t().contiguous())]
            group = {
                "params": params,
                "group_type": f"AttentionHeadGrouperDim{axis}",
                "prox_type": "MinSparsityConstraint",
                "min_sparsity": 0.5,
                "head_dim": 3,
            }
            opt = PruneOptimizer(torch.optim.SGD([group], lr=0.0))
            params[0].grad = torch.zeros_like(params[0])
            opt.step()
            self.assertEqual(int(params[0].eq(0).sum()), 12)
            for kwargs in ({}, {"num_heads": 2, "head_dim": 3}):
                invalid = {"group_type": group["group_type"], **kwargs}
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    opt._get_grouper_kwargs(invalid)


class TestDimGrouper(common_utils.TestCase):
    def test_noncontiguous_flatten_writes_back(self):
        param = torch.arange(24.0).reshape(2, 3, 4).transpose(1, 2)
        self.assertFalse(param.is_contiguous())

        with Dim0Grouper(param) as grouper:
            self.assertEqual(grouper.p.shape, (2, 12))
            grouper.p.zero_()

        self.assertTrue(param.eq(0).all())


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
common_utils.instantiate_parametrized_tests(TestDimGrouper)
common_utils.instantiate_parametrized_tests(TestSVDGrouper)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
