# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch
from torch.testing._internal import common_utils

from torchao.prototype.pat.group import KElementGrouper
from torchao.prototype.pat.optim import ProxGroupLasso


class TestKElementGrouper(common_utils.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)
        self.reg_lambda = 1.0
        self.prox_map = ProxGroupLasso(self.reg_lambda)

    @common_utils.parametrize("k", (2, 4))
    def test_even_divisible(self, k):
        """Tensor (2, 8) with k dividing 8 evenly."""
        M, N = 2, 8
        p = torch.nn.Parameter(torch.randn(M, N))
        orig_shape = p.shape
        with KElementGrouper(p, k=k) as grouper:
            self.assertEqual(grouper.group_size(), k)
            self.assertEqual(grouper.n_groups(), M * N // k)
            self.assertEqual(grouper.p.data.shape, (M * N // k, k))
        # Shape restored after exit
        self.assertEqual(p.shape, orig_shape)

    @common_utils.parametrize("k", (4,))
    def test_remainder(self, k):
        """Tensor (2, 6) with k=4: 6 is not divisible by 4."""
        M, N = 2, 6
        p = torch.nn.Parameter(torch.randn(M, N))
        orig_data = p.data.clone()
        orig_shape = p.shape
        n_groups_per_row = math.ceil(N / k)
        with KElementGrouper(p, k=k) as grouper:
            self.assertEqual(grouper.group_size(), k)
            self.assertEqual(grouper.n_groups(), M * n_groups_per_row)
            self.assertEqual(grouper.p.data.shape, (M * n_groups_per_row, k))
        # Shape and data restored after exit
        self.assertEqual(p.shape, orig_shape)
        torch.testing.assert_close(p.data, orig_data)

    def test_data_preserved_after_context(self):
        """Verify data values survive the enter/exit round-trip."""
        p = torch.nn.Parameter(torch.arange(12, dtype=torch.float).reshape(3, 4))
        orig_data = p.data.clone()
        with KElementGrouper(p, k=2):
            pass
        torch.testing.assert_close(p.data, orig_data)

    def test_data_preserved_remainder(self):
        """Verify data round-trips correctly with padding."""
        p = torch.nn.Parameter(torch.arange(15, dtype=torch.float).reshape(3, 5))
        orig_data = p.data.clone()
        with KElementGrouper(p, k=4):
            pass
        torch.testing.assert_close(p.data, orig_data)

    @common_utils.parametrize(
        "shape,k",
        [
            ((2, 5, 7), 5),  # 3D: (out, ...) -> 35 cols
            ((4, 3, 3, 3), 9),  # 4D conv-like: (out, in, H, W) -> 27 cols
            ((3, 2, 2, 2, 2), 4),  # 5D: (out, ...) -> 16 cols
        ],
    )
    def test_nd_tensor(self, shape, k):
        """Higher-dim tensors are flattened via DimGrouperMixin before chunking."""
        p = torch.nn.Parameter(torch.randn(*shape))
        orig_shape = p.shape
        with KElementGrouper(p, k=k) as grouper:
            M = shape[0]  # dim 0 preserved
            N = math.prod(shape[1:])  # remaining dims flattened
            n_groups_per_row = math.ceil(N / k)
            self.assertEqual(grouper.group_size(), k)
            self.assertEqual(grouper.n_groups(), M * n_groups_per_row)
        self.assertEqual(p.shape, orig_shape)

    @common_utils.parametrize(
        "M,N,k",
        [(2, 8, 4), (3, 6, 2), (1, 12, 3)],
    )
    @torch.no_grad()
    def test_vmap_prox(self, M, N, k):
        """ProxGroupLasso via vmap zeroes out small-norm groups."""
        p = torch.nn.Parameter(torch.randn(M, N))
        with KElementGrouper(p, k=k) as grouper:
            # Compute the minimum gamma that prunes all groups, plus a small margin
            max_group_norm = grouper.p.data.norm(dim=-1).max()
            gamma = torch.tensor(
                max_group_norm.item() / (self.reg_lambda * math.sqrt(k)) * 1.01
            )
            torch.vmap(
                self.prox_map.apply_,
                in_dims=(grouper.in_dims, None),
                out_dims=0,
            )(grouper.p, gamma)
        # With a large enough gamma, all groups should be pruned to zero
        self.assertTrue(torch.all(p.data == 0))

    def test_k_equals_one(self):
        """k=1 should behave like element-wise grouping."""
        M, N = 3, 5
        p = torch.nn.Parameter(torch.randn(M, N))
        with KElementGrouper(p, k=1) as grouper:
            self.assertEqual(grouper.group_size(), 1)
            self.assertEqual(grouper.n_groups(), M * N)

    def test_k_equals_n(self):
        """k=N means each row is one group."""
        M, N = 3, 6
        p = torch.nn.Parameter(torch.randn(M, N))
        with KElementGrouper(p, k=N) as grouper:
            self.assertEqual(grouper.group_size(), N)
            self.assertEqual(grouper.n_groups(), M)

    def test_k_must_be_positive(self):
        p = torch.nn.Parameter(torch.randn(2, 4))
        with self.assertRaises(AssertionError):
            KElementGrouper(p, k=0)


common_utils.instantiate_parametrized_tests(TestKElementGrouper)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
