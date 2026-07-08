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
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal import common_utils

from test.prototype.pat.test_common import (
    DistributedTestMixin,
    TwoLayerMLP,
    make_prox_kwargs,
    optim_step,
)
from torchao.prototype.pat.group import KElementGrouper
from torchao.prototype.pat.optim import NMSparseConstraint, PruneOptimizer
from torchao.prototype.pat.utils import get_param_groups


class TestNMSparseConstraint(common_utils.TestCase):
    """Direct unit tests for NMSparseConstraint.apply_."""

    def test_keeps_n_nonzero_largest(self):
        """Exactly n_nonzero largest-magnitude elements survive."""
        p = torch.tensor([1.0, -3.0, 2.0, -0.5])
        n_nonzero = 2
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=n_nonzero)
        zeros_count, group_norm = prox.apply_(p, gamma=1.0)

        self.assertEqual(p.ne(0).sum().item(), n_nonzero)
        self.assertEqual(zeros_count.item(), p.numel() - n_nonzero)
        self.assertEqual(p.ne(0).tolist(), [False, True, True, False])

    def test_n_nonzero_equals_zero(self):
        """Keep nothing: all elements are zeroed."""
        p = torch.tensor([1.0, 2.0, 3.0, 4.0])
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=0)
        zeros_count, group_norm = prox.apply_(p, gamma=1.0)

        self.assertEqual(zeros_count.item(), 4)
        self.assertTrue(p.eq(0).all())
        self.assertEqual(group_norm.item(), 0.0)

    def test_group_norm_is_scalar(self):
        """group_norm must be a 0-dim tensor (scalar)."""
        p = torch.tensor([1.0, -3.0, 2.0, -0.5])
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=2)
        _, group_norm = prox.apply_(p, gamma=1.0)
        self.assertEqual(group_norm.dim(), 0)


class TestNMSparseWithKElementGrouper(common_utils.TestCase):
    """Integration tests: NMSparseConstraint + KElementGrouper via _apply_prox."""

    @staticmethod
    def assert_nm_pattern(p, k, n_nonzero, exact=True):
        """Each consecutive block of k elements has (== or <=) n_nonzero non-zeros."""
        M, N = p.shape
        for i in range(M):
            for start in range(0, N, k):
                block = p[i, start : start + k]
                count = block.ne(0).sum().item()
                if exact:
                    assert count == n_nonzero, (
                        f"Block ({i},{start}:{start + k}) has {count}, want {n_nonzero}"
                    )
                else:
                    assert count <= n_nonzero, (
                        f"Block ({i},{start}:{start + k}) has {count} > {n_nonzero}"
                    )

    @common_utils.parametrize(
        "n_nonzero,k",
        [(2, 4), (1, 4), (3, 4)],
    )
    def test_nm_sparsity(self, n_nonzero, k):
        """Each block of k elements has exactly n_nonzero non-zeros."""
        torch.manual_seed(42)
        M, N = 4, 8
        p = torch.randn(M, N)
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=n_nonzero)

        grouper = KElementGrouper(p, k=k)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        zero_elts, group_norm, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox, p, **prox_kwargs
        )

        self.assert_nm_pattern(p, k, n_nonzero, exact=True)
        n_groups = M * N // k
        self.assertEqual(zero_elts, (k - n_nonzero) * n_groups)

    def test_k_equals_one_elementwise(self):
        """k=1 makes each element its own group; n_nonzero=0 zeroes everything."""
        torch.manual_seed(42)
        M, N = 3, 5
        p = torch.randn(M, N)
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=0)

        grouper = KElementGrouper(p, k=1)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        zero_elts, _, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox, p, **prox_kwargs
        )

        self.assertTrue(p.eq(0).all())
        self.assertEqual(zero_elts, M * N)
        self.assertTrue(zeros_are_summed)

    def test_nm_sparsity_with_padding(self):
        """N:M pattern holds when N is not divisible by k (padding needed)."""
        torch.manual_seed(42)
        M, N, k, n_nonzero = 2, 6, 4, 2
        p = torch.randn(M, N)
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=n_nonzero)

        grouper = KElementGrouper(p, k=k)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        _, _, zeros_are_summed = PruneOptimizer._apply_prox(
            grouper, prox, p, **prox_kwargs
        )

        # Padded blocks may have <= n_nonzero (partial last block).
        self.assert_nm_pattern(p, k, n_nonzero, exact=False)
        self.assertTrue(zeros_are_summed)

    def test_group_norm_is_scalar_per_group(self):
        """group_norm shape should be (n_groups,), not (n_groups, k)."""
        torch.manual_seed(42)
        M, N, k, n_nonzero = 4, 8, 4, 2
        p = torch.randn(M, N)
        prox = NMSparseConstraint(reg_lambda=0.0, n_nonzero=n_nonzero)

        grouper = KElementGrouper(p, k=k)
        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)
        _, group_norm, _ = PruneOptimizer._apply_prox(grouper, prox, p, **prox_kwargs)

        n_groups = M * N // k
        self.assertEqual(group_norm.shape, (n_groups,))


@unittest.skipUnless(dist.is_available(), "torch.distributed not available")
class TestNMSparseDTensor(DistributedTestMixin, common_utils.TestCase):
    """DTensor tests for NMSparseConstraint + KElementGrouper."""

    def test_dtensor_matches_regular(self):
        """DTensor path produces same results as regular tensor path."""
        torch.manual_seed(42)
        M, N, k, n_nonzero = 4, 8, 4, 2
        reg_lambda = 0.0

        p_regular = torch.randn(M, N)
        p_dtensor = distribute_tensor(
            p_regular.clone(),
            device_mesh=self.mesh,
            placements=(Shard(0), Replicate()),
        )

        prox_kwargs = make_prox_kwargs(gamma=1.0, zero_elts_are_counts=True)

        # Regular path
        grouper_reg = KElementGrouper(p_regular, k=k)
        z_reg, gn_reg, _ = PruneOptimizer._apply_prox(
            grouper_reg,
            NMSparseConstraint(reg_lambda, n_nonzero=n_nonzero),
            p_regular,
            **prox_kwargs,
        )

        # DTensor path
        grouper_dt = KElementGrouper(p_dtensor, k=k)
        z_dt, gn_dt, _ = PruneOptimizer._apply_prox(
            grouper_dt,
            NMSparseConstraint(reg_lambda, n_nonzero=n_nonzero),
            p_dtensor,
            **prox_kwargs,
        )

        self.assertEqual(z_reg, z_dt)
        self.assertEqual(p_regular, p_dtensor.full_tensor())


class TestNMSparseOptimizer(common_utils.TestCase):
    """End-to-end test: PruneOptimizer with KElementGrouper + NMSparseConstraint."""

    def test_nm_sparsity_e2e(self):
        """After pruning steps, weights exhibit N:M sparsity pattern."""
        torch.manual_seed(42)
        k, n_nonzero, total_steps = 4, 2, 5
        model = TwoLayerMLP(input_size=8, output_size=2)
        prune_config = {
            (torch.nn.Linear, "weight"): {
                "group_type": "KElementGrouper",
                "prox_type": "NMSparseConstraint",
                "k": k,
                "n_nonzero": n_nonzero,
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
                # Training may not yet hit the full N:M pattern; allow <=.
                TestNMSparseWithKElementGrouper.assert_nm_pattern(
                    flat, k, n_nonzero, exact=False
                )

        self.assertGreater(optimizer.relative_sparsity, 0)
        self.assertLessEqual(optimizer.relative_sparsity, 1.0)


common_utils.instantiate_parametrized_tests(TestNMSparseConstraint)
common_utils.instantiate_parametrized_tests(TestNMSparseWithKElementGrouper)
common_utils.instantiate_parametrized_tests(TestNMSparseDTensor)
common_utils.instantiate_parametrized_tests(TestNMSparseOptimizer)


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    unittest.main()
