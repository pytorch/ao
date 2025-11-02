# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal import common_utils

from torchao.prototype.quantization.quantize_.workflows.float8.float8_semisparse_tensor import (
    Float8SemiSparseTensor,
)
from torchao.testing.utils import TorchAOIntegrationTestCase


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@common_utils.instantiate_parametrized_tests
class TestFloat8SemiSparseTensor(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)

        # Use 512x512 for 2:4 compatibility (multiples of 32)
        self.shape = (512, 512)
        self.dtype = torch.bfloat16
        self.block_size = [1, 512]
        self.weight_fp = torch.randn(*self.shape, dtype=self.dtype, device="cuda")

    def test_creation_and_shape(self):
        """Test tensor creation and shape preservation"""
        tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)

        self.assertEqual(tensor.shape, self.shape)
        self.assertEqual(tensor.original_shape, self.shape)
        self.assertEqual(tensor.scale.shape[0], self.shape[0])

    def test_sparsity_pattern(self):
        """Test 2:4 sparsity pattern is maintained"""
        tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)
        dequantized = tensor.dequantize()

        # Check 2:4 pattern (skip overall sparsity check for compressed format)
        reshaped = dequantized.reshape(-1, 4)
        zeros_per_group = (reshaped == 0).sum(dim=1)
        valid_groups = (zeros_per_group == 2).sum().item()
        total_groups = zeros_per_group.numel()

        self.assertGreaterEqual(valid_groups / total_groups, 0.99)

    def test_dequantization_accuracy(self):
        """Test dequantization error is reasonable"""
        tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)
        dequantized = tensor.dequantize()

        # Apply same pruning to original for fair comparison
        w_sparse = self.weight_fp.detach().clone()
        pruning_inds = w_sparse.abs().view(-1, 4).argsort(dim=1)[:, :2]
        w_sparse.view(-1, 4).scatter_(1, pruning_inds, value=0)

        error = (dequantized - w_sparse).abs().max()
        rel_error = error / w_sparse.abs().max()

        self.assertLess(error.item(), 2.5)
        self.assertLess(rel_error.item(), 0.5)

    def test_invalid_dimensions(self):
        """Test dimension validation"""
        # Not multiple of 32
        invalid_weight = torch.randn(100, 100, dtype=self.dtype, device="cuda")

        with self.assertRaises(ValueError):
            Float8SemiSparseTensor.from_hp(invalid_weight, [1, 100])

    def test_cpu_tensor_rejection(self):
        """Test CPU tensor is rejected"""
        cpu_weight = torch.randn(*self.shape, dtype=self.dtype)

        with self.assertRaises(ValueError):
            Float8SemiSparseTensor.from_hp(cpu_weight, self.block_size)


if __name__ == "__main__":
    common_utils.run_tests()
