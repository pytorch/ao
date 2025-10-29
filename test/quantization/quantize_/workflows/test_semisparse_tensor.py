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
from torchao.prototype.quantization.quantize_.workflows.int8.int8_semisparse_tensor import (
    Int8SemiSparseTensor,
)
from torchao.testing.utils import TorchAOIntegrationTestCase


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@common_utils.instantiate_parametrized_tests
class TestSemiSparseTensor(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)

        # Use 512x512 for 2:4 compatibility (multiples of 32)
        self.shape = (512, 512)
        self.dtype = torch.bfloat16
        self.block_size = [1, 512]
        self.weight_fp = torch.randn(*self.shape, dtype=self.dtype, device="cuda")

    @common_utils.parametrize("config", [Int8SemiSparseTensor, Float8SemiSparseTensor])
    def test_creation_and_shape(self, config):
        """Test tensor creation and shape preservation"""
        tensor = config.from_hp(self.weight_fp, self.block_size)

        self.assertEqual(tensor.shape, self.shape)
        self.assertEqual(tensor.original_shape, self.shape)
        self.assertEqual(tensor.scale.shape[0], self.shape[0])

    @common_utils.parametrize("config", [Int8SemiSparseTensor, Float8SemiSparseTensor])
    def test_sparsity_pattern(self, config):
        """Test 2:4 sparsity pattern is maintained"""
        tensor = config.from_hp(self.weight_fp, self.block_size)
        dequantized = tensor.dequantize()

        # Check 2:4 pattern (skip overall sparsity check for compressed format)
        reshaped = dequantized.reshape(-1, 4)
        zeros_per_group = (reshaped == 0).sum(dim=1)
        valid_groups = (zeros_per_group == 2).sum().item()
        total_groups = zeros_per_group.numel()

        self.assertGreaterEqual(valid_groups / total_groups, 0.99)

    def test_int8_quantization_range(self):
        """Test Int8 quantization stays in valid range"""
        tensor = Int8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)

        self.assertEqual(tensor.qdata_int8.dtype, torch.int8)
        self.assertTrue(torch.all(tensor.qdata_int8 >= -128))
        self.assertTrue(torch.all(tensor.qdata_int8 <= 127))

    def test_float8_quantization_no_nan(self):
        """Test Float8 quantization produces no NaN"""
        tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)

        self.assertEqual(tensor.qdata_fp8.dtype, torch.float8_e4m3fn)
        self.assertFalse(tensor.qdata_fp8.isnan().any())
        self.assertFalse(tensor.scale.isnan().any())

    @common_utils.parametrize("config", [Int8SemiSparseTensor, Float8SemiSparseTensor])
    def test_dequantization_accuracy(self, config):
        """Test dequantization error is reasonable"""
        tensor = config.from_hp(self.weight_fp, self.block_size)
        dequantized = tensor.dequantize()

        # Apply same pruning to original for fair comparison
        w_sparse = self.weight_fp.detach().clone()
        pruning_inds = w_sparse.abs().view(-1, 4).argsort(dim=1)[:, :2]
        w_sparse.view(-1, 4).scatter_(1, pruning_inds, value=0)

        error = (dequantized - w_sparse).abs().max()
        rel_error = error / w_sparse.abs().max()

        # Int8: ~2.0, Float8: ~0.3
        max_error = 2.5 if config == Int8SemiSparseTensor else 0.5
        self.assertLess(error.item(), max_error)
        self.assertLess(rel_error.item(), 0.5)

    @common_utils.parametrize("config", [Int8SemiSparseTensor, Float8SemiSparseTensor])
    def test_invalid_dimensions(self, config):
        """Test dimension validation"""
        # Not multiple of 32
        invalid_weight = torch.randn(100, 100, dtype=self.dtype, device="cuda")

        with self.assertRaises(ValueError):
            config.from_hp(invalid_weight, [1, 100])

    @common_utils.parametrize("config", [Int8SemiSparseTensor, Float8SemiSparseTensor])
    def test_cpu_tensor_rejection(self, config):
        """Test CPU tensor is rejected"""
        cpu_weight = torch.randn(*self.shape, dtype=self.dtype)

        with self.assertRaises(ValueError):
            config.from_hp(cpu_weight, self.block_size)

    def test_float8_dtype_selection(self):
        """Test Float8 dtype variants"""
        tensor_e4m3 = Float8SemiSparseTensor.from_hp(
            self.weight_fp, self.block_size, float8_dtype=torch.float8_e4m3fn
        )
        self.assertEqual(tensor_e4m3.qdata_fp8.dtype, torch.float8_e4m3fn)

        tensor_e5m2 = Float8SemiSparseTensor.from_hp(
            self.weight_fp, self.block_size, float8_dtype=torch.float8_e5m2
        )
        self.assertEqual(tensor_e5m2.qdata_fp8.dtype, torch.float8_e5m2)


if __name__ == "__main__":
    common_utils.run_tests()
