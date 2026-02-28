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

        # Check 2:4 pattern
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

        # Norm-based metrics for numerical stability
        error = (dequantized - w_sparse).abs()
        max_error = error.max()
        mean_error = error.mean()

        # Relative error using non-zero elements
        non_zero_mask = w_sparse != 0
        if non_zero_mask.any():
            rel_error = (error[non_zero_mask] / w_sparse[non_zero_mask].abs()).mean()
            self.assertLess(rel_error.item(), 0.1)

        self.assertLess(max_error.item(), 1.0)
        self.assertLess(mean_error.item(), 0.1)

    def test_invalid_dimensions(self):
        """Test dimension validation"""
        # Not multiple of 32
        invalid_weight = torch.randn(100, 100, dtype=self.dtype, device="cuda")

        with self.assertRaises(ValueError):
            Float8SemiSparseTensor.from_hp(invalid_weight, [1, 100])

    def test_device(self):
        """Test if device handler work"""
        # CPU tensor should be rejected
        cpu_weight = torch.randn(*self.shape, dtype=self.dtype, device="cpu")
        with self.assertRaises(ValueError):
            Float8SemiSparseTensor.from_hp(cpu_weight, self.block_size)

        # CUDA tensor components should all be on CUDA
        tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)
        self.assertEqual(tensor.qdata.device, tensor.qdata_compressed.device)
        self.assertEqual(tensor.qdata.device, tensor.scale.device)
        self.assertTrue(tensor.qdata.is_cuda)

    def test_w8a8_dynamic_activation(self):
        """Test W8A8-FP-CSR with dynamic activation quantization"""
        weight_tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)

        batch_size = 32
        in_features = self.shape[1]
        activation = torch.randn(
            batch_size, in_features, dtype=self.dtype, device="cuda"
        )
        output = torch.nn.functional.linear(activation, weight_tensor)

        expected_shape = (batch_size, self.shape[0])
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(output.isnan().any())
        self.assertFalse(output.isinf().any())

    def test_linear_with_bias(self):
        """Test linear operation with bias"""
        weight_tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)
        activation = torch.randn(32, self.shape[1], dtype=self.dtype, device="cuda")
        bias = torch.randn(self.shape[0], dtype=self.dtype, device="cuda")

        output = torch.nn.functional.linear(activation, weight_tensor, bias)

        self.assertEqual(output.shape, (32, self.shape[0]))
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(output.isnan().any())
        self.assertFalse(output.isinf().any())

    def test_batched_input(self):
        """Test 3D batched input"""
        weight_tensor = Float8SemiSparseTensor.from_hp(self.weight_fp, self.block_size)
        batch_dims = (4, 8)
        activation = torch.randn(
            *batch_dims, self.shape[1], dtype=self.dtype, device="cuda"
        )

        output = torch.nn.functional.linear(activation, weight_tensor)

        expected_shape = (*batch_dims, self.shape[0])
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, self.dtype)
        self.assertFalse(output.isnan().any())

    def test_zero_weight_validation(self):
        """Test scale validation with zero weights"""
        zero_weight = torch.zeros(*self.shape, dtype=self.dtype, device="cuda")

        with self.assertRaises(ValueError):
            Float8SemiSparseTensor.from_hp(zero_weight, self.block_size)


if __name__ == "__main__":
    common_utils.run_tests()
