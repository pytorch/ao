# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import run_tests

from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
    Int8Tensor,
    QuantizeTensorToInt8Kwargs,
)
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
class TestInt8Tensor(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.weight_fp = torch.randn(4, 3, dtype=torch.float32)
        self.input_fp = torch.randn(2, 3, dtype=torch.float32)
        self.bias = torch.randn(4)
        self.block_size = [4, 3]

    def test_creation_and_attributes(self):
        """Test tensor creation, dtypes, and ranges"""
        tensor = Int8Tensor.from_hp(self.weight_fp, self.block_size)

        self.assertEqual(tensor.shape, (4, 3))
        self.assertEqual(tensor.qdata.dtype, torch.int8)
        self.assertTrue(
            torch.all(tensor.qdata >= -128) and torch.all(tensor.qdata <= 127)
        )

    @common_utils.parametrize(
        "kernel_preference",
        [KernelPreference.AUTO, KernelPreference.TORCH, KernelPreference.FBGEMM],
    )
    def test_kernel_preference(self, kernel_preference):
        """Test Int8Tensor with different kernels"""
        tensor = Int8Tensor.from_hp(
            self.weight_fp, self.block_size, kernel_preference=kernel_preference
        )

        self.assertEqual(tensor.kernel_preference, kernel_preference)

    def test_linear_operations(self):
        """Test fp+int8 and int8+int8 linear ops with quantization error check"""
        weight_q8 = Int8Tensor.from_hp(self.weight_fp, self.block_size)
        input_q8 = Int8Tensor.from_hp(self.input_fp, self.block_size)

        reference = torch.nn.functional.linear(self.input_fp, self.weight_fp, self.bias)
        result_fp = torch.nn.functional.linear(self.input_fp, weight_q8, self.bias)
        result_q8 = torch.nn.functional.linear(input_q8, weight_q8, self.bias)

        self.assertEqual(result_fp.shape, reference.shape)
        self.assertEqual(result_q8.shape, reference.shape)
        self.assertTrue(compute_error(result_fp, reference) > 10)
        self.assertTrue(compute_error(result_q8, reference) > 10)

    def test_dynamic_quantization(self):
        weight_q8_dynamic = Int8Tensor.from_hp(
            self.weight_fp,
            self.block_size,
            act_quant_kwargs=QuantizeTensorToInt8Kwargs(),
        )

        reference = torch.nn.functional.linear(self.input_fp, self.weight_fp, self.bias)
        result_dynamic = torch.nn.functional.linear(
            self.input_fp, weight_q8_dynamic, self.bias
        )

        self.assertEqual(result_dynamic.shape, reference.shape)

    def test_error_handling_and_dequant(self):
        """Test input validation and dequantization accuracy"""
        # Test 1D tensor validation
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            Int8Tensor.from_hp(torch.randn(5), [1])

        # Test wrong block_size validation
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            Int8Tensor.from_hp(self.weight_fp, [1])

        # Test dequantization with exact values
        test_data = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        tensor = Int8Tensor.from_hp(test_data, [1, 1])

        dequantized = torch.ops.aten.dequantize.self(tensor)
        self.assertEqual(dequantized.shape, test_data.shape)
        self.assertLess(torch.abs(dequantized - test_data).max().item(), 0.1)


if __name__ == "__main__":
    run_tests()
