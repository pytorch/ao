# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from contextlib import nullcontext
from typing import Tuple

import torch
from torch.testing._internal import common_utils

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
    Int8Tensor,
    QuantizeTensorToInt8Kwargs,
)
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase


# TODO: Refactor after https://github.com/pytorch/ao/pull/2729 is merged
class ToyTwoLinearModel(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        has_bias=False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.linear1 = torch.nn.Linear(
            input_dim, hidden_dim, bias=has_bias, dtype=dtype, device=device
        )
        self.linear2 = torch.nn.Linear(
            hidden_dim, output_dim, bias=has_bias, dtype=dtype, device=device
        )

    # Note: tinygemm kernel only uses bfloat16 inputs
    def example_inputs(self, batch_size=1):
        return (
            torch.randn(
                batch_size,
                self.linear1.in_features,
                dtype=self.dtype,
                device=self.device,
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@common_utils.instantiate_parametrized_tests
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

    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("compile", [False, True])
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    def test_int8_linear_variants(
        self,
        dtype: torch.dtype,
        compile: bool,
        sizes: Tuple,
        config,
    ):
        error_message = None

        error_context = (
            self.assertRaisesRegex(AssertionError, error_message)
            if error_message
            else nullcontext()
        )

        with error_context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")

            # Create a linear layer
            m = ToyTwoLinearModel(K, N, K).eval().to(dtype).to("cuda")
            m_q = copy.deepcopy(m)

            # Quantize
            quantize_(m_q, config)

            output_original = m(input_tensor)
            output_quantized = m_q(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert compute_error(output_original, output_quantized) > 20, (
                f"Quantization error is too high got a SQNR of {error}"
            )

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
    common_utils.run_tests()
