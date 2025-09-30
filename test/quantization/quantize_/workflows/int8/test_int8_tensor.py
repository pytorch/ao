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
    PerRow,
    PerTensor,
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
        self.input_fp = torch.randn(4, 3, dtype=torch.float32)
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

    @unittest.skip("granularity parameter not supported in current API")
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_slice_preserves_aliasing(self, granularity):
        config = Int8DynamicActivationInt8WeightConfig(
            granularity=granularity, version=2
        )
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        # Making sure the aliasing is preserved in sliced quantized Tensor
        assert param.data.qdata.data_ptr() == param_data.qdata.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()

    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    @common_utils.parametrize("device", ["cpu", "cuda"])
    @common_utils.parametrize("dtype", [torch.bfloat16])
    def test_slice(self, config, device, dtype):
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)
        dummy1 = torch.nn.Linear(256, 64, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 64), requires_grad=False
        )
        dummy2 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 128), requires_grad=False
        )

        quantize_(dummy, config)
        weight1 = dummy.weight.clone().narrow(0, 0, 64)
        weight2 = dummy.weight.clone().narrow(1, 0, 128)
        self.assertEqual(
            weight1.qdata,
            dummy.weight.qdata.narrow(0, 0, 64),
        )
        self.assertEqual(
            weight2.qdata,
            dummy.weight.qdata.narrow(1, 0, 128),
        )

        # check for sliced weight, before and after int8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1.contiguous(), requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 20, f"sqnr: {sqnr}")

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2.contiguous(), requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 15, f"sqnr: {sqnr}")

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
        tensor = Int8Tensor.from_hp(test_data, [1, 2])

        dequantized = torch.ops.aten.dequantize.self(tensor)
        self.assertEqual(dequantized.shape, test_data.shape)
        self.assertLess(torch.abs(dequantized - test_data).max().item(), 0.1)


if __name__ == "__main__":
    common_utils.run_tests()
