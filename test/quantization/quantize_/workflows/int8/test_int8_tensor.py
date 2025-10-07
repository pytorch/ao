# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
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
from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor
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
        self.weight_fp = torch.randn(4, 3, dtype=torch.bfloat16)
        self.input_fp = torch.randn(4, 3, dtype=torch.bfloat16)
        self.bias = torch.randn(4, dtype=torch.bfloat16)
        self.block_size = [4, 3]

    def test_creation_and_attributes(self):
        """Test tensor creation, dtypes, and ranges"""
        tensor = Int8Tensor.from_hp(self.weight_fp, self.block_size)

        self.assertEqual(tensor.shape, (4, 3))
        self.assertEqual(tensor.qdata.dtype, torch.int8)
        self.assertTrue(
            torch.all(tensor.qdata >= -128) and torch.all(tensor.qdata <= 127)
        )

    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
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
        sizes: Tuple,
        config,
    ):
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
        assert error > 20, f"Quantization error is too high got a SQNR of {error}"

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
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_slice(self, config, device, dtype):
        """Test tensor slicing"""
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)
        quantize_(dummy, config)

        weight1 = dummy.weight.clone().narrow(0, 0, 64)
        weight2 = dummy.weight.clone().narrow(1, 0, 128)

        self.assertEqual(weight1.qdata, dummy.weight.qdata.narrow(0, 0, 64))
        self.assertEqual(weight2.qdata, dummy.weight.qdata.narrow(1, 0, 128))

        # Int8DynamicActivationInt8WeightConfig uses per-row (PerRow)
        # Int8WeightOnlyConfig uses per-tensor (PerTensor)
        if isinstance(config, Int8DynamicActivationInt8WeightConfig):
            # PerRow: dim 0 slicing affects scale, dim 1 doesn't
            self.assertEqual(weight1.scale, dummy.weight.scale.narrow(0, 0, 64))
            self.assertEqual(weight2.scale, dummy.weight.scale)
        else:
            # PerTensor: scale unchanged by slicing
            self.assertEqual(weight1.scale, dummy.weight.scale)
            self.assertEqual(weight2.scale, dummy.weight.scale)

    def test_index_select(self):
        """test that `x_0 = x[0]` works when `x` is a 2D `Int8Tensor`."""
        N, K = 256, 512
        x = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        x_int8 = Int8Tensor.from_hp(x, block_size=[N, K])
        x_int8_0 = x_int8[0]
        torch.testing.assert_close(
            x_int8.dequantize()[0], x_int8_0.dequantize(), atol=0, rtol=0
        )

    def test_error_handling_and_dequant(self):
        """Test input validation and dequantization accuracy"""
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            Int8Tensor.from_hp(torch.randn(5), [1])

        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            Int8Tensor.from_hp(self.weight_fp, [1])

        test_data = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
        tensor = Int8Tensor.from_hp(test_data, [1, 2])

        dequantized = torch.ops.aten.dequantize.self(tensor)
        self.assertEqual(dequantized.shape, test_data.shape)
        self.assertLess(torch.abs(dequantized - test_data).max().item(), 0.1)


if __name__ == "__main__":
    common_utils.run_tests()
