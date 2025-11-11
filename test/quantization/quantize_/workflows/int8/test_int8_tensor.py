# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal import common_utils

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
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

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@common_utils.instantiate_parametrized_tests
class TestInt8Tensor(TorchAOIntegrationTestCase):
    def setUp(self):
        super().setUp()

        self.test_shape = (32, 20)
        self.dtype = torch.bfloat16
        self.batch_size = 32

        torch.manual_seed(42)
        self.weight_fp = torch.randn(*self.test_shape, dtype=self.dtype)
        self.input_fp = torch.randn(*self.test_shape, dtype=self.dtype)
        self.bias = torch.randn(self.test_shape[0], dtype=self.dtype)
        self.block_size = list(self.test_shape)

    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    def test_creation_and_attributes(self, config):
        """Test tensor creation, dtypes, and ranges"""
        linear = torch.nn.Linear(
            self.test_shape[1],
            self.test_shape[0],
            bias=False,
            dtype=self.dtype,
            device="cuda",
        )
        linear.weight.data = self.weight_fp.cuda()
        quantize_(linear, config)

        tensor = linear.weight

        self.assertEqual(tensor.shape, self.test_shape)
        self.assertEqual(tensor.qdata.dtype, torch.int8)
        self.assertTrue(
            torch.all(tensor.qdata >= -128) and torch.all(tensor.qdata <= 127)
        )

    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),  # 2D
            ((32, 128), 64, 256),  # 3D
        ],
    )
    def test_int8_linear_variants(
        self,
        dtype: torch.dtype,
        config,
        compile: bool,
        sizes: tuple,
    ):
        """Test linear operation supports including shape and compile"""
        M, N, K = sizes
        input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
        model = ToyTwoLinearModel(K, N, K, dtype=dtype, device="cuda").eval()
        model_q = copy.deepcopy(model)

        quantize_(model_q, config)

        if compile:
            model_q = torch.compile(model_q, fullgraph=True)

        output_fp = model(input_tensor)
        output_quantized = model_q(input_tensor)

        assert compute_error(output_fp, output_quantized) > 20, (
            f"Quantization error is too high got a SQNR of {compute_error(output_fp, output_quantized)}"
        )

    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float16])
    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    def test_per_row_scale_shape(self, dtype, config):
        """Test per-row quantization maintains 1D scale"""
        N, K = 64, 128
        linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device="cuda")
        quantize_(linear, config)

        self.assertEqual(linear.weight.scale.shape, (N,))
        self.assertEqual(linear.weight.scale.ndim, 1)

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
        """Test tensor slicing with per-row quantization"""
        tensor_size = 256
        slice_sizes = (64, 128)

        dummy = torch.nn.Linear(
            tensor_size, tensor_size, bias=False, dtype=dtype, device=device
        )
        quantize_(dummy, config)

        weight1 = dummy.weight.clone().narrow(0, 0, slice_sizes[0])
        weight2 = dummy.weight.clone().narrow(1, 0, slice_sizes[1])

        self.assertEqual(weight1.qdata, dummy.weight.qdata.narrow(0, 0, slice_sizes[0]))
        self.assertEqual(weight2.qdata, dummy.weight.qdata.narrow(1, 0, slice_sizes[1]))
        self.assertEqual(weight1.scale, dummy.weight.scale.narrow(0, 0, slice_sizes[0]))
        self.assertEqual(weight2.scale, dummy.weight.scale)
        with self.assertRaises(NotImplementedError):
            _ = dummy.weight[::2]

    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    def test_index_select(self, config):
        """test that `x_0 = x[0]` works when `x` is a 2D quantized tensor."""
        N, K = 256, 512
        x = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        linear = torch.nn.Linear(K, N, bias=False, dtype=torch.bfloat16, device="cuda")
        linear.weight.data = x
        quantize_(linear, config)

        x_int8 = linear.weight
        x_int8_0 = x_int8[0]
        torch.testing.assert_close(
            x_int8.dequantize()[0], x_int8_0.dequantize(), atol=0, rtol=0
        )

    @common_utils.parametrize(
        "config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8WeightOnlyConfig(version=2),
        ],
    )
    def test_dequantization_accuracy(self, config):
        """Test dequantization accuracy separately"""
        test_data = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16, device="cuda")
        linear = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16, device="cuda")
        linear.weight.data = test_data
        quantize_(linear, config)

        tensor = linear.weight
        dequantized = tensor.dequantize()
        self.assertEqual(dequantized.shape, test_data.shape)
        assert compute_error(dequantized, test_data) > 20, (
            f"Dequantization error is too high to get a SQNR of {compute_error(dequantized, test_data)}"
        )

    @common_utils.parametrize(
        "kernel",
        ["triton_per_fused", "extern_kernels._int_mm", "triton_poi_fused"],
    )
    def test_available_gpu_kernels(self, kernel):
        """Check which GPU kernels are available"""
        M, K, N = 128, 256, 512
        m = torch.nn.Sequential(
            torch.nn.Linear(K, N, device="cuda", dtype=torch.bfloat16)
        )
        config = Int8DynamicActivationInt8WeightConfig(version=2)
        quantize_(m, config)
        m = torch.compile(m)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        out, code = run_and_get_code(m, x)
        FileCheck().check(kernel).run(code[0])


if __name__ == "__main__":
    common_utils.run_tests()
