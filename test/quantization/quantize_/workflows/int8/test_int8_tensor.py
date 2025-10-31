# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch.testing._internal import common_utils

from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
    Int8Tensor,
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

        self.test_shape = (4, 3)
        self.dtype = torch.bfloat16
        self.batch_size = 32

        torch.manual_seed(42)
        self.weight_fp = torch.randn(*self.test_shape, dtype=self.dtype)
        self.input_fp = torch.randn(*self.test_shape, dtype=self.dtype)
        self.bias = torch.randn(self.test_shape[0], dtype=self.dtype)
        self.block_size = list(self.test_shape)

    def test_creation_and_attributes(self):
        """Test tensor creation, dtypes, and ranges"""
        tensor = Int8Tensor.from_hp(self.weight_fp, self.block_size)

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
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),  # 2D
            ((32, 128), 64, 256),  # 3D
        ],
    )
    def test_int8_linear_quantization_accuracy(
        self,
        dtype: torch.dtype,
        sizes: tuple,
        config,
    ):
        """Test quantization preserves reasonable accuracy"""
        M, N, K = sizes
        input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")

        # Create a linear layer
        m = ToyTwoLinearModel(K, N, K, dtype=dtype, device="cuda").eval()
        m_q = copy.deepcopy(m)

        # Quantize
        quantize_(m_q, config)

        output_fp = m(input_tensor)
        output_quantized = m_q(input_tensor)

        error = compute_error(output_fp, output_quantized)
        assert error > 20, (
            f"Quantization quality is too low, SQNR: {error}dB (expected > {20}dB)"
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

        # Dynamic: per-row (1D scale [N]), Weight-only: per-tensor (scalar)
        if isinstance(config, Int8DynamicActivationInt8WeightConfig):
            self.assertEqual(linear.weight.scale.shape, (N,))
            self.assertEqual(linear.weight.scale.ndim, 1)
        else:
            self.assertEqual(linear.weight.scale.numel(), 1)

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

        # Int8DynamicActivationInt8WeightConfig uses per-row (PerRow)
        # Int8WeightOnlyConfig uses per-tensor (PerTensor)
        if isinstance(config, Int8DynamicActivationInt8WeightConfig):
            # PerRow: dim 0 slicing affects scale, dim 1 doesn't
            self.assertEqual(
                weight1.scale, dummy.weight.scale.narrow(0, 0, slice_sizes[0])
            )
            self.assertEqual(weight2.scale, dummy.weight.scale)
        else:
            # PerTensor: scale unchanged by slicing
            self.assertEqual(weight1.scale, dummy.weight.scale)
            self.assertEqual(weight2.scale, dummy.weight.scale)
        with self.assertRaises(NotImplementedError):
            _ = dummy.weight[::2]

    def test_index_select(self):
        """test that `x_0 = x[0]` works when `x` is a 2D `Int8Tensor`."""
        N, K = 256, 512
        x = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        x_int8 = Int8Tensor.from_hp(x, block_size=[N, K])
        x_int8_0 = x_int8[0]
        torch.testing.assert_close(
            x_int8.dequantize()[0], x_int8_0.dequantize(), atol=0, rtol=0
        )

    def test_invalid_input_handling(self):
        """Test input validation with specific error types"""
        invalid_tensor = torch.randn(5)
        incompatible_block_size = [1]

        with self.assertRaises(
            ValueError, msg="Should reject incompatible tensor dimensions"
        ):
            Int8Tensor.from_hp(invalid_tensor, incompatible_block_size)

        with self.assertRaises(
            ValueError, msg="Should reject mismatched block size dimensions"
        ):
            Int8Tensor.from_hp(self.weight_fp, [1])

    def test_dequantization_accuracy(self):
        """Test dequantization accuracy separately"""
        test_data = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
        tensor = Int8Tensor.from_hp(test_data, [1, 2])

        dequantized = tensor.dequantize()
        self.assertEqual(dequantized.shape, test_data.shape)
        self.assertLess(
            torch.abs(dequantized - test_data).max().item(),
            0.1,
            msg=f"Dequantization error exceeds tolerance of {0.1}",
        )


if __name__ == "__main__":
    common_utils.run_tests()
