# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
)

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import copy
import io
import random
import unittest
from contextlib import nullcontext
from functools import partial
from typing import Tuple

import pytest
import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal import common_utils

from torchao.dtypes.floatx.float8_layout import Float8AQTTensorImpl
from torchao.float8.float8_utils import compute_error
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    quantize_,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_api import (
    float8_static_activation_float8_weight,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    choose_qparams_affine_float8,
    dequantize_affine_float8,
    quantize_affine_float8,
)
from torchao.utils import (
    is_sm_at_least_89,
    is_sm_at_least_90,
    is_sm_version,
)

random.seed(0)
torch.manual_seed(0)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestAffineQuantizedFloat8Compile(InductorTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only", "static"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_fp8_linear_variants(
        self, dtype: torch.dtype, mode: str, compile: bool, sizes: Tuple, granularity
    ):
        error_message = None
        if isinstance(granularity, PerRow):
            if mode == "dynamic" and dtype != torch.bfloat16:
                error_message = "PerRow quantization only works for bfloat16 precision"
            elif mode == "static":
                error_message = (
                    "Static quantization only supports PerTensor granularity"
                )

        error_context = (
            pytest.raises(AssertionError, match=error_message)
            if error_message
            else nullcontext()
        )

        with error_context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
            # Get a "reasonable" scale for the input tensor even though
            # we use the same scale for multiple activations
            scale, _ = choose_qparams_affine(
                input_tensor,
                MappingType.SYMMETRIC,
                input_tensor.shape,
                torch.float8_e4m3fn,
                scale_dtype=torch.float32,
            )
            mode_map = {
                "dynamic": partial(
                    float8_dynamic_activation_float8_weight, granularity=granularity
                ),
                "weight-only": float8_weight_only,
                "static": partial(
                    float8_static_activation_float8_weight,
                    scale=scale,
                    granularity=granularity,
                ),
            }

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

            quantized_model = copy.deepcopy(model)
            factory = mode_map[mode]()
            quantize_(quantized_model, factory)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert compute_error(output_original, output_quantized) > 20, (
                f"Quantization error is too high got a SQNR of {error}"
            )

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity specification"):
            float8_dynamic_activation_float8_weight(granularity="invalid")

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_mismatched_granularity(self):
        with pytest.raises(
            ValueError,
            match="Different granularities for activation and weight are not supported",
        ):
            float8_dynamic_activation_float8_weight(granularity=(PerTensor(), PerRow()))

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_unsupported_granularity(self):
        class UnsupportedGranularity:
            pass

        with pytest.raises(ValueError, match="Invalid granularity types"):
            float8_dynamic_activation_float8_weight(
                granularity=(UnsupportedGranularity(), UnsupportedGranularity())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_per_row_with_float32(self):
        with pytest.raises(
            AssertionError,
            match="PerRow quantization only works for bfloat16 precision",
        ):
            model = ToyLinearModel(64, 64).eval().to(torch.float32).to("cuda")
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=PerRow())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("mode", ["dynamic", "weight-only", "static"])
    def test_serialization(self, mode: str):
        # Create and quantize the model
        model = ToyLinearModel(16, 32).to(device="cuda")

        mode_map = {
            "dynamic": partial(
                float8_dynamic_activation_float8_weight, granularity=PerTensor()
            ),
            "weight-only": float8_weight_only,
            "static": partial(
                float8_static_activation_float8_weight,
                scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
                granularity=PerTensor(),
            ),
        }
        factory = mode_map[mode]()
        quantize_(model, factory)

        # Save the state dict to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)

        # Reset the buffer position
        buffer.seek(0)

        # Load the state dict from the buffer
        weights_only_load = True
        loaded_state_dict = torch.load(buffer, weights_only=weights_only_load)

        # Create a new model and load the state dict
        with torch.device("meta"):
            new_model = ToyLinearModel(16, 32)
            if mode == "static":
                quantize_(new_model, factory)
            new_model.load_state_dict(loaded_state_dict, assign=True)

        # Compare the original and loaded models
        for layer_name in ["linear1", "linear2"]:
            original_layer = getattr(model, layer_name)
            new_layer = getattr(new_model, layer_name)

            # Compare weights
            if mode == "weight-only":
                original_weight = original_layer.weight.tensor_impl.float8_data.to(
                    torch.float32
                )
                new_weight = new_layer.weight.tensor_impl.float8_data.to(torch.float32)
            else:
                original_weight = original_layer.weight.original_weight_tensor.tensor_impl.float8_data.to(
                    torch.float32
                )
                new_weight = (
                    new_layer.weight.original_weight_tensor.tensor_impl.float8_data.to(
                        torch.float32
                    )
                )

            assert torch.allclose(original_weight, new_weight), (
                f"Weights do not match for {layer_name}"
            )

            # Compare scales
            if hasattr(original_layer.weight, "scale"):
                assert torch.allclose(
                    original_layer.weight.scale, new_layer.weight.scale
                ), f"Scales do not match for {layer_name}"

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_fp8_weight_dimension_warning(self):
        # Create model with incompatible dimensions (not multiples of 16)
        model = ToyLinearModel(10, 25).cuda()  # 10x25 and 25x10 weights

        # Set up logging capture
        with self.assertLogs(
            "torchao.quantization.quant_api", level="INFO"
        ) as log_context:
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=PerTensor())
            )
            print(model)

        # Verify warning messages for both layers
        expected_messages = [
            "Skipping float8 quantization: weight shape torch.Size([25, 10])",
            "Skipping float8 quantization: weight shape torch.Size([10, 25])",
        ]
        # Check that we got warnings for both incompatible layers
        warning_count = sum(
            1 for msg in log_context.output if "Skipping float8 quantization" in msg
        )
        self.assertEqual(warning_count, 2, "Expected warnings for both linear layers")

        # Check warning message content
        for expected in expected_messages:
            self.assertTrue(
                any(expected in msg for msg in log_context.output),
                f"Expected warning message containing: {expected}",
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize(
        "in_features,out_features", [(512, 1024), (256, 768), (1024, 512)]
    )
    @common_utils.parametrize(
        "leading_shape", [(1,), (8,), (16,), (2, 8,), (2, 2, 16,)]
    )  # fmt: skip
    @common_utils.parametrize("bias", [True, False])
    def test_mm_float8dq_per_row(
        self, in_features, out_features, leading_shape, bias: bool
    ):
        device = "cuda"
        dtype = torch.bfloat16
        input_shape = leading_shape + (in_features,)

        ref_linear = (
            torch.nn.Linear(in_features, out_features, bias=bias).to(device).to(dtype)
        )
        test_linear = copy.deepcopy(ref_linear)
        quantize_(
            test_linear, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        )

        quant_weight = test_linear.weight

        self.assertTrue(hasattr(quant_weight, "original_weight_tensor"))
        weight_impl = quant_weight.original_weight_tensor.tensor_impl

        self.assertTrue(hasattr(weight_impl, "float8_data"))
        self.assertTrue(hasattr(weight_impl, "scale"))
        self.assertFalse(weight_impl.transposed)

        # Verify scale shape for row-wise quantization
        expected_scale_shape = (out_features, 1)
        actual_scale_shape = weight_impl.scale.shape
        self.assertEqual(actual_scale_shape, expected_scale_shape)

        self.assertEqual(weight_impl.float8_data.shape, (out_features, in_features))

        input_tensor = torch.randn(*input_shape, device=device, dtype=dtype)

        with torch.no_grad():
            ref_output = ref_linear(input_tensor)
            quant_output = torch.nn.functional.linear(input_tensor, quant_weight)

        expected_output_shape = input_tensor.shape[:-1] + (out_features,)
        self.assertEqual(quant_output.shape, expected_output_shape)

        error = compute_error(ref_output, quant_output)
        assert error > 20, f"Quantization error is too high got a SQNR of {error}"

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @common_utils.parametrize("output_dtype", [torch.float32, torch.bfloat16])
    @common_utils.parametrize("block_size", [None, (1, 32), (2, 16), (4, 8)])
    def test_dequantize_affine_float8(self, float8_dtype, output_dtype, block_size):
        """Test dequantize_affine_float8 with various configurations"""

        device = "cuda"
        input_tensor = torch.randn(8, 64, device=device, dtype=torch.float32)

        # Choose quantization parameters
        scale = choose_qparams_affine_float8(
            input_tensor, float8_dtype=float8_dtype, block_size=block_size
        )

        # Quantize
        quantized = quantize_affine_float8(input_tensor, scale, float8_dtype)

        # Dequantize
        dequantized = dequantize_affine_float8(quantized, scale, output_dtype)

        # Verify output properties
        self.assertEqual(dequantized.dtype, output_dtype)
        self.assertEqual(dequantized.shape, input_tensor.shape)
        self.assertEqual(dequantized.device, input_tensor.device)

        # Verify quantization/dequantization roundtrip is reasonable
        error = torch.abs(input_tensor.to(output_dtype) - dequantized).mean()
        self.assertLess(error, 0.1, "Quantization error too high")

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_dequantize_affine_float8_scale_broadcasting(self):
        """Test that scale broadcasting works correctly for block-wise quantization"""
        device = "cuda"
        # Create input tensor with known block structure
        input_tensor = torch.randn(4, 32, device=device, dtype=torch.float32)
        block_size = (2, 16)  # 2x2 blocks in first dim, 2x16 blocks in second dim

        # Choose quantization parameters
        scale = choose_qparams_affine_float8(
            input_tensor, float8_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # Verify scale shape
        expected_scale_shape = (
            input_tensor.shape[0] // block_size[0],
            input_tensor.shape[1] // block_size[1],
        )
        self.assertEqual(scale.shape, expected_scale_shape)

        # Quantize
        quantized = quantize_affine_float8(input_tensor, scale, torch.float8_e4m3fn)

        # Dequantize
        dequantized = dequantize_affine_float8(quantized, scale, torch.float32)

        # Verify shapes match
        self.assertEqual(dequantized.shape, input_tensor.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_float8_tensor_slicing_basic(self, granularity):
        """Test basic slicing operations on Float8 tensors"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create and quantize a model
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)
        quantize_(
            model, Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        )

        weight_impl = model.weight.original_weight_tensor.tensor_impl

        # Test dimension 0 slicing (rows)
        sliced_0 = weight_impl[10:20]
        self.assertEqual(sliced_0.shape, (10, 64))

        # Test dimension 1 slicing (columns)
        sliced_1 = weight_impl[:, 20:40]
        self.assertEqual(sliced_1.shape, (32, 20))

        # Test combined slicing
        sliced_both = weight_impl[5:15, 10:30]
        self.assertEqual(sliced_both.shape, (10, 20))

        # Verify the sliced tensors are still Float8 tensors
        self.assertTrue(isinstance(sliced_0, Float8AQTTensorImpl))
        self.assertTrue(isinstance(sliced_1, Float8AQTTensorImpl))
        self.assertTrue(isinstance(sliced_both, Float8AQTTensorImpl))

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_float8_tensor_slicing_per_tensor(self):
        """Test slicing with per-tensor quantization (scale should not change)"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create and quantize with per-tensor granularity
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)
        quantize_(
            model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        )

        original_weight = model.weight
        original_impl = original_weight.original_weight_tensor.tensor_impl
        original_scale = original_impl.scale

        # Test slicing
        sliced_weight = original_weight[10:20, 20:40]
        sliced_impl = sliced_weight.original_weight_tensor.tensor_impl

        # For per-tensor quantization, scale should be identical
        self.assertTrue(torch.equal(original_scale, sliced_impl.scale))
        self.assertEqual(sliced_impl.scale.numel(), 1)

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @unittest.skipIf(
        not is_sm_at_least_90(),
        "Per-row quantization requires compute capability >= 9.0",
    )
    def test_float8_tensor_slicing_per_row(self):
        """Test slicing with per-row quantization (scale should be sliced appropriately)"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create and quantize with per-row granularity
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)
        quantize_(
            model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        )

        original_weight = model.weight  # Shape: (32, 64)
        original_impl = original_weight.original_weight_tensor.tensor_impl
        original_scale = original_impl.scale  # Shape: (32, 1)

        # Test row slicing (dimension 0)
        sliced_rows = original_weight[10:20]  # Shape: (10, 64)
        sliced_impl = sliced_rows.original_weight_tensor.tensor_impl

        # Scale should be sliced to match the rows
        expected_scale_shape = (10, 1)
        self.assertEqual(sliced_impl.scale.shape, expected_scale_shape)

        # Verify the scale values are correct (should be subset of original)
        self.assertTrue(torch.equal(sliced_impl.scale, original_scale[10:20]))

        # Test column slicing (dimension 1) - scale should not change for per-row
        sliced_cols = original_weight[:, 20:40]  # Shape: (32, 20)
        sliced_cols_impl = sliced_cols.original_weight_tensor.tensor_impl

        # Scale shape should remain the same since we're not changing rows
        self.assertEqual(sliced_cols_impl.scale.shape, (32, 1))
        self.assertTrue(torch.equal(sliced_cols_impl.scale, original_scale))

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_float8_tensor_slicing_edge_cases(self):
        """Test edge cases in slicing"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create and quantize a model
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)
        quantize_(
            model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        )

        original_weight = model.weight

        # Test empty slice
        empty_slice = original_weight[0:0]
        self.assertEqual(empty_slice.shape, (0, 64))

        # Test single element slice
        single_row = original_weight[0:1]
        self.assertEqual(single_row.shape, (1, 64))

        # Test out of bounds (should be handled by PyTorch)
        large_slice = original_weight[:100]  # More than available rows
        self.assertEqual(large_slice.shape, (32, 64))  # Should clamp to available

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    @unittest.skipIf(
        is_sm_version(8, 9),
        "TODO: AssertionError: tensor(-2.1562, device='cuda:0', dtype=torch.bfloat16) not greater than 15",
    )
    def test_float8_tensor_slicing_functional_correctness(self, granularity):
        """Test that sliced tensors produce correct results in computations"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create reference and quantized models with dimensions that are multiples of 16
        ref_model = (
            torch.nn.Linear(64, 48, bias=False).to(device).to(dtype)
        )  # 48 is divisible by 16
        quant_model = copy.deepcopy(ref_model)
        quantize_(
            quant_model,
            Float8DynamicActivationFloat8WeightConfig(granularity=granularity),
        )

        # Create input with batch size that works well with slicing
        input_tensor = torch.randn(8, 64, device=device, dtype=dtype)

        ref_weight_slice = ref_model.weight[0:16, 0:32]
        quant_weight_slice = quant_model.weight[0:16, 0:32]

        # Verify that the sliced weights maintain Float8 properties
        self.assertTrue(hasattr(quant_weight_slice, "original_weight_tensor"))
        sliced_impl = quant_weight_slice.original_weight_tensor.tensor_impl
        self.assertTrue(isinstance(sliced_impl, Float8AQTTensorImpl))

        # Verify sliced weight shapes
        self.assertEqual(sliced_impl.float8_data.shape, (16, 32))

        # Get original quantized weight implementation for scale comparison
        original_quant_impl = quant_model.weight.original_weight_tensor.tensor_impl

        # Verify scale properties based on granularity
        if isinstance(granularity, PerTensor):
            # Per-tensor: scale should be identical to original (scalar)
            self.assertEqual(sliced_impl.scale.numel(), 1)
            self.assertTrue(torch.equal(sliced_impl.scale, original_quant_impl.scale))
        else:  # PerRow
            # Per-row: scale should be sliced to match the selected rows (0:16)
            expected_scale_shape = (16, 1)
            self.assertEqual(sliced_impl.scale.shape, expected_scale_shape)
            # Verify the scale values are the correct slice from the original
            self.assertTrue(
                torch.equal(sliced_impl.scale, original_quant_impl.scale[0:16])
            )

        # Verify that sliced quantized data matches the correct slice from original
        original_float8_data_slice = original_quant_impl.float8_data[0:16, 0:32]
        self.assertTrue(
            torch.equal(sliced_impl.float8_data, original_float8_data_slice)
        )

        # Verify that sliced weights can be converted back to float with correct values
        sliced_float_weight = quant_weight_slice.to(dtype)
        self.assertEqual(sliced_float_weight.shape, (16, 32))
        self.assertEqual(sliced_float_weight.dtype, dtype)

        input_slice = input_tensor[:, 0:32]  # (8, 32) to match sliced weight

        # Compute with sliced weights
        with torch.no_grad():
            ref_output = torch.nn.functional.linear(input_slice, ref_weight_slice)
            quant_output = torch.nn.functional.linear(input_slice, quant_weight_slice)

        # Verify shapes
        expected_shape = (8, 16)  # batch_size x out_features_sliced
        self.assertEqual(ref_output.shape, expected_shape)
        self.assertEqual(quant_output.shape, expected_shape)

        # Verify reasonable quantization error
        error = compute_error(ref_output, quant_output)
        self.assertGreater(error, 15, f"Quantization SQNR too low: {error}")

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_power_of_2_scaling_weight_only(self):
        """Test that Float8WeightOnlyConfig with round_scales_to_power_of_2=True works correctly"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create model
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)

        # Test with round_scales_to_power_of_2=True
        config = Float8WeightOnlyConfig(round_scales_to_power_of_2=True)
        quantized_model = copy.deepcopy(model)
        quantize_(quantized_model, config)

        # Verify the model was quantized
        self.assertTrue(hasattr(quantized_model.weight, "tensor_impl"))
        weight_impl = quantized_model.weight.tensor_impl
        self.assertTrue(hasattr(weight_impl, "scale"))

        # Check that scales are powers of 2
        scale = weight_impl.scale.float()
        # For power of 2, log2(scale) should be integer
        log2_scale = torch.log2(scale)
        is_power_of_2 = torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
        self.assertTrue(is_power_of_2, "Scales should be powers of 2")

        # Test inference works
        input_tensor = torch.randn(8, 64, device=device, dtype=dtype)
        with torch.no_grad():
            ref_output = model(input_tensor)
            quant_output = quantized_model(input_tensor)

        # Verify shapes match
        self.assertEqual(ref_output.shape, quant_output.shape)

        # Verify reasonable quantization error
        error = compute_error(ref_output, quant_output)
        self.assertGreater(error, 15, f"Quantization SQNR too low: {error}")

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_power_of_2_scaling_backward_compatibility(self):
        """Test that default behavior (round_scales_to_power_of_2=False) is unchanged"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create model
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)

        # Test default behavior (should be False)
        config_default = Float8WeightOnlyConfig()
        quantized_model_default = copy.deepcopy(model)
        quantize_(quantized_model_default, config_default)

        # Test explicit False
        config_false = Float8WeightOnlyConfig(round_scales_to_power_of_2=False)
        quantized_model_false = copy.deepcopy(model)
        quantize_(quantized_model_false, config_false)

        # Get scales from both models
        scale_default = quantized_model_default.weight.tensor_impl.scale
        scale_false = quantized_model_false.weight.tensor_impl.scale

        # They should be identical (backward compatibility)
        self.assertTrue(torch.allclose(scale_default, scale_false))

        # Test that they produce the same results
        input_tensor = torch.randn(8, 64, device=device, dtype=dtype)
        with torch.no_grad():
            output_default = quantized_model_default(input_tensor)
            output_false = quantized_model_false(input_tensor)

        # Outputs should be identical
        self.assertTrue(torch.allclose(output_default, output_false))

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_power_of_2_vs_regular_scaling(self):
        """Test that power of 2 scaling produces different (but reasonable) results compared to regular scaling"""
        device = "cuda"
        dtype = torch.bfloat16

        # Create model
        model = torch.nn.Linear(64, 32, bias=False).to(device).to(dtype)

        # Test with regular scaling
        config_regular = Float8WeightOnlyConfig(round_scales_to_power_of_2=False)
        quantized_model_regular = copy.deepcopy(model)
        quantize_(quantized_model_regular, config_regular)

        # Test with power of 2 scaling
        config_power2 = Float8WeightOnlyConfig(round_scales_to_power_of_2=True)
        quantized_model_power2 = copy.deepcopy(model)
        quantize_(quantized_model_power2, config_power2)

        # Get scales from both models
        scale_regular = quantized_model_regular.weight.tensor_impl.scale.float()
        scale_power2 = quantized_model_power2.weight.tensor_impl.scale.float()

        # Power of 2 scale should be different from regular scale (unless it was already power of 2)
        # But the power of 2 scale should be <= regular scale (since we round down)
        self.assertTrue(torch.all(scale_power2 <= scale_regular))

        # Verify power of 2 scale is actually power of 2
        log2_scale = torch.log2(scale_power2)
        is_power_of_2 = torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
        self.assertTrue(is_power_of_2, "Power-of-2 scales should be powers of 2")

        # Test that both produce reasonable results
        input_tensor = torch.randn(8, 64, device=device, dtype=dtype)
        with torch.no_grad():
            ref_output = model(input_tensor)
            output_regular = quantized_model_regular(input_tensor)
            output_power2 = quantized_model_power2(input_tensor)

        # Both should have reasonable quantization error
        error_regular = compute_error(ref_output, output_regular)
        error_power2 = compute_error(ref_output, output_power2)

        self.assertGreater(
            error_regular, 15, f"Regular quantization SQNR too low: {error_regular}"
        )
        self.assertGreater(
            error_power2, 15, f"Power-of-2 quantization SQNR too low: {error_power2}"
        )


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])
