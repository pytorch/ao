# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
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
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal import common_utils

from torchao.dtypes.floatx.float8_layout import preprocess_scale
from torchao.float8.float8_utils import compute_error
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    quantize_,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    _choose_scale_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
    choose_qparams_affine,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import (
    get_current_accelerator_device,
    is_sm_at_least_89,
    is_sm_at_least_90,
)

random.seed(0)
torch.manual_seed(0)
_DEVICE = get_current_accelerator_device()


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
    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["static"])
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
            input_tensor = torch.randn(*M, K, dtype=dtype, device=_DEVICE)
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
                "static": partial(
                    Float8StaticActivationFloat8WeightConfig,
                    scale=scale,
                    granularity=granularity,
                ),
            }

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to(_DEVICE)

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
        _DEVICE == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity specification"):
            Float8DynamicActivationFloat8WeightConfig(granularity="invalid")

    @unittest.skipIf(
        _DEVICE == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    def test_mismatched_granularity(self):
        with pytest.raises(
            ValueError,
            match="Unsupported granularity types",
        ):
            Float8DynamicActivationFloat8WeightConfig(
                granularity=(PerTensor(), PerRow())
            )

    @unittest.skipIf(
        _DEVICE == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    def test_unsupported_granularity(self):
        class UnsupportedGranularity:
            pass

        with pytest.raises(ValueError, match="Unsupported granularity types"):
            Float8DynamicActivationFloat8WeightConfig(
                granularity=(UnsupportedGranularity(), UnsupportedGranularity()),
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
            model = ToyLinearModel(64, 64).eval().to(torch.float32).to(_DEVICE)
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("mode", ["static"])
    def test_serialization(self, mode: str):
        # Create and quantize the model
        model = ToyLinearModel(16, 32).to(device=_DEVICE)

        mode_map = {
            "static": partial(
                Float8StaticActivationFloat8WeightConfig,
                scale=torch.tensor(1.0, dtype=torch.float32, device=_DEVICE),
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

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(
        _DEVICE == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    @common_utils.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @common_utils.parametrize("output_dtype", [torch.float32, torch.bfloat16])
    def test_choose_scale_float8_bounds(self, float8_dtype, output_dtype):
        block_size = ()
        device = _DEVICE
        input_tensor = torch.randn(8, 64, device=device, dtype=torch.float32)

        # testing upper bounds
        input_tensor[0][0] = 2000
        scale_ref = _choose_scale_float8(
            input_tensor, float8_dtype=float8_dtype, block_size=block_size
        )

        hp_value_ub = 1200
        scale_with_ub = _choose_scale_float8(
            input_tensor,
            float8_dtype=float8_dtype,
            block_size=block_size,
            hp_value_ub=hp_value_ub,
        )
        # since scale = abs_max / quant_max, larger abs_max means scale is larger
        self.assertTrue(scale_ref > scale_with_ub)

        # tesing lower bounds settings
        # making sure that abs is on the scale of 1e-20, so hp_value_lb can take effect
        input_tensor = torch.randn(8, 64, device=device, dtype=torch.float32) * 1e-20
        scale_ref = _choose_scale_float8(
            input_tensor, float8_dtype=float8_dtype, block_size=block_size
        )
        hp_value_lb = 1e-12
        scale_with_lb = _choose_scale_float8(
            input_tensor,
            float8_dtype=float8_dtype,
            block_size=block_size,
            hp_value_lb=hp_value_lb,
        )
        # since scale = abs_max / quant_max, larger abs_max means scale is larger
        self.assertTrue(scale_ref < scale_with_lb)

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(
        _DEVICE == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    @common_utils.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @common_utils.parametrize("output_dtype", [torch.float32, torch.bfloat16])
    @common_utils.parametrize("block_size", [(), (1, 32), (2, 16), (4, 8)])
    def test_dequantize_affine_float8(self, float8_dtype, output_dtype, block_size):
        """Test _dequantize_affine_float8 with various configurations"""

        device = _DEVICE
        input_tensor = torch.randn(8, 64, device=device, dtype=torch.float32)

        # Choose quantization parameters
        scale = _choose_scale_float8(
            input_tensor, float8_dtype=float8_dtype, block_size=block_size
        )

        # Quantize
        quantized = _quantize_affine_float8(input_tensor, scale, float8_dtype)

        # Dequantize
        dequantized = _dequantize_affine_float8(quantized, scale, output_dtype)

        # Verify output properties
        self.assertEqual(dequantized.dtype, output_dtype)
        self.assertEqual(dequantized.shape, input_tensor.shape)
        self.assertEqual(dequantized.device, input_tensor.device)

        # Verify quantization/dequantization roundtrip is reasonable
        error = torch.abs(input_tensor.to(output_dtype) - dequantized).mean()
        self.assertLess(error, 0.1, "Quantization error too high")

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(
        _DEVICE == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    def test_dequantize_affine_float8_scale_broadcasting(self):
        """Test that scale broadcasting works correctly for block-wise quantization"""
        device = _DEVICE
        # Create input tensor with known block structure
        input_tensor = torch.randn(4, 32, device=device, dtype=torch.float32)
        block_size = (2, 16)  # 2x2 blocks in first dim, 2x16 blocks in second dim

        # Choose quantization parameters
        scale = _choose_scale_float8(
            input_tensor, float8_dtype=torch.float8_e4m3fn, block_size=block_size
        )

        # Verify scale shape
        expected_scale_shape = (
            input_tensor.shape[0] // block_size[0],
            input_tensor.shape[1] // block_size[1],
        )
        self.assertEqual(scale.shape, expected_scale_shape)

        # Quantize
        quantized = _quantize_affine_float8(input_tensor, scale, torch.float8_e4m3fn)

        # Dequantize
        dequantized = _dequantize_affine_float8(quantized, scale, torch.float32)

        # Verify shapes match
        self.assertEqual(dequantized.shape, input_tensor.shape)

    def test_preprocess_scale_3d_reshape(self):
        """Test that preprocess_scale correctly handles 3D scale tensors"""
        device = "cpu"  # Use CPU for basic functionality test

        # Test 1: PerTensor scale (scalar) - should reshape to (1, 1)
        per_tensor_scale = torch.tensor(0.5, device=device)
        result = preprocess_scale(per_tensor_scale, (2, 4, 8))
        expected_shape = (1, 1)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.item(), 0.5)

        # Test 2: 1D scale tensor with one element - should reshape to (1, 1)
        one_element_scale = torch.tensor([0.3], device=device)
        result = preprocess_scale(one_element_scale, (2, 4, 8))
        expected_shape = (1, 1)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.item(), 0.3)

        # Test 3: 3D scale tensor for per-row quantization - should flatten first N-1 dims
        # This is the key test for the 3D reshape fix
        scale_3d = torch.randn(
            2, 4, device=device
        )  # Shape matches first 2 dims of (2, 4, 8)
        result = preprocess_scale(scale_3d, (2, 4, 8))
        expected_shape = (8, 1)  # Flattened (2*4, 1)
        self.assertEqual(result.shape, expected_shape)

        # Verify the values are preserved correctly
        expected_values = scale_3d.flatten().unsqueeze(-1)
        self.assertTrue(torch.allclose(result, expected_values))

        # Test 4: 2D scale tensor (already correct shape) - should just add last dimension
        scale_2d = torch.randn(8, device=device)
        result = preprocess_scale(scale_2d, (8, 16))
        expected_shape = (8, 1)
        self.assertEqual(result.shape, expected_shape)

        # Test 5: Edge case with higher dimensions (4D)
        scale_4d = torch.randn(
            2, 2, 2, device=device
        )  # Shape matches first 3 dims of (2, 2, 2, 8)
        result = preprocess_scale(scale_4d, (2, 2, 2, 8))
        expected_shape = (8, 1)  # Flattened (2*2*2, 1)
        self.assertEqual(result.shape, expected_shape)

    @common_utils.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @common_utils.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
    def test_quantize_dequantize_fp8_inductor(self, float8_dtype, hp_dtype):
        quantize_affine_float8 = torch.ops.torchao.quantize_affine_float8_non_decomposed
        dequantize_affine_float8 = (
            torch.ops.torchao.dequantize_affine_float8_non_decomposed
        )
        input = torch.randn(10, 10)
        with torch.no_grad():
            torch._dynamo.reset()
            expected_scale = torch.tensor(2.0)
            expected_quantized = quantize_affine_float8(
                input,
                expected_scale,
                float8_dtype=float8_dtype,
            )
            expected_dequantized = dequantize_affine_float8(
                expected_quantized,
                expected_scale,
                output_dtype=hp_dtype,
            )
            test_q, (code_q,) = torch._inductor.utils.run_and_get_code(
                torch.compile(quantize_affine_float8),
                input,
                expected_scale,
                float8_dtype=float8_dtype,
            )
            # After lowering the op is not in the output code but the base name is
            quant_op_base_name = f"{quantize_affine_float8}".split(".")[-1]
            torch.testing.FileCheck().check(quant_op_base_name).run(code_q)
            test_dq, (code_dq,) = torch._inductor.utils.run_and_get_code(
                torch.compile(dequantize_affine_float8),
                test_q,
                expected_scale,
                hp_dtype,
            )
            torch.testing.FileCheck().check(f"{dequantize_affine_float8}.default").run(
                code_dq
            )
            torch.testing.assert_close(expected_quantized, test_q)
            torch.testing.assert_close(expected_dequantized, test_dq)

    @torch.no_grad()
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_90(), "Requires GPU with compute capability >= 9.0"
    )
    @common_utils.parametrize("granularity", [PerTensor(), PerRow()])
    def test_expected_kernels_on_gpu(self, granularity):
        """
        Verify that float8 quantization + torch.compile results in the
        expected number of kernels in the GPU trace.
        """
        torch.compiler.reset()

        M, K, N = 128, 256, 512
        m = torch.nn.Sequential(
            torch.nn.Linear(K, N, device=_DEVICE, dtype=torch.bfloat16)
        )
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity,
            version=2,
            kernel_preference=KernelPreference.TORCH,
        )
        quantize_(
            m,
            config,
        )

        m = torch.compile(m)
        x = torch.randn(M, K, device=_DEVICE, dtype=torch.bfloat16)
        out, code = run_and_get_code(m, x)

        # triton kernel call looks like:
        #   triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_clone_div_expand_permute_transpose_unsqueeze_view_0.run(arg3_1, buf1, buf2, 128, 256, stream=stream0)
        # scaled_mm call looks like:
        #   extern_kernels._scaled_mm(buf1, reinterpret_tensor(arg0_1, (256, 512), (1, 256), 0), buf2, reinterpret_tensor(arg1_1, (1, 512), (1, 1), 0), arg2_1, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf3)
        if granularity == PerRow():
            # one triton kernel for quantizing the activation
            FileCheck().check("def call(").check_count(".run(", 1, exactly=True).run(
                code[0]
            )
            # one scaled_mm call
            FileCheck().check("def call(").check_count(
                "._scaled_mm(", 1, exactly=True
            ).run(code[0])
        else:
            assert granularity == PerTensor(), "unsupported"
            # three triton kernels for quantizing the activation:
            # kernel 1: x_max_tmp = max(x, ...)
            # kernel 2: x_max = max(x_max_tmp)
            # kernel 3: x_float8 = to_float8(x, x_max)
            FileCheck().check("def call(").check_count(".run(", 3, exactly=True).run(
                code[0]
            )
            # one scaled_mm call
            FileCheck().check("def call(").check_count(
                "._scaled_mm(", 1, exactly=True
            ).run(code[0])


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])
