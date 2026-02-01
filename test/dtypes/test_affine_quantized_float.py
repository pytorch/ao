# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import random
import unittest

import pytest
import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal import common_utils

from torchao.dtypes.floatx.float8_layout import preprocess_scale
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    quantize_,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _dequantize_affine_float8,
    _quantize_affine_float8,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import (
    get_current_accelerator_device,
    is_sm_at_least_89,
    is_sm_at_least_90,
    is_sm_at_least_100,
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
    @unittest.skipIf(
        torch.cuda.is_available() and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity specification"):
            Float8DynamicActivationFloat8WeightConfig(granularity="invalid")

    @unittest.skipIf(
        torch.cuda.is_available() and not is_sm_at_least_89(),
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
        torch.cuda.is_available() and not is_sm_at_least_89(),
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
            model = ToyLinearModel(64, 64).eval().to(torch.float32).to("cuda")
            quantize_(
                model,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            )

    @unittest.skipIf(not torch.accelerator.is_available(), "Need GPU available")
    @unittest.skipIf(
        torch.cuda.is_available() and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    @common_utils.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @common_utils.parametrize("output_dtype", [torch.float32, torch.bfloat16])
    def test_choose_scale_float8_bounds(self, float8_dtype, output_dtype):
        device = get_current_accelerator_device()
        input_tensor = torch.randn(8, 64, device=device, dtype=torch.float32)
        block_size = input_tensor.shape

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
        torch.cuda.is_available() == "cuda" and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    @common_utils.parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @common_utils.parametrize("output_dtype", [torch.float32, torch.bfloat16])
    @common_utils.parametrize("block_size", [(8, 64), (1, 32), (2, 16), (4, 8)])
    def test_dequantize_affine_float8(self, float8_dtype, output_dtype, block_size):
        """Test _dequantize_affine_float8 with various configurations"""
        device = get_current_accelerator_device()
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
        torch.cuda.is_available() and not is_sm_at_least_89(),
        "Requires GPU with compute capability >= 8.9",
    )
    def test_dequantize_affine_float8_scale_broadcasting(self):
        """Test that scale broadcasting works correctly for block-wise quantization"""
        device = get_current_accelerator_device()
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
        expected number of kernels in the GPU trace for both TORCH and AUTO
        kernel preferences.
        """
        torch.compiler.reset()

        M, K, N = 128, 256, 512
        m = torch.nn.Sequential(
            torch.nn.Linear(K, N, device="cuda", dtype=torch.bfloat16)
        )

        for kernel_pref in (KernelPreference.TORCH, KernelPreference.AUTO):
            config = Float8DynamicActivationFloat8WeightConfig(
                granularity=granularity,
                version=2,
                kernel_preference=kernel_pref,
            )
            quantize_(
                m,
                config,
            )

            m = torch.compile(m)
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            out, code = run_and_get_code(m, x)

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
                # For TORCH, expect scaled_mm. For AUTO, behavior differs by GPU:
                if kernel_pref == KernelPreference.TORCH:
                    FileCheck().check("def call(").check_count(
                        "._scaled_mm(", 1, exactly=True
                    ).run(code[0])
                else:  # AUTO
                    # On B200+ hardware with per-tensor scales, AUTO should use scaled_mm
                    # On non-B200 hardware, AUTO may select MSLK
                    if is_sm_at_least_100():
                        # B200/GB200: expect scaled_mm for per-tensor scales
                        FileCheck().check("def call(").check_count(
                            "._scaled_mm(", 1, exactly=True
                        ).run(code[0])
                    else:
                        # Non-B200: MSLK should be selected if available
                        # This check is flexible since MSLK may not be installed
                        has_scaled_mm = "._scaled_mm(" in code[0]
                        has_mslk = "mslk" in code[0]
                        assert has_scaled_mm or has_mslk, "Expected either scaled_mm or mslk kernel"



common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])
