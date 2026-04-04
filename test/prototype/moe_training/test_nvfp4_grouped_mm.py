# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for emulated NVFP4 scaled grouped GEMM.
Verifies that quantize -> dequantize -> grouped_mm produces reasonable results.
"""

import pytest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.prototype.moe_training.kernels.nvfp4 import (
    emulated_nvfp4_scaled_grouped_mm_2d_2d,
    emulated_nvfp4_scaled_grouped_mm_2d_3d,
)
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

BLOCK_SIZE = 16


def _quantize_for_test(x: torch.Tensor):
    """Quantize a tensor using nvfp4_quantize and return (packed_data, scales)."""
    scales, packed_data = nvfp4_quantize(x, block_size=BLOCK_SIZE)
    return packed_data, scales


class TestEmulatedNVFP4GroupedMM(TestCase):
    def test_2d_3d_basic(self):
        """Test basic 2D @ 3D grouped GEMM with NVFP4 quantization."""
        E, M, N, K = 2, 32, 64, 64
        torch.manual_seed(42)

        A_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B_t_hp = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16)
        offs = torch.tensor([16, 32], dtype=torch.int32, device="cuda")

        # Quantize
        A_packed, A_scales = _quantize_for_test(A_hp)
        # B_t is (E, K, N) — quantize along K, so transpose to (E, N, K) first
        B_t_transposed = B_t_hp.transpose(-2, -1).contiguous()  # (E, N, K)
        # Quantize each expert
        B_packed_list, B_scales_list = [], []
        for i in range(E):
            packed, scales = _quantize_for_test(
                B_t_transposed[i].contiguous()
            )  # (N, K)
            B_packed_list.append(packed)
            B_scales_list.append(scales)
        B_packed = torch.stack(B_packed_list)  # (E, N, K//2)
        B_scales = torch.stack(B_scales_list)  # (E, N, K//16)

        # B_t convention: (E, K, N) with scales (E, K//16, N)
        B_t_packed = B_packed.transpose(-2, -1)  # (E, K//2, N)
        B_t_scales = B_scales.transpose(-2, -1)  # (E, K//16, N)

        # Run emulated NVFP4 grouped mm
        out = emulated_nvfp4_scaled_grouped_mm_2d_3d(
            A_packed, A_scales, B_t_packed, B_t_scales, offs=offs
        )

        # Run BF16 reference
        ref = torch._grouped_mm(A_hp, B_t_hp, offs=offs, out_dtype=torch.bfloat16)

        # Check shapes match
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.shape, (M, N))

        # Check numerics are reasonable (FP4 quantization loses precision)
        rel_error = (out - ref).norm() / ref.norm()
        self.assertLess(rel_error, 0.5, f"Relative error too high: {rel_error}")

    def test_2d_2d_basic(self):
        """Test basic 2D @ 2D grouped GEMM with NVFP4 quantization."""
        M, N, K = 32, 64, 64
        torch.manual_seed(42)

        A_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        offs = torch.tensor([16, 32], dtype=torch.int32, device="cuda")

        A_packed, A_scales = _quantize_for_test(A_hp)
        B_packed, B_scales = _quantize_for_test(B_hp)

        out = emulated_nvfp4_scaled_grouped_mm_2d_2d(
            A_packed, A_scales, B_packed, B_scales, offs=offs
        )

        ref = torch._grouped_mm(
            A_hp, B_hp.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16
        )

        # _grouped_mm 2D@2D may return 3D output; compare flattened
        self.assertEqual(out.numel(), ref.numel())

        rel_error = (out.flatten() - ref.flatten()).norm() / ref.norm()
        self.assertLess(rel_error, 0.5, f"Relative error too high: {rel_error}")

    def test_dequant_roundtrip(self):
        """Test that quantize -> dequantize preserves values approximately."""
        from torchao.prototype.moe_training.kernels.nvfp4.quant import (
            _nvfp4_dequantize,
        )

        torch.manual_seed(42)
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        scales, packed = nvfp4_quantize(x, block_size=BLOCK_SIZE)
        x_recon = _nvfp4_dequantize(packed, scales, output_dtype=torch.bfloat16)

        self.assertEqual(x_recon.shape, x.shape)
        rel_error = (x_recon - x).norm() / x.norm()
        self.assertLess(rel_error, 0.5, f"Roundtrip error too high: {rel_error}")


if __name__ == "__main__":
    run_tests()
