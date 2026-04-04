# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Emulated NVFP4 scaled grouped GEMM functions.

These functions quantize inputs to NVFP4 (FP4 E2M1 with FP8 E4M3 block scales),
dequantize back to BF16, and run a standard grouped GEMM. This provides a
numerics reference for NVFP4 MoE training without requiring SM 10.0 hardware.

The pattern follows the MXFP8 emulated path established in PR #2626.
"""

from typing import Optional

import torch

from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    unpack_uint4,
)

NVFP4_BLOCK_SIZE = 16


def _nvfp4_dequantize(
    data_packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed NVFP4 data using block scales.

    Args:
        data_packed: Packed FP4 data (uint8, 2 values per byte).
            Shape: (..., K // 2)
        scales: Block scales in float8_e4m3fn.
            Shape: (..., K // block_size)
        block_size: Block size for quantization (must be 16 for NVFP4).
        output_dtype: Output dtype (default: bfloat16).

    Returns:
        Dequantized tensor of shape (..., K) in output_dtype.
    """
    # Unpack FP4: uint8 (K//2,) -> uint8 (K,) with one value per byte
    data_unpacked = unpack_uint4(data_packed.contiguous().view(torch.uint8))
    # Convert FP4 to float32
    data_f32 = f4_unpacked_to_f32(data_unpacked)

    # Reshape for per-block scaling: (..., K) -> (..., K//block_size, block_size)
    leading_shape = data_f32.shape[:-1]
    K = data_f32.shape[-1]
    data_f32 = data_f32.view(*leading_shape, K // block_size, block_size)
    scales_f32 = scales.to(torch.float32).unsqueeze(-1)

    # Scale and reshape back
    data_scaled = data_f32 * scales_f32
    return data_scaled.reshape(*leading_shape, K).to(output_dtype)


def emulated_nvfp4_scaled_grouped_mm_2d_3d(
    A_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_t_packed: torch.Tensor,
    B_t_scales: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Emulated NVFP4 scaled grouped GEMM: 2D activations @ 3D expert weights.

    Dequantizes NVFP4 inputs to BF16, then runs torch._grouped_mm.
    This provides a numerics reference without requiring SM 10.0 hardware.

    Args:
        A_packed: Packed FP4 activations, shape (M, K // 2).
        A_scales: Block scales for A, shape (M, K // block_size),
            dtype float8_e4m3fn.
        B_t_packed: Packed FP4 expert weights (transposed),
            shape (E, K, N // 2). Stored as (E, K, N) with N packed.
        B_t_scales: Block scales for B_t, shape (E, K // block_size, N),
            dtype float8_e4m3fn.
        offs: Group end offsets, shape (E,), dtype int32.
        out_dtype: Output dtype (default: bfloat16).
        block_size: Block size for quantization (default: 16).

    Returns:
        Output tensor of shape (M, N).
    """
    assert A_packed.ndim == 2, f"A must be 2D, got {A_packed.ndim}D"
    assert B_t_packed.ndim == 3, f"B must be 3D, got {B_t_packed.ndim}D"

    # Dequantize activations: (M, K//2) -> (M, K)
    A = _nvfp4_dequantize(A_packed, A_scales, block_size, output_dtype=out_dtype)

    # Dequantize expert weights
    # B_t is stored as (E, K, N) with N-dim packed, so we transpose to (E, N, K)
    # to align the packed dim with the last axis for unpacking, then transpose back.
    E, K_packed, N = B_t_packed.shape
    # B_t_packed shape: (E, K, N//2) — K is full, N is packed
    # Actually we need to be careful about which dim is packed.
    # Following MXFP8 convention: B_t shape is (E, K, N), scales are (E, K//block_size, N)
    # For NVFP4: the block scaling is along K dim, so we transpose to get K on last axis.

    # Transpose to (E, N, K) so K is the last dim for block-wise dequant
    B_mx = B_t_packed.transpose(-2, -1)  # (E, N, K//2) — K is packed on last dim
    B_scales = B_t_scales.transpose(-2, -1)  # (E, N, K//block_size)

    # Dequantize per expert: (E, N, K//2) -> (E, N, K)
    B = _nvfp4_dequantize(B_mx, B_scales, block_size, output_dtype=out_dtype)

    # Transpose back to (E, K, N)
    B_t = B.transpose(-2, -1)

    # Perform BF16 grouped GEMM
    return torch._grouped_mm(A, B_t, offs=offs, out_dtype=out_dtype)


def emulated_nvfp4_scaled_grouped_mm_2d_2d(
    A_packed: torch.Tensor,
    A_scales: torch.Tensor,
    B_packed: torch.Tensor,
    B_scales: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Emulated NVFP4 scaled grouped GEMM: 2D @ 2D (for wgrad computation).

    Dequantizes NVFP4 inputs to BF16, then runs torch._grouped_mm.

    Args:
        A_packed: Packed FP4 left operand, shape (M, K // 2).
        A_scales: Block scales for A, shape (M, K // block_size),
            dtype float8_e4m3fn.
        B_packed: Packed FP4 right operand, shape (N, K // 2).
        B_scales: Block scales for B, shape (N, K // block_size),
            dtype float8_e4m3fn.
        offs: Group end offsets, shape (E,), dtype int32.
        out_dtype: Output dtype (default: bfloat16).
        block_size: Block size for quantization (default: 16).

    Returns:
        Output tensor of shape (M, N).
    """
    assert A_packed.ndim == 2, f"A must be 2D, got {A_packed.ndim}D"
    assert B_packed.ndim == 2, f"B must be 2D, got {B_packed.ndim}D"

    # Dequantize both operands
    A = _nvfp4_dequantize(A_packed, A_scales, block_size, output_dtype=out_dtype)
    B = _nvfp4_dequantize(B_packed, B_scales, block_size, output_dtype=out_dtype)

    # B is (N, K), need (K, N) for matmul — _grouped_mm handles this
    return torch._grouped_mm(A, B.transpose(-2, -1), offs=offs, out_dtype=out_dtype)
