# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    unpack_uint4,
)
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

NVFP4_BLOCK_SIZE = 16


class _NVFP4GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_act,
        weight_t,
        group_end_offsets,
        out_dtype=torch.bfloat16,
        wgrad_with_hp=True,
    ):
        """
        Args:
            input_act: (M, K) high-precision activations
            weight_t: (E, K, N) high-precision expert weights
            group_end_offsets: (E,) int32 cumulative token indices
            out_dtype: output dtype (bf16 or fp32)
            wgrad_with_hp: whether to compute wgrad in HP
        Returns:
            output: (M, N)
        """
        assert input_act.ndim == 2
        assert weight_t.ndim == 3
        assert group_end_offsets is not None
        assert out_dtype in (torch.bfloat16, torch.float32)

        # Quantize input_act (M, K) along last dim
        x_scales, x_packed = nvfp4_quantize(input_act, block_size=NVFP4_BLOCK_SIZE)
        # x_packed: (M, K//2), x_scales: (M, K//block_size)

        # Quantize weight: transpose to (E, N, K), then quantize per-expert
        w = weight_t.transpose(-2, -1).contiguous()  # (E, K, N) -> (E, N, K)
        w_packed, w_scales = _nvfp4_quantize_3d(w)
        # w_packed: (E, N, K//2), w_scales: (E, N, K//block_size)

        # Emulated grouped GEMM
        output = _emulated_nvfp4_scaled_grouped_mm_2d_3d(
            x_packed,
            x_scales,
            w_packed,
            w_scales,
            offs=group_end_offsets,
            out_dtype=out_dtype,
        )

        # Save for backward
        ctx.save_for_backward(input_act, weight_t, group_end_offsets)
        ctx.out_dtype = out_dtype
        ctx.wgrad_with_hp = wgrad_with_hp

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_act, weight_t, group_end_offsets = ctx.saved_tensors

        # --- dgrad: grad_input = grad_output @ weight ---
        # Use HP grouped GEMM for dgrad because NVFP4's scale dtype
        # (float8_e4m3fn, 4-bit exponent) has limited dynamic range.
        # Small gradient values (e.g., from MSE loss with large numel)
        # underflow to zero during FP4 block scale computation.
        # MXFP8 doesn't have this issue because its scale dtype is
        # float8_e8m0fnu (8-bit exponent) with much wider dynamic range.
        # grad_output(M,N) @ weight(E,N,K) = (M,K)
        # weight_t is (E,K,N), transpose to (E,N,K) so contraction dim N matches
        grad_input = torch._grouped_mm(
            grad_output,
            weight_t.transpose(-2, -1),
            offs=group_end_offsets,
            out_dtype=ctx.out_dtype,
        )

        # --- wgrad ---
        if ctx.wgrad_with_hp:
            # HP grouped GEMM: grad_output^T (N,M) @ input_act (M,K) = (E,N,K)
            grad_weight = torch._grouped_mm(
                grad_output.transpose(-2, -1),
                input_act,
                offs=group_end_offsets,
                out_dtype=ctx.out_dtype,
            )
            grad_weight_t = grad_weight.transpose(-2, -1)  # (E,N,K) -> (E,K,N)
        else:
            # Quantized wgrad via 2d_2d
            go_t = grad_output.t().contiguous()  # (N, M)
            go_t_scales, go_t_packed = nvfp4_quantize(go_t, block_size=NVFP4_BLOCK_SIZE)

            x_t = input_act.t().contiguous()  # (K, M)
            x_t_scales, x_t_packed = nvfp4_quantize(x_t, block_size=NVFP4_BLOCK_SIZE)

            # 2d_2d: A=(N,M), B=(K,M) -> B^T=(M,K) internal -> (N,M)@(M,K) = (E,N,K)
            grad_weight = _emulated_nvfp4_scaled_grouped_mm_2d_2d(
                go_t_packed,
                go_t_scales,
                x_t_packed,
                x_t_scales,
                offs=group_end_offsets,
                out_dtype=ctx.out_dtype,
            )
            grad_weight_t = grad_weight.transpose(-2, -1)  # (E,N,K) -> (E,K,N)

        return grad_input, grad_weight_t, None, None, None


def _nvfp4_quantize_3d(
    w: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 3D tensor per-expert. Returns (packed_data, scales)."""
    packed_list, scales_list = [], []
    for i in range(w.shape[0]):
        scales, packed = nvfp4_quantize(w[i].contiguous(), block_size=block_size)
        packed_list.append(packed)
        scales_list.append(scales)
    return torch.stack(packed_list), torch.stack(scales_list)


def _nvfp4_dequantize(
    data_packed: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    per_tensor_scale: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed NVFP4 data using block scales.

    Unlike MXFP8 where data.to(bfloat16) suffices, NVFP4 requires unpacking
    (2 FP4 values per uint8 byte) and explicit FP4-to-FP32 conversion before
    applying block scales.

    When per_tensor_scale is provided, the effective scale is
    per_tensor_scale * block_scale (two-level scaling as described in the
    NVIDIA NVFP4 paper). In this emulated path we simply multiply through
    after block-level dequantization.
    """
    # Unpack FP4: uint8 (K//2) -> uint8 (K) with one value per byte
    data_unpacked = unpack_uint4(data_packed.contiguous().view(torch.uint8))

    # Convert FP4 E2M1 to float32
    data_f32 = f4_unpacked_to_f32(data_unpacked)

    # Reshape for per-block scaling
    # data_f32 shape: (M, K) -> (M, K//block_size, block_size)
    # scale shape: (M, K//block_size) -> (M, K//block_size, 1)
    leading_shape = data_f32.shape[:-1]
    K = data_f32.shape[-1]
    data_f32 = data_f32.view(*leading_shape, K // block_size, block_size)
    scale_f32 = scale.to(torch.float32).unsqueeze(-1)

    # Rescale with block scales
    data_scaled = data_f32 * scale_f32

    # Apply per-tensor scale if using two-level scaling
    if per_tensor_scale is not None:
        data_scaled = data_scaled * per_tensor_scale.to(torch.float32)

    # Reshape back: (M, K//block_size, block_size) -> (M, K)
    return data_scaled.reshape(*leading_shape, K).to(output_dtype)


def _emulated_nvfp4_scaled_grouped_mm_2d_3d(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Emulated NVFP4 scaled grouped GEMM: 2D activations @ 3D expert weights.

    Dequantizes NVFP4 inputs to BF16, then runs torch._grouped_mm.

    TODO: plumb per_tensor_scale through when adding the autograd function,
    to support NVFP4 two-level scaling (block scale * per-tensor scale).
    """
    assert A_data.ndim == 2, f"A must be 2D, got {A_data.ndim}"
    assert B_data.ndim == 3, f"B must be 3D, got {B_data.ndim}"
    assert A_scale.shape[0] == A_data.shape[0], (
        f"A_scale must have same M dim as A_data, got A={A_data.shape} and A_scale={A_scale.shape}"
    )
    # FP4 is packed: A_data has K//2 cols, so scale dim1 == A_data dim1 * 2 // block_size
    assert A_scale.shape[1] == A_data.shape[1] * 2 // block_size, (
        f"A_scale dim1 should be K//block_size (where K = A_data.shape[1]*2 for packed FP4), "
        f"got A={A_data.shape} and A_scale={A_scale.shape}"
    )
    assert B_scale.shape[0] == B_data.shape[0], (
        f"B_scale must have same E dim as B_data, got B={B_data.shape} and B_scale={B_scale.shape}"
    )
    assert B_scale.shape[1] == B_data.shape[1], (
        f"B_scale must have same N dim as B_data, got B={B_data.shape} and B_scale={B_scale.shape}"
    )
    assert B_scale.shape[2] == B_data.shape[2] * 2 // block_size, (
        f"B_scale dim2 should be K//block_size (where K = B_data.shape[2]*2 for packed FP4), "
        f"got B={B_data.shape} and B_scale={B_scale.shape}"
    )

    # Dequantize activations
    # A_data shape: (M, K//2) packed
    # A_scale shape: (M, K//block_size)
    A = _nvfp4_dequantize(A_data, A_scale, block_size, output_dtype=out_dtype)
    # A shape: (M, K)

    # Dequantize expert weights
    # B_data shape: (E, N, K//2) packed
    # B_scale shape: (E, N, K//block_size)
    B = _nvfp4_dequantize(B_data, B_scale, block_size, output_dtype=out_dtype)
    # B shape: (E, N, K)

    # Transpose to (E, K, N) for grouped GEMM: (M, K) @ (E, K, N) = (M, N)
    B_t = B.transpose(-2, -1)

    # Perform bf16 grouped GEMM
    out = torch._grouped_mm(A, B_t, offs=offs, out_dtype=out_dtype)
    return out


def _emulated_nvfp4_scaled_grouped_mm_2d_2d(
    A_data: torch.Tensor,  # (M, K//2) packed
    A_scale: torch.Tensor,  # (M, K//block_size)
    B_data: torch.Tensor,  # (N, K//2) packed
    B_scale: torch.Tensor,  # (N, K//block_size)
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Emulated NVFP4 scaled grouped GEMM: 2D @ 2D (for wgrad computation).

    Dequantizes NVFP4 inputs to BF16, then runs torch._grouped_mm.

    Following the MXFP8 convention, B_data is provided as (N, K) and
    transposed internally to (K, N) for the matmul.

    TODO: plumb per_tensor_scale through when adding the autograd function.
    """
    assert A_data.ndim == 2, "A must be 2D"
    assert B_data.ndim == 2, "B must be 2D"

    # Dequantize A: (M, K//2) packed -> (M, K)
    A_dequant = _nvfp4_dequantize(A_data, A_scale, block_size, output_dtype=out_dtype)

    # Dequantize B: (N, K//2) packed -> (N, K)
    B_dequant = _nvfp4_dequantize(B_data, B_scale, block_size, output_dtype=out_dtype)

    # Transpose B from (N, K) to (K, N) for matmul: A (M, K) @ B^T (K, N) = (M, N)
    out = torch._grouped_mm(
        A_dequant, B_dequant.transpose(-2, -1), offs=offs, out_dtype=out_dtype
    )
    return out
