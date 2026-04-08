# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.prototype.moe_training.utils import conditional_nostrict_trace
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    unpack_uint4,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)

NVFP4_BLOCK_SIZE = 16


@conditional_nostrict_trace
def _to_nvfp4_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    wgrad_with_hp: bool = False,
) -> torch.Tensor:
    """NVFP4 emulated scaled grouped matrix multiplication with autograd support.

    Quantizes inputs to NVFP4, performs emulated grouped GEMM, and supports
    backward pass for training.

    Args:
        A: Input activations, shape (M, K)
        B_t: Expert weights, shape (E, K, N)
        offs: Group end offsets, shape (E,)
        out_dtype: Output dtype (bfloat16 or float32)
        wgrad_with_hp: If True, compute weight gradients in high precision (BF16).
                       If False, use NVFP4 quantized weight gradients.
    Returns:
        Output tensor, shape (M, N)
    """
    return _NVFP4GroupedMM.apply(A, B_t, offs, out_dtype, wgrad_with_hp)


class _NVFP4GroupedMM(torch.autograd.Function):
    """
    Differentiable implementation of grouped GEMM with dynamic NVFP4 quantization.

    This autograd function performs grouped matrix multiplication with NVFP4 quantization
    for efficient MoE training. It uses two-level scaling (per-block + per-tensor) to
    handle the limited dynamic range of FP4 E2M1 format in backward passes.
    """

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
        assert input_act.ndim == 2, f"input_act must be 2D, got {input_act.ndim}D"
        assert weight_t.ndim == 3, f"weight_t must be 3D, got {weight_t.ndim}D"
        assert group_end_offsets is not None, "group_end_offsets (offs) is required"
        assert out_dtype in (torch.bfloat16, torch.float32), (
            f"out_dtype must be bfloat16 or float32, got {out_dtype}"
        )

        # Quantize input_act (M, K) along last dim
        x_scales, x_packed = nvfp4_quantize(input_act, block_size=NVFP4_BLOCK_SIZE)
        # x_packed: (M, K//2), x_scales: (M, K//block_size)

        # Quantize weight: transpose to (E, N, K), then quantize per-expert
        w = weight_t.transpose(-2, -1).contiguous()  # (E, K, N) -> (E, N, K)
        w_packed, w_scales = _nvfp4_quantize_3d(w)
        # w_packed: (E, N, K//2), w_scales: (E, N, K//block_size)

        # Emulated grouped GEMM
        output = _emulated_to_nvfp4_then_scaled_grouped_mm_2d_3d(
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
        # NVFP4's scale dtype (float8_e4m3fn) has limited dynamic range
        # (4-bit exponent, min subnormal ~0.002). per_tensor_scale extends
        # this range so small gradient values don't underflow to zero.
        # grad_output(M,N) @ weight(E,N,K) = (M,K)
        go_pts = per_tensor_amax_to_scale(grad_output.abs().max())
        go_scales, go_packed = nvfp4_quantize(
            grad_output, block_size=NVFP4_BLOCK_SIZE, per_tensor_scale=go_pts
        )

        # weight_t is (E,K,N) — quantize directly, last dim N is contraction dim
        w_packed, w_scales, w_pts = _nvfp4_quantize_3d(
            weight_t, with_per_tensor_scale=True
        )

        grad_input = _emulated_to_nvfp4_then_scaled_grouped_mm_2d_3d(
            go_packed,
            go_scales,
            w_packed,
            w_scales,
            offs=group_end_offsets,
            out_dtype=ctx.out_dtype,
            A_per_tensor_scale=go_pts,
            B_per_tensor_scale=w_pts,
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
            # Quantized wgrad via 2d_2d with per_tensor_scale
            go_t = grad_output.t().contiguous()  # (N, M)
            go_t_pts = per_tensor_amax_to_scale(go_t.abs().max())
            go_t_scales, go_t_packed = nvfp4_quantize(
                go_t, block_size=NVFP4_BLOCK_SIZE, per_tensor_scale=go_t_pts
            )

            x_t = input_act.t().contiguous()  # (K, M)
            x_t_pts = per_tensor_amax_to_scale(x_t.abs().max())
            x_t_scales, x_t_packed = nvfp4_quantize(
                x_t, block_size=NVFP4_BLOCK_SIZE, per_tensor_scale=x_t_pts
            )

            # 2d_2d: A=(N,M), B=(K,M) -> B^T=(M,K) internal -> (N,M)@(M,K) = (E,N,K)
            grad_weight = _emulated_to_nvfp4_then_scaled_grouped_mm_2d_2d(
                go_t_packed,
                go_t_scales,
                x_t_packed,
                x_t_scales,
                offs=group_end_offsets,
                out_dtype=ctx.out_dtype,
                A_per_tensor_scale=go_t_pts,
                B_per_tensor_scale=x_t_pts,
            )
            grad_weight_t = grad_weight.transpose(-2, -1)  # (E,N,K) -> (E,K,N)

        return grad_input, grad_weight_t, None, None, None


def _nvfp4_quantize_3d(
    w: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
    with_per_tensor_scale: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Quantize 3D tensor per-expert. Returns (packed_data, scales) or
    (packed_data, scales, per_tensor_scales) when with_per_tensor_scale=True.

    When with_per_tensor_scale=True, computes a single per_tensor_scale from
    the global amax across ALL experts (not per-expert), matching the semantics
    of per_tensor_scale as a tensor-level normalizer.
    """
    pts = None
    if with_per_tensor_scale:
        pts = per_tensor_amax_to_scale(w.abs().max())

    packed_list, scales_list = [], []
    for i in range(w.shape[0]):
        scales, packed = nvfp4_quantize(
            w[i].contiguous(), block_size=block_size, per_tensor_scale=pts
        )
        packed_list.append(packed)
        scales_list.append(scales)

    if with_per_tensor_scale:
        return torch.stack(packed_list), torch.stack(scales_list), pts
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


def _emulated_to_nvfp4_then_scaled_grouped_mm_2d_3d(
    A_data: torch.Tensor,
    A_scale: torch.Tensor,
    B_data: torch.Tensor,
    B_scale: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = NVFP4_BLOCK_SIZE,
    A_per_tensor_scale: Optional[torch.Tensor] = None,
    B_per_tensor_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Emulated NVFP4 scaled grouped GEMM: 2D activations @ 3D expert weights.

    Dequantizes NVFP4 inputs to BF16, then runs torch._grouped_mm.
    Supports optional two-level scaling via per_tensor_scale parameters.
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
    A = _nvfp4_dequantize(
        A_data,
        A_scale,
        block_size,
        per_tensor_scale=A_per_tensor_scale,
        output_dtype=out_dtype,
    )

    # Dequantize expert weights
    B = _nvfp4_dequantize(
        B_data,
        B_scale,
        block_size,
        per_tensor_scale=B_per_tensor_scale,
        output_dtype=out_dtype,
    )

    # Transpose to (E, K, N) for grouped GEMM: (M, K) @ (E, K, N) = (M, N)
    B_t = B.transpose(-2, -1)

    # Perform bf16 grouped GEMM
    out = torch._grouped_mm(A, B_t, offs=offs, out_dtype=out_dtype)
    return out


def _emulated_to_nvfp4_then_scaled_grouped_mm_2d_2d(
    A_data: torch.Tensor,  # (M, K//2) packed
    A_scale: torch.Tensor,  # (M, K//block_size)
    B_data: torch.Tensor,  # (N, K//2) packed
    B_scale: torch.Tensor,  # (N, K//block_size)
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = NVFP4_BLOCK_SIZE,
    A_per_tensor_scale: Optional[torch.Tensor] = None,
    B_per_tensor_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Emulated NVFP4 scaled grouped GEMM: 2D @ 2D (for wgrad computation).

    Dequantizes NVFP4 inputs to BF16, then runs torch._grouped_mm.

    Following the MXFP8 convention, B_data is provided as (N, K) and
    transposed internally to (K, N) for the matmul.
    """
    assert A_data.ndim == 2, "A must be 2D"
    assert B_data.ndim == 2, "B must be 2D"

    # Dequantize A: (M, K//2) packed -> (M, K)
    A_dequant = _nvfp4_dequantize(
        A_data,
        A_scale,
        block_size,
        per_tensor_scale=A_per_tensor_scale,
        output_dtype=out_dtype,
    )

    # Dequantize B: (N, K//2) packed -> (N, K)
    B_dequant = _nvfp4_dequantize(
        B_data,
        B_scale,
        block_size,
        per_tensor_scale=B_per_tensor_scale,
        output_dtype=out_dtype,
    )

    # Transpose B from (N, K) to (K, N) for matmul: A (M, K) @ B^T (K, N) = (M, N)
    out = torch._grouped_mm(
        A_dequant, B_dequant.transpose(-2, -1), offs=offs, out_dtype=out_dtype
    )
    return out
