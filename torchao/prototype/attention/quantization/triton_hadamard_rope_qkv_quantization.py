# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused RoPE + Hadamard + FP8 quantization kernels for Q, K, V.

Input: [B, S, H, D], output: [B, H, S, D].
Supports GQA (different head counts for Q vs K/V).

Q and K receive RoPE + Hadamard; V receives Hadamard only (no RoPE).
Supports both NeoX/LLaMA half-split and FLUX/GPT-J interleaved RoPE.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from torchao.prototype.attention.quantization.triton_hadamard_utils import (
    _apply_hadamard,
    _compute_num_chunks,
    _get_log2_d,
)
from torchao.prototype.attention.quantization.triton_rope_qkv_quantization import (
    group_reduce_kernel,
    rope_single_phase2_kernel,
    single_reduce_kernel,
)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def hadamard_rope_single_phase1_kernel(
    # Input tensor [B, S, H, D]
    x_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Intermediate output tensor [B, H, S, D]
    x_out_ptr,
    # Temp buffer for Hadamard [B, H, num_chunks, D]
    temp_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, S, H, D] layout)
    stride_in_b,
    stride_in_s,
    stride_in_h,
    stride_in_d,
    # Output strides (for [B, H, S, D] layout)
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    # Temp buffer strides [B, H, num_chunks, D]
    stride_temp_b,
    stride_temp_h,
    stride_temp_c,
    stride_temp_d,
    # Dimensions
    S,
    H,
    D_HALF,
    chunk_size,
    num_chunks,
    # Compile-time constants
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
    USE_BFLOAT16: tl.constexpr,
    ROPE_INTERLEAVED: tl.constexpr,
):
    """
    Phase 1 for Q or K: Apply RoPE + Hadamard, store to intermediate,
    compute partial max.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.

    Supports NeoX half-split (pair j, j+D/2) and interleaved (pair 2i, 2i+1).
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )
    s_start = pid_chunk * chunk_size

    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    # RoPE partner index and sign
    if ROPE_INTERLEAVED:
        # FLUX/GPT-J interleaved: pair (2i, 2i+1)
        partner_d = d_idx ^ 1
        is_first = (d_idx % 2) == 0
    else:
        # NeoX/LLaMA half-split: pair (j, j+D/2)
        partner_d = d_idx ^ D_HALF
        is_first = d_idx < D_HALF
    # first element: out = x*cos - partner*sin  (sign = -1)
    # second element: out = x*cos + partner*sin (sign = +1)
    sign = tl.where(is_first, -1.0, 1.0)

    x_max = tl.zeros([D], dtype=tl.float32)

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        # Load x and its RoPE partner from input [B, S, H, D]
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d
        partner_in_offset = (
            in_base_b + s_idx * stride_in_s + in_base_h + partner_d * stride_in_d
        )

        x_val = tl.load(x_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)
        x_partner = tl.load(x_ptr + partner_in_offset, mask=s_mask, other=0.0).to(
            tl.float32
        )

        # Load cos/sin [S, D] — both elements of a pair share the same
        # rotation angle (values are duplicated in the cos/sin tensors).
        cos_offset = s_idx * D + d_idx
        cos_val = tl.load(cos_ptr + cos_offset, mask=s_mask, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptr + cos_offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply RoPE rotation
        x_rope = tl.math.fma(x_val, cos_val, sign * x_partner * sin_val)

        # Apply Hadamard transform with 1/sqrt(D) normalization
        x_rope = _apply_hadamard(x_rope, temp_ptr, temp_base, d_idx, D, LOG2_D)

        # Store to intermediate buffer [B, H, S, D]
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d
        if USE_BFLOAT16:
            tl.store(x_out_ptr + out_offset, x_rope.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(x_out_ptr + out_offset, x_rope.to(tl.float16), mask=s_mask)

        x_max = tl.maximum(x_max, tl.abs(x_rope))

    x_max_scalar = tl.max(x_max)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max_scalar)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def hadamard_v_phase1_kernel(
    # Input tensor [B, S, H, D]
    v_ptr,
    # Intermediate output tensor [B, H, S, D] - Hadamard'd and transposed
    v_out_ptr,
    # Temp buffer for Hadamard [B, H, num_chunks, D]
    temp_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, S, H, D] layout)
    stride_in_b,
    stride_in_s,
    stride_in_h,
    stride_in_d,
    # Output strides (for [B, H, S, D] layout)
    stride_out_b,
    stride_out_h,
    stride_out_s,
    stride_out_d,
    # Temp buffer strides [B, H, num_chunks, D]
    stride_temp_b,
    stride_temp_h,
    stride_temp_c,
    stride_temp_d,
    # Dimensions
    S,
    H,
    chunk_size,
    num_chunks,
    # Compile-time constants
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
    USE_BFLOAT16: tl.constexpr,
):
    """
    Phase 1 for V: Apply Hadamard (no RoPE), transpose [B,S,H,D] -> [B,H,S,D],
    compute partial max.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )
    s_start = pid_chunk * chunk_size

    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    v_max = tl.zeros([D], dtype=tl.float32)

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        # Load V from input [B, S, H, D]
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d
        v_val = tl.load(v_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply Hadamard transform with 1/sqrt(D) normalization
        v_val = _apply_hadamard(v_val, temp_ptr, temp_base, d_idx, D, LOG2_D)

        # Store to intermediate buffer [B, H, S, D] (transposed)
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d
        if USE_BFLOAT16:
            tl.store(v_out_ptr + out_offset, v_val.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(v_out_ptr + out_offset, v_val.to(tl.float16), mask=s_mask)

        v_max = tl.maximum(v_max, tl.abs(v_val))

    v_max_scalar = tl.max(v_max)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, v_max_scalar)


def triton_fp8_hadamard_rope_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_chunks: Optional[int] = None,
    rope_interleaved: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Fused RoPE + Hadamard + FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then Hadamard transform to Q, K, and V,
    then quantizes all tensors to FP8 with per-head scaling. Also performs
    layout transformation from [B, S, H, D] to [B, H, S, D].
    Each of Q, K, V is processed with independent kernel launches.

    Supports GQA where Q has more heads than K/V (H_q = groups * H_kv).
    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    The caller must apply inverse Hadamard to the attention output:
        output = inverse_hadamard_transform(attention_output)

    Args:
        q: Query tensor of shape [B, S, H_q, D] in bf16/fp16
        k: Key tensor of shape [B, S, H_kv, D] in bf16/fp16
        v: Value tensor of shape [B, S, H_kv, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.
        rope_interleaved: If True, use FLUX/GPT-J interleaved RoPE pairing
                          (2i, 2i+1). If False, use NeoX/LLaMA half-split
                          pairing (j, j+D/2).

    Returns:
        q_fp8: Quantized query with RoPE+Hadamard, shape [B, H_q, S, D] in fp8
        k_fp8: Quantized key with RoPE+Hadamard, shape [B, H_kv, S, D] in fp8
        v_fp8: Quantized value with Hadamard, shape [B, H_kv, S, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32

    Note:
        D must be a power of 2 and <= 256 for the Hadamard transform.
        Q, K, V must have the same sequence length (RoPE requires matching positions).
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {v.dim()}D"
    assert k.shape == v.shape, (
        f"K and V must have the same shape, got {k.shape} vs {v.shape}"
    )
    assert q.shape[0] == k.shape[0], (
        f"Batch size mismatch: {q.shape[0]} vs {k.shape[0]}"
    )
    assert q.shape[1] == k.shape[1], (
        f"Sequence length mismatch: {q.shape[1]} vs {k.shape[1]}"
    )
    assert q.shape[3] == k.shape[3], f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}"
    assert q.shape[2] % k.shape[2] == 0, (
        f"Q heads ({q.shape[2]}) must be a multiple of K heads ({k.shape[2]})"
    )
    assert cos.dim() == 2, f"Expected 2D cos tensor [S, D], got {cos.dim()}D"
    assert sin.dim() == 2, f"Expected 2D sin tensor [S, D], got {sin.dim()}D"

    B, S, H_q, D = q.shape
    H_kv = k.shape[2]
    groups = H_q // H_kv

    assert D % 2 == 0, f"Head dimension D must be even for RoPE, got D={D}"
    assert cos.shape == (S, D), f"Expected cos shape [{S}, {D}], got {cos.shape}"
    assert sin.shape == (S, D), f"Expected sin shape [{S}, {D}], got {sin.shape}"

    LOG2_D = _get_log2_d(D)
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"

    D_HALF = D // 2

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    use_bfloat16 = q.dtype == torch.bfloat16

    # Compute number of chunks
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q.device, B, H_q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    # Allocate output tensors in [B, H, S, D] layout for SDPA
    q_fp8 = torch.empty(B, H_q, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(B, H_kv, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    v_fp8 = torch.empty(B, H_kv, S, D, dtype=torch.float8_e4m3fn, device=q.device)

    # Intermediate buffers [B, H, S, D] for transformed values
    q_intermediate = torch.empty(B, H_q, S, D, dtype=q.dtype, device=q.device)
    k_intermediate = torch.empty(B, H_kv, S, D, dtype=k.dtype, device=q.device)
    v_intermediate = torch.empty(B, H_kv, S, D, dtype=v.dtype, device=q.device)

    # Temp buffers for Hadamard butterfly
    q_temp = torch.empty(B, H_q, num_chunks, D, dtype=torch.float32, device=q.device)
    kv_temp = torch.empty(B, H_kv, num_chunks, D, dtype=torch.float32, device=q.device)

    # Partial max buffers
    q_partial_max = torch.empty(
        B * H_q * num_chunks, dtype=torch.float32, device=q.device
    )
    k_partial_max = torch.empty(
        B * H_kv * num_chunks, dtype=torch.float32, device=q.device
    )
    v_partial_max = torch.empty(
        B * H_kv * num_chunks, dtype=torch.float32, device=q.device
    )

    # Scale/descale tensors
    q_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)

    q_grid = (B, H_q, num_chunks)
    kv_grid = (B, H_kv, num_chunks)

    # ---- Phase 1: RoPE + Hadamard + max for Q ----
    hadamard_rope_single_phase1_kernel[q_grid](
        q,
        cos,
        sin,
        q_intermediate,
        q_temp,
        q_partial_max,
        # Input strides [B, S, H_q, D]
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # Output strides [B, H_q, S, D]
        q_intermediate.stride(0),
        q_intermediate.stride(1),
        q_intermediate.stride(2),
        q_intermediate.stride(3),
        # Temp strides
        q_temp.stride(0),
        q_temp.stride(1),
        q_temp.stride(2),
        q_temp.stride(3),
        S,
        H_q,
        D_HALF,
        chunk_size,
        num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
        ROPE_INTERLEAVED=rope_interleaved,
    )

    # ---- Phase 1: RoPE + Hadamard + max for K ----
    hadamard_rope_single_phase1_kernel[kv_grid](
        k,
        cos,
        sin,
        k_intermediate,
        kv_temp,
        k_partial_max,
        # Input strides [B, S, H_kv, D]
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # Output strides [B, H_kv, S, D]
        k_intermediate.stride(0),
        k_intermediate.stride(1),
        k_intermediate.stride(2),
        k_intermediate.stride(3),
        # Temp strides
        kv_temp.stride(0),
        kv_temp.stride(1),
        kv_temp.stride(2),
        kv_temp.stride(3),
        S,
        H_kv,
        D_HALF,
        chunk_size,
        num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
        ROPE_INTERLEAVED=rope_interleaved,
    )

    # ---- Phase 1: Hadamard + max for V (no RoPE, with transpose) ----
    hadamard_v_phase1_kernel[kv_grid](
        v,
        v_intermediate,
        kv_temp,
        v_partial_max,
        # Input strides [B, S, H_kv, D]
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # Output strides [B, H_kv, S, D]
        v_intermediate.stride(0),
        v_intermediate.stride(1),
        v_intermediate.stride(2),
        v_intermediate.stride(3),
        # Temp strides
        kv_temp.stride(0),
        kv_temp.stride(1),
        kv_temp.stride(2),
        kv_temp.stride(3),
        S,
        H_kv,
        chunk_size,
        num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
    )

    # ---- Reduce ----
    # Q: group reduce across `groups` Q heads per KV head
    group_reduce_kernel[(B, H_kv)](
        q_partial_max, q_scale, q_descale, H_q, H_kv, groups, num_chunks
    )
    # K, V: per-head reduce
    single_reduce_kernel[(B, H_kv)](k_partial_max, k_scale, k_descale, H_kv, num_chunks)
    single_reduce_kernel[(B, H_kv)](v_partial_max, v_scale, v_descale, H_kv, num_chunks)

    # ---- Phase 2: Quantize Q from intermediate ----
    rope_single_phase2_kernel[q_grid](
        q_intermediate,
        q_fp8,
        q_scale,
        # Strides [B, H_q, S, D]
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        S,
        D,
        H_q,
        chunk_size,
        H_kv,
        groups,
    )

    # ---- Phase 2: Quantize K from intermediate ----
    rope_single_phase2_kernel[kv_grid](
        k_intermediate,
        k_fp8,
        k_scale,
        # Strides [B, H_kv, S, D]
        k_fp8.stride(0),
        k_fp8.stride(1),
        k_fp8.stride(2),
        k_fp8.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        H_kv,
        1,
    )

    # ---- Phase 2: Quantize V from intermediate ----
    # V intermediate is already [B, H, S, D] (transposed in phase1)
    rope_single_phase2_kernel[kv_grid](
        v_intermediate,
        v_fp8,
        v_scale,
        # Strides [B, H_kv, S, D]
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        H_kv,
        1,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
