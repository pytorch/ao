# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused FP8 RoPE + Quantization for SDPA.

This module provides Triton kernels that fuse:
- RoPE (Rotary Position Embeddings) for Q and K
- FP8 quantization for Q, K, V

The fused kernel reduces memory traffic by:
1. Using linearized iteration for coalesced memory access
2. Reading Q, K, V in a single pass during max computation
3. Applying RoPE + quantization in a second pass

Input format: [B, S, H, D] (FLUX-style)
Output format: [B, H, S, D] (SDPA-style)
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# =============================================================================
# Helper functions
# =============================================================================


def _compute_num_chunks(tensor: torch.Tensor, S: int) -> int:
    """Compute optimal number of chunks based on GPU properties."""
    props = torch.cuda.get_device_properties(tensor.device)
    num_sms = props.multi_processor_count
    B, _, H, _ = tensor.shape  # [B, S, H, D]
    base_parallelism = B * H
    # Target 2-4x SMs for good occupancy/latency hiding
    target_blocks = num_sms * 4
    num_chunks = max(1, target_blocks // base_parallelism)
    # Ensure each chunk has at least 32 S positions for efficiency
    num_chunks = min(num_chunks, S // 32) if S >= 32 else 1
    # Cap at reasonable maximum
    num_chunks = min(num_chunks, 64)
    # Adjust if S is small
    num_chunks = min(num_chunks, S)
    return num_chunks


# =============================================================================
# Phase 1: RoPE + Max computation (parallel across chunks)
# Uses linearized iteration for coalesced memory access
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["chunk_size", "D"],
)
@triton.jit
def rope_qkv_phase1_kernel(
    # Input tensors [B, S, H, D]
    q_ptr,
    k_ptr,
    v_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Output: partial max values [B * H * num_chunks, 3]
    partial_max_ptr,
    # Input strides (for [B, S, H, D] layout)
    stride_b,
    stride_s,
    stride_h,
    stride_d,
    # Dimensions
    S,
    D,
    H,
    chunk_size,
    num_chunks,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Apply RoPE to Q and K, compute partial max for all tensors.

    Grid: (B, H, num_chunks)
    Each block processes chunk_size * D elements for one (batch, head) pair.

    Uses linearized iteration: treats chunk_size * D as a flat array and
    computes (s, d) coordinates from linear offsets. This ensures coalesced
    memory access (adjacent threads access adjacent D elements).
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointer for this (batch, head)
    base_b = pid_b * stride_b
    base_h = pid_h * stride_h

    # Initialize max accumulators
    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # Compute input pointer offset - coalesced access pattern
        ptr_offset = base_b + s_idx * stride_s + base_h + d_idx * stride_d

        # Load Q, K, V values (coalesced: adjacent threads access adjacent d)
        q_vals = tl.load(q_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)
        k_vals = tl.load(k_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)
        v_vals = tl.load(v_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for RoPE (indexed by [s_idx, d_idx])
        cos_offset = s_idx * D + d_idx
        cos_vals = tl.load(cos_ptr + cos_offset, mask=mask, other=1.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)

        # RoPE rotation: compute pair indices
        # pair_d = d ^ 1 swaps 0<->1, 2<->3, etc.
        # This creates access pattern 1,0,3,2,5,4,... which is mostly coalesced
        pair_d = d_idx ^ 1
        pair_offset = base_b + s_idx * stride_s + base_h + pair_d * stride_d

        # Load paired values for rotation (XOR pattern is cache-friendly)
        q_pair = tl.load(q_ptr + pair_offset, mask=mask, other=0.0).to(tl.float32)
        k_pair = tl.load(k_ptr + pair_offset, mask=mask, other=0.0).to(tl.float32)

        # Compute sign: -1 for even d indices, +1 for odd d indices
        is_even = (d_idx % 2) == 0
        sign = tl.where(is_even, -1.0, 1.0)

        # Compute rotated values and apply RoPE
        q_rotated = sign * q_pair
        k_rotated = sign * k_pair
        q_rope = q_vals * cos_vals + q_rotated * sin_vals
        k_rope = k_vals * cos_vals + k_rotated * sin_vals

        # Update max values (V doesn't get RoPE)
        q_max = tl.maximum(q_max, tl.max(tl.abs(q_rope)))
        k_max = tl.maximum(k_max, tl.max(tl.abs(k_rope)))
        v_max = tl.maximum(v_max, tl.max(tl.abs(v_vals)))

    # Store partial maxes: layout is (B * H * num_chunks, 3)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx * 3 + 0, q_max)
    tl.store(partial_max_ptr + chunk_idx * 3 + 1, k_max)
    tl.store(partial_max_ptr + chunk_idx * 3 + 2, v_max)


# =============================================================================
# Reduce kernel: Aggregate partial maxes and compute scales
# =============================================================================


@triton.jit
def rope_qkv_reduce_kernel(
    partial_max_ptr,  # [B * H * num_chunks, 3]
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    H,
    num_chunks,
):
    """
    Reduce partial maxes and compute scales/descales.

    Grid: (B, H)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Reduce across chunks for this (batch, head)
    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    base_idx = (pid_b * H + pid_h) * num_chunks * 3
    for c in range(num_chunks):
        idx = base_idx + c * 3
        q_max = tl.maximum(q_max, tl.load(partial_max_ptr + idx + 0))
        k_max = tl.maximum(k_max, tl.load(partial_max_ptr + idx + 1))
        v_max = tl.maximum(v_max, tl.load(partial_max_ptr + idx + 2))

    # Compute scales and descales
    # FP8 E4M3 max value is 448.0
    FP8_MAX = 448.0
    eps = 1e-12
    scale_idx = pid_b * H + pid_h

    q_scale = tl.where(q_max > eps, FP8_MAX / q_max, 1.0)
    k_scale = tl.where(k_max > eps, FP8_MAX / k_max, 1.0)
    v_scale = tl.where(v_max > eps, FP8_MAX / v_max, 1.0)

    tl.store(q_scale_ptr + scale_idx, q_scale)
    tl.store(k_scale_ptr + scale_idx, k_scale)
    tl.store(v_scale_ptr + scale_idx, v_scale)

    tl.store(q_descale_ptr + scale_idx, tl.where(q_max > eps, q_max / FP8_MAX, 1.0))
    tl.store(k_descale_ptr + scale_idx, tl.where(k_max > eps, k_max / FP8_MAX, 1.0))
    tl.store(v_descale_ptr + scale_idx, tl.where(v_max > eps, v_max / FP8_MAX, 1.0))


# =============================================================================
# Phase 2: RoPE + Quantize (parallel across chunks)
# Uses linearized iteration for coalesced memory access
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["chunk_size", "D"],
)
@triton.jit
def rope_qkv_phase2_kernel(
    # Input tensors [B, S, H, D]
    q_ptr,
    k_ptr,
    v_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Output tensors [B, H, S, D] - transposed for SDPA
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    # Precomputed scales [B, H]
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
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
    # Dimensions
    S,
    D,
    H,
    chunk_size,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Apply RoPE to Q and K, then quantize all tensors to FP8.

    Grid: (B, H, num_chunks)

    Input format: [B, S, H, D]
    Output format: [B, H, S, D] (transposed for SDPA)

    Uses linearized iteration: treats chunk_size * D as a flat array.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Load scales for this head
    scale_idx = pid_b * H + pid_h
    q_scale = tl.load(q_scale_ptr + scale_idx)
    k_scale = tl.load(k_scale_ptr + scale_idx)
    v_scale = tl.load(v_scale_ptr + scale_idx)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointers
    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h
    out_base_b = pid_b * stride_out_b
    out_base_h = pid_h * stride_out_h

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # Input offset (B, S, H, D layout) - coalesced access
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d
        # Output offset (B, H, S, D layout) - coalesced access
        out_offset = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_idx * stride_out_d
        )

        # Load Q, K, V values (coalesced)
        q_vals = tl.load(q_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)
        k_vals = tl.load(k_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)
        v_vals = tl.load(v_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for RoPE
        cos_offset = s_idx * D + d_idx
        cos_vals = tl.load(cos_ptr + cos_offset, mask=mask, other=1.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)

        # RoPE rotation: compute pair indices (XOR pattern)
        pair_d = d_idx ^ 1
        pair_in_offset = (
            in_base_b + s_idx * stride_in_s + in_base_h + pair_d * stride_in_d
        )

        # Load paired values (XOR pattern is cache-friendly)
        q_pair = tl.load(q_ptr + pair_in_offset, mask=mask, other=0.0).to(tl.float32)
        k_pair = tl.load(k_ptr + pair_in_offset, mask=mask, other=0.0).to(tl.float32)

        # Compute sign for rotation
        is_even = (d_idx % 2) == 0
        sign = tl.where(is_even, -1.0, 1.0)

        # Compute rotated values and apply RoPE
        q_rotated = sign * q_pair
        k_rotated = sign * k_pair
        q_rope = q_vals * cos_vals + q_rotated * sin_vals
        k_rope = k_vals * cos_vals + k_rotated * sin_vals

        # Quantize to FP8
        q_fp8 = (q_rope * q_scale).to(tl.float8e4nv)
        k_fp8 = (k_rope * k_scale).to(tl.float8e4nv)
        v_fp8 = (v_vals * v_scale).to(tl.float8e4nv)

        # Store to output (transposed layout, coalesced)
        tl.store(q_out_ptr + out_offset, q_fp8, mask=mask)
        tl.store(k_out_ptr + out_offset, k_fp8, mask=mask)
        tl.store(v_out_ptr + out_offset, v_fp8, mask=mask)


# =============================================================================
# Main entry point
# =============================================================================


def fp8_rope_quantize_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Fused RoPE + FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then quantizes all tensors to FP8 with per-head scaling.
    Also performs layout transformation from [B, S, H, D] to [B, H, S, D].

    Args:
        q: Query tensor of shape [B, S, H, D] in bf16/fp16
        k: Key tensor of shape [B, S, H, D] in bf16/fp16
        v: Value tensor of shape [B, S, H, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query with RoPE, shape [B, H, S, D] in fp8
        k_fp8: Quantized key with RoPE, shape [B, H, S, D] in fp8
        v_fp8: Quantized value, shape [B, H, S, D] in fp8
        q_descale: Query descale factors, shape [B, H] in fp32
        k_descale: Key descale factors, shape [B, H] in fp32
        v_descale: Value descale factors, shape [B, H] in fp32
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {v.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
    assert cos.dim() == 2, f"Expected 2D cos tensor [S, D], got {cos.dim()}D"
    assert sin.dim() == 2, f"Expected 2D sin tensor [S, D], got {sin.dim()}D"

    B, S, H, D = q.shape

    assert cos.shape == (S, D), f"Expected cos shape [{S}, {D}], got {cos.shape}"
    assert sin.shape == (S, D), f"Expected sin shape [{S}, {D}], got {sin.shape}"

    # Make tensors contiguous if needed
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    # Compute number of chunks
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    # Allocate output tensors in [B, H, S, D] layout for SDPA
    q_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    v_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)

    # Allocate temporary buffer for partial maxes
    partial_max = torch.empty(
        B * H * num_chunks, 3, dtype=torch.float32, device=q.device
    )

    # Allocate scale/descale tensors
    q_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)

    # Phase 1: Apply RoPE and compute partial maxes
    grid_phase1 = (B, H, num_chunks)
    rope_qkv_phase1_kernel[grid_phase1](
        q,
        k,
        v,
        cos,
        sin,
        partial_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S,
        D,
        H,
        chunk_size,
        num_chunks,
    )

    # Reduce: Aggregate maxes and compute scales
    rope_qkv_reduce_kernel[(B, H)](
        partial_max,
        q_scale,
        k_scale,
        v_scale,
        q_descale,
        k_descale,
        v_descale,
        H,
        num_chunks,
    )

    # Phase 2: Apply RoPE and quantize
    grid_phase2 = (B, H, num_chunks)
    rope_qkv_phase2_kernel[grid_phase2](
        q,
        k,
        v,
        cos,
        sin,
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        # Input strides [B, S, H, D]
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # Output strides [B, H, S, D]
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        S,
        D,
        H,
        chunk_size,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
