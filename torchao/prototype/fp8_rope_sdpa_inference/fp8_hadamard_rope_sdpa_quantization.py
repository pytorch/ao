# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused FP8 RoPE + Hadamard + Quantization for SDPA.

This module provides Triton kernels that fuse:
- RoPE (Rotary Position Embeddings) for Q and K
- Hadamard transform for Q, K, V (improves quantization by spreading outliers)
- FP8 quantization for Q, K, V

The Hadamard transform improves per-head FP8 quantization by redistributing
outlier values across the head dimension, leading to better dynamic range
utilization. The transform is orthogonal and can be inverted after dequantization.

Input format: [B, S, H, D] (FLUX-style)
Output format: [B, H, S, D] (SDPA-style)

Design note: Since D is typically small (64-128), the kernel processes multiple
S positions per block to maximize SM utilization. Each block has D threads that
collaboratively process BLOCK_S sequence positions.
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


def _get_log2_d(D: int) -> int:
    """Get log2(D), asserting D is a power of 2."""
    assert D > 0 and (D & (D - 1)) == 0, f"D must be a power of 2, got {D}"
    log2_d = 0
    temp = D
    while temp > 1:
        temp >>= 1
        log2_d += 1
    return log2_d


# =============================================================================
# Hadamard transform helper (in-place via temp buffer)
# =============================================================================


@triton.jit
def _hadamard_butterfly_stage(
    x,
    temp_ptr,
    temp_base,
    d_idx,
    stage: tl.constexpr,
):
    """
    Perform one stage of the Hadamard butterfly transform.

    Uses temp buffer for inter-thread data exchange within the block.
    Each thread handles one D element; all threads collaborate on the butterfly.
    """
    stride = 1 << stage
    partner_d = d_idx ^ stride
    is_left = (d_idx & stride) == 0

    # Store current value to temp buffer
    tl.store(temp_ptr + temp_base + d_idx, x)

    # Load partner value (guaranteed to see stored value due to block-level consistency)
    x_partner = tl.load(temp_ptr + temp_base + partner_d)

    # Butterfly: left gets sum, right gets difference
    return tl.where(is_left, x + x_partner, x_partner - x)


# =============================================================================
# Phase 1: RoPE + Hadamard + Max computation
# Each block has D threads processing chunk_size S positions
# =============================================================================


@triton.jit
def hadamard_rope_qkv_phase1_kernel(
    # Input tensors [B, S, H, D]
    q_ptr,
    k_ptr,
    v_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Temp buffers for Hadamard [B, H, D] - one D-vector per (b, h) pair
    q_temp_ptr,
    k_temp_ptr,
    v_temp_ptr,
    # Output: partial max values [B * H * num_chunks, 3]
    partial_max_ptr,
    # Input strides (for [B, S, H, D] layout)
    stride_b,
    stride_s,
    stride_h,
    stride_d,
    # Dimensions
    S,
    H,
    chunk_size,
    num_chunks,
    # Compile-time constants
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """
    Phase 1: Apply RoPE + Hadamard to Q/K, Hadamard to V, compute partial max.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.

    The Hadamard transform uses a temp buffer for butterfly data exchange.
    Since we process one S position at a time, the temp buffer only needs
    D elements per (b, h) pair.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Thread index = d index (each thread handles one element of D)
    d_idx = tl.arange(0, D)

    # Temp buffer base for this (b, h) pair
    temp_base = (pid_b * H + pid_h) * D

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size

    # Base pointer for this (batch, head)
    base_b = pid_b * stride_b
    base_h = pid_h * stride_h

    # Initialize max accumulators
    q_max = tl.zeros([D], dtype=tl.float32)
    k_max = tl.zeros([D], dtype=tl.float32)
    v_max = tl.zeros([D], dtype=tl.float32)

    # Process each S position in this chunk
    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset

        # Early exit for positions beyond S
        if s_idx >= S:
            continue

        # Compute input pointer offset for this (s, d) position
        ptr_offset = base_b + s_idx * stride_s + base_h + d_idx * stride_d

        # Load Q, K, V values
        q_vals = tl.load(q_ptr + ptr_offset).to(tl.float32)
        k_vals = tl.load(k_ptr + ptr_offset).to(tl.float32)
        v_vals = tl.load(v_ptr + ptr_offset).to(tl.float32)

        # Load cos/sin for RoPE
        cos_offset = s_idx * D + d_idx
        cos_vals = tl.load(cos_ptr + cos_offset).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offset).to(tl.float32)

        # RoPE rotation: pair_d = d ^ 1 swaps 0<->1, 2<->3, etc.
        pair_d = d_idx ^ 1
        pair_offset = base_b + s_idx * stride_s + base_h + pair_d * stride_d

        # Load paired values for rotation
        q_pair = tl.load(q_ptr + pair_offset).to(tl.float32)
        k_pair = tl.load(k_ptr + pair_offset).to(tl.float32)

        # Compute sign: -1 for even d indices, +1 for odd d indices
        is_even = (d_idx % 2) == 0
        sign = tl.where(is_even, -1.0, 1.0)

        # Apply RoPE
        q_rotated = sign * q_pair
        k_rotated = sign * k_pair
        q_rope = q_vals * cos_vals + q_rotated * sin_vals
        k_rope = k_vals * cos_vals + k_rotated * sin_vals

        # Apply Hadamard transform to Q
        # Unrolled butterfly stages for common D values
        if LOG2_D >= 1:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 0)
        if LOG2_D >= 2:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 1)
        if LOG2_D >= 3:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 2)
        if LOG2_D >= 4:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 3)
        if LOG2_D >= 5:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 4)
        if LOG2_D >= 6:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 5)
        if LOG2_D >= 7:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 6)
        if LOG2_D >= 8:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 7)
        q_rope = q_rope / tl.sqrt(D.to(tl.float32))

        # Apply Hadamard transform to K
        if LOG2_D >= 1:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 0)
        if LOG2_D >= 2:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 1)
        if LOG2_D >= 3:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 2)
        if LOG2_D >= 4:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 3)
        if LOG2_D >= 5:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 4)
        if LOG2_D >= 6:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 5)
        if LOG2_D >= 7:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 6)
        if LOG2_D >= 8:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 7)
        k_rope = k_rope / tl.sqrt(D.to(tl.float32))

        # Apply Hadamard transform to V (no RoPE)
        if LOG2_D >= 1:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 0)
        if LOG2_D >= 2:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 1)
        if LOG2_D >= 3:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 2)
        if LOG2_D >= 4:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 3)
        if LOG2_D >= 5:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 4)
        if LOG2_D >= 6:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 5)
        if LOG2_D >= 7:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 6)
        if LOG2_D >= 8:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 7)
        v_vals = v_vals / tl.sqrt(D.to(tl.float32))

        # Update max values (element-wise max across all S positions)
        q_max = tl.maximum(q_max, tl.abs(q_rope))
        k_max = tl.maximum(k_max, tl.abs(k_rope))
        v_max = tl.maximum(v_max, tl.abs(v_vals))

    # Reduce max across D dimension
    q_max_scalar = tl.max(q_max)
    k_max_scalar = tl.max(k_max)
    v_max_scalar = tl.max(v_max)

    # Store partial maxes: layout is (B * H * num_chunks, 3)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx * 3 + 0, q_max_scalar)
    tl.store(partial_max_ptr + chunk_idx * 3 + 1, k_max_scalar)
    tl.store(partial_max_ptr + chunk_idx * 3 + 2, v_max_scalar)


# =============================================================================
# Reduce kernel: Aggregate partial maxes and compute scales
# =============================================================================


@triton.jit
def hadamard_rope_qkv_reduce_kernel(
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
# Phase 2: RoPE + Hadamard + Quantize
# Each block has D threads processing chunk_size S positions
# =============================================================================


@triton.jit
def hadamard_rope_qkv_phase2_kernel(
    # Input tensors [B, S, H, D]
    q_ptr,
    k_ptr,
    v_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Temp buffers for Hadamard [B, H, D]
    q_temp_ptr,
    k_temp_ptr,
    v_temp_ptr,
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
    H,
    chunk_size,
    # Compile-time constants
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """
    Phase 2: Apply RoPE + Hadamard, then quantize to FP8.

    Grid: (B, H, num_chunks)
    Block: D threads

    Input format: [B, S, H, D]
    Output format: [B, H, S, D] (transposed for SDPA)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Thread index = d index
    d_idx = tl.arange(0, D)

    # Load scales for this head
    scale_idx = pid_b * H + pid_h
    q_scale = tl.load(q_scale_ptr + scale_idx)
    k_scale = tl.load(k_scale_ptr + scale_idx)
    v_scale = tl.load(v_scale_ptr + scale_idx)

    # Temp buffer base for this (b, h) pair
    temp_base = (pid_b * H + pid_h) * D

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size

    # Base pointers
    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h
    out_base_b = pid_b * stride_out_b
    out_base_h = pid_h * stride_out_h

    # Process each S position in this chunk
    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset

        # Early exit for positions beyond S
        if s_idx >= S:
            continue

        # Input offset (B, S, H, D layout)
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d
        # Output offset (B, H, S, D layout)
        out_offset = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_idx * stride_out_d
        )

        # Load Q, K, V values
        q_vals = tl.load(q_ptr + in_offset).to(tl.float32)
        k_vals = tl.load(k_ptr + in_offset).to(tl.float32)
        v_vals = tl.load(v_ptr + in_offset).to(tl.float32)

        # Load cos/sin for RoPE
        cos_offset = s_idx * D + d_idx
        cos_vals = tl.load(cos_ptr + cos_offset).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offset).to(tl.float32)

        # RoPE rotation
        pair_d = d_idx ^ 1
        pair_in_offset = (
            in_base_b + s_idx * stride_in_s + in_base_h + pair_d * stride_in_d
        )

        q_pair = tl.load(q_ptr + pair_in_offset).to(tl.float32)
        k_pair = tl.load(k_ptr + pair_in_offset).to(tl.float32)

        is_even = (d_idx % 2) == 0
        sign = tl.where(is_even, -1.0, 1.0)

        q_rotated = sign * q_pair
        k_rotated = sign * k_pair
        q_rope = q_vals * cos_vals + q_rotated * sin_vals
        k_rope = k_vals * cos_vals + k_rotated * sin_vals

        # Apply Hadamard transform to Q
        if LOG2_D >= 1:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 0)
        if LOG2_D >= 2:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 1)
        if LOG2_D >= 3:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 2)
        if LOG2_D >= 4:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 3)
        if LOG2_D >= 5:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 4)
        if LOG2_D >= 6:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 5)
        if LOG2_D >= 7:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 6)
        if LOG2_D >= 8:
            q_rope = _hadamard_butterfly_stage(q_rope, q_temp_ptr, temp_base, d_idx, 7)
        q_rope = q_rope / tl.sqrt(D.to(tl.float32))

        # Apply Hadamard transform to K
        if LOG2_D >= 1:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 0)
        if LOG2_D >= 2:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 1)
        if LOG2_D >= 3:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 2)
        if LOG2_D >= 4:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 3)
        if LOG2_D >= 5:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 4)
        if LOG2_D >= 6:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 5)
        if LOG2_D >= 7:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 6)
        if LOG2_D >= 8:
            k_rope = _hadamard_butterfly_stage(k_rope, k_temp_ptr, temp_base, d_idx, 7)
        k_rope = k_rope / tl.sqrt(D.to(tl.float32))

        # Apply Hadamard transform to V
        if LOG2_D >= 1:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 0)
        if LOG2_D >= 2:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 1)
        if LOG2_D >= 3:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 2)
        if LOG2_D >= 4:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 3)
        if LOG2_D >= 5:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 4)
        if LOG2_D >= 6:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 5)
        if LOG2_D >= 7:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 6)
        if LOG2_D >= 8:
            v_vals = _hadamard_butterfly_stage(v_vals, v_temp_ptr, temp_base, d_idx, 7)
        v_vals = v_vals / tl.sqrt(D.to(tl.float32))

        # Quantize to FP8
        q_fp8 = (q_rope * q_scale).to(tl.float8e4nv)
        k_fp8 = (k_rope * k_scale).to(tl.float8e4nv)
        v_fp8 = (v_vals * v_scale).to(tl.float8e4nv)

        # Store to output (transposed layout)
        tl.store(q_out_ptr + out_offset, q_fp8)
        tl.store(k_out_ptr + out_offset, k_fp8)
        tl.store(v_out_ptr + out_offset, v_fp8)


# =============================================================================
# Main entry point
# =============================================================================


def fp8_hadamard_rope_quantize_func(
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
    Fused RoPE + Hadamard + FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then Hadamard transform to all tensors, then
    quantizes to FP8 with per-head scaling. Also performs layout transformation
    from [B, S, H, D] to [B, H, S, D].

    The Hadamard transform improves FP8 quantization quality by spreading
    outlier values across the head dimension, leading to better utilization
    of the dynamic range.

    Args:
        q: Query tensor of shape [B, S, H, D] in bf16/fp16
        k: Key tensor of shape [B, S, H, D] in bf16/fp16
        v: Value tensor of shape [B, S, H, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query with RoPE+Hadamard, shape [B, H, S, D] in fp8
        k_fp8: Quantized key with RoPE+Hadamard, shape [B, H, S, D] in fp8
        v_fp8: Quantized value with Hadamard, shape [B, H, S, D] in fp8
        q_descale: Query descale factors, shape [B, H] in fp32
        k_descale: Key descale factors, shape [B, H] in fp32
        v_descale: Value descale factors, shape [B, H] in fp32

    Note:
        To recover original values after dequantization, apply inverse Hadamard
        (which is the same as forward Hadamard for normalized transforms).
        D must be a power of 2.
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, S, H, D], got {v.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
    assert cos.dim() == 2, f"Expected 2D cos tensor [S, D], got {cos.dim()}D"
    assert sin.dim() == 2, f"Expected 2D sin tensor [S, D], got {sin.dim()}D"

    B, S, H, D = q.shape
    LOG2_D = _get_log2_d(D)

    assert cos.shape == (S, D), f"Expected cos shape [{S}, {D}], got {cos.shape}"
    assert sin.shape == (S, D), f"Expected sin shape [{S}, {D}], got {sin.shape}"
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"

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

    # Allocate temporary buffers for Hadamard transform [B, H, D]
    # These are reused across all S positions
    q_temp = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
    k_temp = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
    v_temp = torch.empty(B, H, D, dtype=torch.float32, device=q.device)

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

    # Phase 1: Apply RoPE + Hadamard and compute partial maxes
    grid_phase1 = (B, H, num_chunks)
    hadamard_rope_qkv_phase1_kernel[grid_phase1](
        q,
        k,
        v,
        cos,
        sin,
        q_temp,
        k_temp,
        v_temp,
        partial_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S,
        H,
        chunk_size,
        num_chunks,
        D=D,
        LOG2_D=LOG2_D,
    )

    # Reduce: Aggregate maxes and compute scales
    hadamard_rope_qkv_reduce_kernel[(B, H)](
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

    # Phase 2: Apply RoPE + Hadamard and quantize
    grid_phase2 = (B, H, num_chunks)
    hadamard_rope_qkv_phase2_kernel[grid_phase2](
        q,
        k,
        v,
        cos,
        sin,
        q_temp,
        k_temp,
        v_temp,
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
        H,
        chunk_size,
        D=D,
        LOG2_D=LOG2_D,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
