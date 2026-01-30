# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused FP8 RoPE + V-only Hadamard + Quantization for SDPA.

This module provides Triton kernels that fuse:
- RoPE (Rotary Position Embeddings) for Q and K
- Hadamard transform for V only (improves quantization by spreading outliers)
- FP8 quantization for Q, K, V

The fused 3-kernel sequence reduces memory traffic by applying RoPE + quantization in a single pass
Note: The 3-kernel sequence is used for chunking the sequence for parallelization.

Note: Since V is Hadamard-transformed, the caller must apply inverse Hadamard
to the attention output to recover correct results:
    output = inverse_hadamard(softmax(Q @ K^T) @ V)

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
# Hadamard transform helper
# =============================================================================


@triton.jit
def _hadamard_butterfly_stage(
    x,
    temp_ptr,
    temp_base,
    d_idx,
    stage: tl.constexpr,
    D: tl.constexpr,
):
    """
    Perform one stage of the Hadamard butterfly transform.

    The butterfly operation requires swapping elements between different d_idx
    positions. Use global memory as a shuffle buffer:
    1. Store the full D-vector to temp buffer
    2. Use tl.debug_barrier() to ensure stores complete
    3. Load with reordered indices (partner_d) to get the swapped elements

    Args:
        x: Current D-element vector (vectorized across threads)
        temp_ptr: Pointer to temp buffer base
        temp_base: Offset to this block's region in temp buffer
        d_idx: Vectorized index tensor (tl.arange(0, D))
        stage: Butterfly stage (0 to log2(D)-1)
        D: Head dimension (compile-time constant)
    """
    stride = 1 << stage
    partner_d = d_idx ^ stride
    is_left = (d_idx & stride) == 0

    # Store current value to temp buffer (vectorized store)
    tl.store(temp_ptr + temp_base + d_idx, x)

    # Block-level barrier to ensure all stores are visible before loads
    tl.debug_barrier()

    # Load partner value from temp buffer (vectorized load with reordering)
    x_partner = tl.load(temp_ptr + temp_base + partner_d)

    # Barrier after load to ensure all threads finish reading before next stage writes
    # Without this, threads starting the next stage could overwrite temp before
    # slower threads finish reading from the current stage
    tl.debug_barrier()

    # Butterfly: left gets sum, right gets difference
    return tl.where(is_left, x + x_partner, x_partner - x)


# =============================================================================
# Phase 1: RoPE for Q/K + Hadamard for V + Max computation
# Each block has D threads processing chunk_size S positions
# Hadamard is applied to V only (not Q and K) in this variant
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def v_hadamard_rope_qkv_phase1_kernel(
    # Input tensors [B, S, H, D]
    q_ptr,
    k_ptr,
    v_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Temp buffer for Hadamard [B, H, num_chunks, D] - one D-vector per (b, h, chunk) triple
    temp_ptr,
    # Intermediate output tensors [B, H, S, D] - stores RoPE'd Q, K and Hadamard'd V
    q_rope_ptr,
    k_rope_ptr,
    v_out_ptr,
    # Output: partial max values [B * H * num_chunks, 3]
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
    Phase 1: Apply RoPE to Q/K, Hadamard to V, store to intermediate buffers, compute partial max.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across all S positions in chunk.

    IMPORTANT: This requires inverse Hadamard on the attention output to recover correct results.

    Uses per-chunk temp buffer to avoid race conditions between chunks.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Thread index = d index (each thread handles one element of D)
    d_idx = tl.arange(0, D)

    # Temp buffer base for this (b, h, chunk) triple - unique per chunk to avoid races
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size

    # Base pointer for this (batch, head)
    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h

    # Output base pointers [B, H, S, D]
    out_base_b = pid_b * stride_out_b
    out_base_h = pid_h * stride_out_h

    # Initialize max accumulators
    q_max = tl.zeros([D], dtype=tl.float32)
    k_max = tl.zeros([D], dtype=tl.float32)
    v_max = tl.zeros([D], dtype=tl.float32)

    # Precompute normalization factor for Hadamard
    inv_sqrt_d = 1.0 / tl.sqrt(float(D))

    # Precompute sign for RoPE (loop-invariant)
    is_even = (d_idx % 2) == 0
    sign = tl.where(is_even, -1.0, 1.0)

    # Process each S position in this chunk
    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset

        # Mask for valid S positions
        s_mask = s_idx < S

        # Compute input pointer offset for this (s, d) position
        in_offset = in_base_b + s_idx * stride_in_s + in_base_h + d_idx * stride_in_d

        # Output offset [B, H, S, D]
        out_offset = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_idx * stride_out_d
        )

        # Load Q, K, V values
        q_vals = tl.load(q_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)
        k_vals = tl.load(k_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)
        v_vals = tl.load(v_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)

        # Load cos/sin for RoPE
        cos_offset = s_idx * D + d_idx
        cos_vals = tl.load(cos_ptr + cos_offset, mask=s_mask, other=1.0).to(tl.float32)
        sin_vals = tl.load(sin_ptr + cos_offset, mask=s_mask, other=0.0).to(tl.float32)

        # RoPE rotation: pair_d = d ^ 1 swaps 0<->1, 2<->3, etc.
        pair_d = d_idx ^ 1
        pair_in_offset = (
            in_base_b + s_idx * stride_in_s + in_base_h + pair_d * stride_in_d
        )

        # Load paired values for rotation
        q_pair = tl.load(q_ptr + pair_in_offset, mask=s_mask, other=0.0).to(tl.float32)
        k_pair = tl.load(k_ptr + pair_in_offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply RoPE to Q and K using FMA for better performance
        q_rotated = sign * q_pair
        k_rotated = sign * k_pair
        q_rope = tl.math.fma(q_vals, cos_vals, q_rotated * sin_vals)
        k_rope = tl.math.fma(k_vals, cos_vals, k_rotated * sin_vals)

        # Q and K are NOT Hadamard-transformed in this variant
        # Just store them after RoPE

        # Apply Hadamard transform to V only
        # Unrolled butterfly stages for common D values
        v_had = v_vals
        if LOG2_D >= 1:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 0, D)
        if LOG2_D >= 2:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 1, D)
        if LOG2_D >= 3:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 2, D)
        if LOG2_D >= 4:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 3, D)
        if LOG2_D >= 5:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 4, D)
        if LOG2_D >= 6:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 5, D)
        if LOG2_D >= 7:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 6, D)
        if LOG2_D >= 8:
            v_had = _hadamard_butterfly_stage(v_had, temp_ptr, temp_base, d_idx, 7, D)
        v_had = v_had * inv_sqrt_d

        # Store results to intermediate buffers [B, H, S, D]
        # Cast to input dtype (bf16/fp16) to reduce memory bandwidth
        if USE_BFLOAT16:
            tl.store(q_rope_ptr + out_offset, q_rope.to(tl.bfloat16), mask=s_mask)
            tl.store(k_rope_ptr + out_offset, k_rope.to(tl.bfloat16), mask=s_mask)
            tl.store(v_out_ptr + out_offset, v_had.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(q_rope_ptr + out_offset, q_rope.to(tl.float16), mask=s_mask)
            tl.store(k_rope_ptr + out_offset, k_rope.to(tl.float16), mask=s_mask)
            tl.store(v_out_ptr + out_offset, v_had.to(tl.float16), mask=s_mask)

        # Update max values (element-wise max across all S positions)
        q_max = tl.maximum(q_max, tl.abs(q_rope))
        k_max = tl.maximum(k_max, tl.abs(k_rope))
        v_max = tl.maximum(v_max, tl.abs(v_had))

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
def v_hadamard_rope_qkv_reduce_kernel(
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
# Phase 2: Quantize pre-computed RoPE'd Q/K and Hadamard'd V
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
def v_hadamard_rope_qkv_phase2_kernel(
    # Intermediate tensors [B, H, S, D] - already RoPE'd Q, K and Hadamard'd V
    q_rope_ptr,
    k_rope_ptr,
    v_had_ptr,
    # Output tensors [B, H, S, D] - FP8 quantized
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    # Precomputed scales [B, H]
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    # Strides (for [B, H, S, D] layout) - same for intermediate and output
    stride_b,
    stride_h,
    stride_s,
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
    Phase 2: Quantize pre-computed RoPE'd Q, K and Hadamard'd V to FP8.

    Grid: (B, H, num_chunks)

    Q, K are RoPE'd (no Hadamard), V is Hadamard'd (no RoPE).
    Uses linearized iteration for optimal thread utilization.
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

    # Base pointers [B, H, S, D]
    base_b = pid_b * stride_b
    base_h = pid_h * stride_h

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # Offset [B, H, S, D]
        offset = base_b + base_h + s_idx * stride_s + d_idx * stride_d

        # Load pre-computed RoPE'd Q, K and Hadamard'd V from intermediate buffers
        q_rope = tl.load(q_rope_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        k_rope = tl.load(k_rope_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        v_had = tl.load(v_had_ptr + offset, mask=mask, other=0.0).to(tl.float32)

        # Quantize to FP8
        q_fp8 = (q_rope * q_scale).to(tl.float8e4nv)
        k_fp8 = (k_rope * k_scale).to(tl.float8e4nv)
        v_fp8 = (v_had * v_scale).to(tl.float8e4nv)

        # Store to output
        tl.store(q_out_ptr + offset, q_fp8, mask=mask)
        tl.store(k_out_ptr + offset, k_fp8, mask=mask)
        tl.store(v_out_ptr + offset, v_fp8, mask=mask)


# =============================================================================
# Main entry point
# =============================================================================


def fp8_v_hadamard_rope_quantize_func(
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
    Fused RoPE + V-only Hadamard + FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then Hadamard transform to V only,
    then quantizes all tensors to FP8 with per-head scaling. Also performs
    layout transformation from [B, S, H, D] to [B, H, S, D].

    This requires inverse Hadamard on the attention output to recover correct
    results: output = inv_hadamard(softmax(Q @ K^T) @ hadamard(V))

    Args:
        q: Query tensor of shape [B, S, H, D] in bf16/fp16
        k: Key tensor of shape [B, S, H, D] in bf16/fp16
        v: Value tensor of shape [B, S, H, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query with RoPE (no Hadamard), shape [B, H, S, D] in fp8
        k_fp8: Quantized key with RoPE (no Hadamard), shape [B, H, S, D] in fp8
        v_fp8: Quantized value with Hadamard (no RoPE), shape [B, H, S, D] in fp8
        q_descale: Query descale factors, shape [B, H] in fp32
        k_descale: Key descale factors, shape [B, H] in fp32
        v_descale: Value descale factors, shape [B, H] in fp32

    Note:
        D must be a power of 2 for the Hadamard transform.
        The caller must apply inverse Hadamard to the attention output.
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
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Compute number of chunks
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    # Allocate output tensors in [B, H, S, D] layout for SDPA
    q_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    v_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)

    # Allocate temp buffer for Hadamard transform [B, H, num_chunks, D]
    # Each (b, h, chunk) triple gets its own D-sized buffer to avoid race conditions
    temp_buffer = torch.empty(B, H, num_chunks, D, dtype=torch.float32, device=q.device)

    # Allocate intermediate buffers for RoPE'd Q, K and Hadamard'd V in [B, H, S, D] layout
    # Use input dtype (bf16/fp16) to reduce memory bandwidth
    q_rope_intermediate = torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)
    k_rope_intermediate = torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)
    v_intermediate = torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)

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

    # Phase 1: Apply RoPE to Q/K, Hadamard to V, compute partial maxes
    grid_phase1 = (B, H, num_chunks)
    use_bfloat16 = q.dtype == torch.bfloat16
    v_hadamard_rope_qkv_phase1_kernel[grid_phase1](
        q,
        k,
        v,
        cos,
        sin,
        temp_buffer,
        q_rope_intermediate,
        k_rope_intermediate,
        v_intermediate,
        partial_max,
        # Input strides [B, S, H, D]
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # Output strides [B, H, S, D]
        q_rope_intermediate.stride(0),
        q_rope_intermediate.stride(1),
        q_rope_intermediate.stride(2),
        q_rope_intermediate.stride(3),
        # Temp buffer strides [B, H, num_chunks, D]
        temp_buffer.stride(0),
        temp_buffer.stride(1),
        temp_buffer.stride(2),
        temp_buffer.stride(3),
        S,
        H,
        chunk_size,
        num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
    )

    # Reduce: Aggregate maxes and compute scales
    v_hadamard_rope_qkv_reduce_kernel[(B, H)](
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

    # Phase 2: Read from intermediate buffers and quantize to FP8
    grid_phase2 = (B, H, num_chunks)
    v_hadamard_rope_qkv_phase2_kernel[grid_phase2](
        q_rope_intermediate,
        k_rope_intermediate,
        v_intermediate,
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        # Strides [B, H, S, D] - same for intermediate and output
        q_rope_intermediate.stride(0),
        q_rope_intermediate.stride(1),
        q_rope_intermediate.stride(2),
        q_rope_intermediate.stride(3),
        S,
        D,
        H,
        chunk_size,
        num_chunks,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
