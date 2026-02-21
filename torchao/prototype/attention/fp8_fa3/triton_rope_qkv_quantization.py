# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused RoPE + FP8 Quantization kernels.

This module provides Triton kernels that fuse:
- RoPE (Rotary Position Embeddings) for Q and K
- FP8 quantization for Q, K, V

The fused 3-kernel sequence reduces memory traffic by applying RoPE + quantization in a single pass
Note: The 3-kernel sequence is used for chunking the sequence for parallelization.

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
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["chunk_size", "D_HALF"],
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
    # Intermediate output tensors [B, H, S, D] - stores RoPE'd Q, K
    q_rope_ptr,
    k_rope_ptr,
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
    # Dimensions
    S,
    D,
    D_HALF,
    H,
    chunk_size,
    num_chunks,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Apply RoPE to Q and K, store results, compute partial max.

    Grid: (B, H, num_chunks)

    Uses stride-2 access pattern to load each element exactly once:
    - Iterates over D/2 pairs instead of D elements
    - Loads real (even) and imag (odd) elements per pair
    - Applies complex multiplication for RoPE rotation with FMA
    - Uses linearized iteration for optimal thread utilization

    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    actual_chunk_size = s_end - s_start

    # Number of pairs to process in this chunk
    chunk_pairs = actual_chunk_size * D_HALF

    # Base pointers for input [B, S, H, D]
    in_base_b = pid_b * stride_in_b
    in_base_h = pid_h * stride_in_h

    # Base pointers for output [B, H, S, D]
    out_base_b = pid_b * stride_out_b
    out_base_h = pid_h * stride_out_h

    # Initialize max accumulators
    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    # Linearized iteration over chunk_size * D_HALF pairs
    for block_start in range(0, chunk_pairs, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_pairs

        # Convert linear offset to (s, pair_idx) coordinates
        local_s = offs // D_HALF
        pair_idx = offs % D_HALF
        s_idx = s_start + local_s

        # Real (even) and imag (odd) dimension indices
        d_real = pair_idx * 2
        d_imag = pair_idx * 2 + 1

        # Input offsets [B, S, H, D]
        in_offset_real = (
            in_base_b + s_idx * stride_in_s + in_base_h + d_real * stride_in_d
        )
        in_offset_imag = (
            in_base_b + s_idx * stride_in_s + in_base_h + d_imag * stride_in_d
        )

        # Output offsets [B, H, S, D]
        out_offset_real = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_real * stride_out_d
        )
        out_offset_imag = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_imag * stride_out_d
        )

        # Load Q, K, V pairs (each element loaded exactly once)
        q_real = tl.load(q_ptr + in_offset_real, mask=mask, other=0.0).to(tl.float32)
        q_imag = tl.load(q_ptr + in_offset_imag, mask=mask, other=0.0).to(tl.float32)
        k_real = tl.load(k_ptr + in_offset_real, mask=mask, other=0.0).to(tl.float32)
        k_imag = tl.load(k_ptr + in_offset_imag, mask=mask, other=0.0).to(tl.float32)
        v_real = tl.load(v_ptr + in_offset_real, mask=mask, other=0.0).to(tl.float32)
        v_imag = tl.load(v_ptr + in_offset_imag, mask=mask, other=0.0).to(tl.float32)

        # Load cos/sin for BOTH real and imag positions (NO halving)
        # cos/sin shape is [S, D], indexed by (s_idx, d_real) and (s_idx, d_imag)
        cos_offset_real = s_idx * D + d_real
        cos_offset_imag = s_idx * D + d_imag
        cos_real = tl.load(cos_ptr + cos_offset_real, mask=mask, other=1.0).to(
            tl.float32
        )
        sin_real = tl.load(sin_ptr + cos_offset_real, mask=mask, other=0.0).to(
            tl.float32
        )
        cos_imag = tl.load(cos_ptr + cos_offset_imag, mask=mask, other=1.0).to(
            tl.float32
        )
        sin_imag = tl.load(sin_ptr + cos_offset_imag, mask=mask, other=0.0).to(
            tl.float32
        )

        # Apply RoPE using complex multiplication with FMA
        # Standard RoPE: (x_real + i*x_imag) * (cos + i*sin)
        # Since cos[d_real] == cos[d_imag] and sin[d_real] == sin[d_imag] in standard RoPE,
        # we could use just one, but we load both here (no optimization)
        # out_real = x_real * cos - x_imag * sin
        # out_imag = x_real * sin + x_imag * cos
        q_rope_real = tl.math.fma(q_real, cos_real, -(q_imag * sin_real))
        q_rope_imag = tl.math.fma(q_real, sin_imag, q_imag * cos_imag)
        k_rope_real = tl.math.fma(k_real, cos_real, -(k_imag * sin_real))
        k_rope_imag = tl.math.fma(k_real, sin_imag, k_imag * cos_imag)

        # Store RoPE'd Q and K to intermediate buffers [B, H, S, D]
        tl.store(q_rope_ptr + out_offset_real, q_rope_real.to(q_real.dtype), mask=mask)
        tl.store(q_rope_ptr + out_offset_imag, q_rope_imag.to(q_real.dtype), mask=mask)
        tl.store(k_rope_ptr + out_offset_real, k_rope_real.to(k_real.dtype), mask=mask)
        tl.store(k_rope_ptr + out_offset_imag, k_rope_imag.to(k_real.dtype), mask=mask)

        # Update max values for Q and K (RoPE'd values)
        q_max = tl.maximum(q_max, tl.max(tl.abs(q_rope_real)))
        q_max = tl.maximum(q_max, tl.max(tl.abs(q_rope_imag)))
        k_max = tl.maximum(k_max, tl.max(tl.abs(k_rope_real)))
        k_max = tl.maximum(k_max, tl.max(tl.abs(k_rope_imag)))

        # Update max values for V (no RoPE, just max computation)
        v_max = tl.maximum(v_max, tl.max(tl.abs(v_real)))
        v_max = tl.maximum(v_max, tl.max(tl.abs(v_imag)))

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
# Phase 2: Quantize (parallel across chunks)
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
    # Intermediate tensors [B, H, S, D] - already RoPE'd Q, K
    q_rope_ptr,
    k_rope_ptr,
    # Original V tensor [B, S, H, D] - needs transpose only
    v_ptr,
    # Output tensors [B, H, S, D] - FP8 quantized
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    # Precomputed scales [B, H]
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    # V input strides (for [B, S, H, D] layout)
    stride_v_in_b,
    stride_v_in_s,
    stride_v_in_h,
    stride_v_in_d,
    # Output strides (for [B, H, S, D] layout) - same for intermediate and output
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
    Phase 2: Quantize pre-computed RoPE'd Q, K and transpose+quantize V.

    Grid: (B, H, num_chunks)

    Q and K are read from intermediate buffers (already RoPE'd in Phase 1).
    V is read from original input and transposed during quantization.
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

    # Base pointers for V input [B, S, H, D]
    v_in_base_b = pid_b * stride_v_in_b
    v_in_base_h = pid_h * stride_v_in_h

    # Base pointers for output [B, H, S, D] - same layout for intermediate and output
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

        # Output offset [B, H, S, D] - used for reading intermediate and writing output
        out_offset = (
            out_base_b + out_base_h + s_idx * stride_out_s + d_idx * stride_out_d
        )

        # V input offset [B, S, H, D]
        v_in_offset = (
            v_in_base_b + s_idx * stride_v_in_s + v_in_base_h + d_idx * stride_v_in_d
        )

        # Load pre-computed RoPE'd Q, K from intermediate buffers
        q_rope = tl.load(q_rope_ptr + out_offset, mask=mask, other=0.0).to(tl.float32)
        k_rope = tl.load(k_rope_ptr + out_offset, mask=mask, other=0.0).to(tl.float32)

        # Load V from original input (no RoPE, just transpose)
        v_vals = tl.load(v_ptr + v_in_offset, mask=mask, other=0.0).to(tl.float32)

        # Quantize to FP8
        q_fp8 = (q_rope * q_scale).to(tl.float8e4nv)
        k_fp8 = (k_rope * k_scale).to(tl.float8e4nv)
        v_fp8 = (v_vals * v_scale).to(tl.float8e4nv)

        # Store to output (same layout as intermediate)
        tl.store(q_out_ptr + out_offset, q_fp8, mask=mask)
        tl.store(k_out_ptr + out_offset, k_fp8, mask=mask)
        tl.store(v_out_ptr + out_offset, v_fp8, mask=mask)


# =============================================================================
# Main entry point
# =============================================================================


def triton_fp8_rope_sdpa_quantize(
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

    assert D % 2 == 0, f"Head dimension D must be even for RoPE, got D={D}"
    assert cos.shape == (S, D), f"Expected cos shape [{S}, {D}], got {cos.shape}"
    assert sin.shape == (S, D), f"Expected sin shape [{S}, {D}], got {sin.shape}"

    D_HALF = D // 2

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

    # Allocate intermediate buffers for RoPE'd Q, K in [B, H, S, D] layout
    q_rope_intermediate = torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)
    k_rope_intermediate = torch.empty(B, H, S, D, dtype=k.dtype, device=q.device)

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

    # Phase 1: Apply RoPE, store to intermediate, compute partial maxes
    # Uses FULL [S, D] cos/sin tensors (no halving)
    grid_phase1 = (B, H, num_chunks)
    rope_qkv_phase1_kernel[grid_phase1](
        q,
        k,
        v,
        cos,  # Full size [S, D]
        sin,  # Full size [S, D]
        q_rope_intermediate,
        k_rope_intermediate,
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
        S,
        D,
        D_HALF,
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

    # Phase 2: Read RoPE'd Q, K from intermediate, quantize all to FP8
    grid_phase2 = (B, H, num_chunks)
    rope_qkv_phase2_kernel[grid_phase2](
        q_rope_intermediate,
        k_rope_intermediate,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        # V input strides [B, S, H, D]
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
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
