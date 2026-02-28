# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Separated FP8 Quantization kernels for Q, K, V.

This module processes Q, K, and V independently with separate kernel launches:
- Q: FP8 quantization (phase1 + reduce + phase2)
- K: FP8 quantization (phase1 + reduce + phase2)
- V: FP8 quantization (phase1 + reduce + phase2)

Each tensor is processed independently, allowing different head counts (GQA).

Input format: [B, H, S, D] (SDPA-style)
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
    B, H = tensor.shape[:2]  # [B, H, S, D]
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
# Phase 1: Compute partial absmax for a single tensor
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
def single_phase1_kernel(
    # Input tensor [B, H, S, D]
    x_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, H, S, D] layout)
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
    Phase 1 for a single tensor: Compute partial absmax.

    Grid: (B, H, num_chunks)

    Uses linearized iteration over chunk_size * D elements.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointer for input [B, H, S, D]
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Initialize max accumulator
    x_max = 0.0

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        # Input offset [B, H, S, D]
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        x_val = tl.load(x_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_val)))

    # Store partial max
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max)


# =============================================================================
# Reduce kernels
# =============================================================================


@triton.jit
def single_reduce_kernel(
    partial_max_ptr,  # [B * H * num_chunks]
    scale_ptr,
    descale_ptr,
    H,
    num_chunks,
):
    """
    Reduce partial maxes and compute scale/descale for a single tensor.

    Grid: (B, H)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Reduce across chunks for this (batch, head)
    x_max = 0.0

    base_idx = (pid_b * H + pid_h) * num_chunks
    for c in range(num_chunks):
        x_max = tl.maximum(x_max, tl.load(partial_max_ptr + base_idx + c))

    # Compute scale and descale
    # FP8 E4M3 max value is 448.0
    FP8_MAX = 448.0
    eps = 1e-12
    scale_idx = pid_b * H + pid_h

    tl.store(scale_ptr + scale_idx, tl.where(x_max > eps, FP8_MAX / x_max, 1.0))
    tl.store(descale_ptr + scale_idx, tl.where(x_max > eps, x_max / FP8_MAX, 1.0))


@triton.jit
def group_reduce_kernel(
    partial_max_ptr,  # [B * H_q * num_chunks]
    scale_ptr,  # [B, H_kv]
    descale_ptr,  # [B, H_kv]
    H_q,
    H_kv,
    groups,  # H_q // H_kv
    num_chunks,
):
    """
    Reduce partial maxes across head groups for GQA Q tensor.

    For each KV group, reduces the max across all Q heads in that group
    and all chunks, producing one scale per (batch, kv_head).

    Grid: (B, H_kv)
    """
    pid_b = tl.program_id(axis=0)
    pid_hkv = tl.program_id(axis=1)

    x_max = 0.0

    for g in range(groups):
        h_q = pid_hkv * groups + g
        base_idx = (pid_b * H_q + h_q) * num_chunks
        for c in range(num_chunks):
            x_max = tl.maximum(x_max, tl.load(partial_max_ptr + base_idx + c))

    FP8_MAX = 448.0
    eps = 1e-12
    scale_idx = pid_b * H_kv + pid_hkv

    tl.store(scale_ptr + scale_idx, tl.where(x_max > eps, FP8_MAX / x_max, 1.0))
    tl.store(descale_ptr + scale_idx, tl.where(x_max > eps, x_max / FP8_MAX, 1.0))


# =============================================================================
# Phase 2: Quantize to FP8
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
def single_phase2_kernel(
    # Input tensor [B, H, S, D]
    x_ptr,
    # Output tensor [B, H, S, D] - FP8 quantized
    x_out_ptr,
    # Precomputed scale [B, H_scale]
    scale_ptr,
    # Strides (for [B, H, S, D] layout)
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    # Dimensions
    S,
    D,
    H,
    chunk_size,
    # Scale indexing for GQA: scale has H_scale entries per batch,
    # and each group of `groups` heads shares one scale.
    # For non-GQA: H_scale = H, groups = 1.
    H_scale,
    groups,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2 for a single tensor: Quantize to FP8 using precomputed scale.

    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Load scale for this head (or head group for GQA)
    scale = tl.load(scale_ptr + pid_b * H_scale + pid_h // groups)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointer
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Linearized iteration over chunk_size * D elements
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s

        ptr_offset = base_offset + s_idx * stride_s + d_idx * stride_d

        # Load input value
        x_val = tl.load(x_ptr + ptr_offset, mask=mask, other=0.0).to(tl.float32)

        # Quantize to FP8
        x_fp8 = (x_val * scale).to(tl.float8e4nv)

        # Store to output
        tl.store(x_out_ptr + ptr_offset, x_fp8, mask=mask)


# =============================================================================
# Main entry point
# =============================================================================


def triton_fp8_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
    Separated FP8 quantization for Q, K, V tensors.

    Quantizes all tensors to FP8 with per-head scaling.
    Each of Q, K, V is processed with independent kernel launches,
    supporting GQA where Q has more heads than K/V (H_q = groups * H_kv).

    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    Args:
        q: Query tensor of shape [B, H_q, S, D] in bf16/fp16
        k: Key tensor of shape [B, H_kv, S, D] in bf16/fp16
        v: Value tensor of shape [B, H_kv, S, D] in bf16/fp16
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query, shape [B, H_q, S, D] in fp8
        k_fp8: Quantized key, shape [B, H_kv, S, D] in fp8
        v_fp8: Quantized value, shape [B, H_kv, S, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {v.dim()}D"
    assert k.shape == v.shape, (
        f"K and V must have the same shape, got {k.shape} vs {v.shape}"
    )
    assert q.shape[0] == k.shape[0], (
        f"Batch size mismatch: {q.shape[0]} vs {k.shape[0]}"
    )
    assert q.shape[2] == k.shape[2], (
        f"Sequence length mismatch: {q.shape[2]} vs {k.shape[2]}"
    )
    assert q.shape[3] == k.shape[3], f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}"
    assert q.shape[1] % k.shape[1] == 0, (
        f"Q heads ({q.shape[1]}) must be a multiple of K heads ({k.shape[1]})"
    )

    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    groups = H_q // H_kv

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Compute number of chunks
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    # Allocate output tensors
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)

    # Allocate partial max buffers (one per tensor)
    q_partial_max = torch.empty(
        B * H_q * num_chunks, dtype=torch.float32, device=q.device
    )
    k_partial_max = torch.empty(
        B * H_kv * num_chunks, dtype=torch.float32, device=q.device
    )
    v_partial_max = torch.empty(
        B * H_kv * num_chunks, dtype=torch.float32, device=q.device
    )

    # Allocate scale/descale tensors.
    # For GQA, Q scale/descale are [B, H_kv] (per KV group).
    # K and V are always [B, H_kv] (per head).
    q_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)

    q_grid_chunked = (B, H_q, num_chunks)
    kv_grid_chunked = (B, H_kv, num_chunks)

    # ---- Phase 1: Max for Q ----
    single_phase1_kernel[q_grid_chunked](
        q,
        q_partial_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S,
        D,
        H_q,
        chunk_size,
        num_chunks,
    )

    # ---- Phase 1: Max for K ----
    single_phase1_kernel[kv_grid_chunked](
        k,
        k_partial_max,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        num_chunks,
    )

    # ---- Phase 1: Max for V ----
    single_phase1_kernel[kv_grid_chunked](
        v,
        v_partial_max,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        num_chunks,
    )

    # ---- Reduce ----
    # Q: group reduce across `groups` Q heads per KV head
    group_reduce_kernel[(B, H_kv)](
        q_partial_max, q_scale, q_descale, H_q, H_kv, groups, num_chunks
    )
    # K, V: per-head reduce
    single_reduce_kernel[(B, H_kv)](k_partial_max, k_scale, k_descale, H_kv, num_chunks)
    single_reduce_kernel[(B, H_kv)](v_partial_max, v_scale, v_descale, H_kv, num_chunks)

    # ---- Phase 2: Quantize Q ----
    # Q scale is [B, H_kv]; each group of `groups` Q heads shares one scale.
    single_phase2_kernel[q_grid_chunked](
        q,
        q_fp8,
        q_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S,
        D,
        H_q,
        chunk_size,
        H_kv,
        groups,
    )

    # ---- Phase 2: Quantize K ----
    # K scale is [B, H_kv]; groups=1 (per-head).
    single_phase2_kernel[kv_grid_chunked](
        k,
        k_fp8,
        k_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        H_kv,
        1,
    )

    # ---- Phase 2: Quantize V ----
    # V scale is [B, H_kv]; groups=1 (per-head).
    single_phase2_kernel[kv_grid_chunked](
        v,
        v_fp8,
        v_scale,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        H_kv,
        1,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
