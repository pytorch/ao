# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 quantization utilities for attention inputs.
Blockwise quantization for Q, K, V tensors.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# =============================================================================
# Parallelized version: Better SM utilization by splitting work across chunks
# =============================================================================


def _compute_num_chunks(tensor: torch.Tensor, S: int) -> int:
    """Compute optimal number of chunks based on GPU properties."""
    props = torch.cuda.get_device_properties(tensor.device)
    num_sms = props.multi_processor_count
    B, H = tensor.shape[:2]
    base_parallelism = B * H
    # Target 2x SMs for good occupancy/latency hiding
    target_blocks = num_sms * 2
    num_chunks = max(1, target_blocks // base_parallelism)
    # Ensure each chunk has at least 64 S positions for efficiency
    num_chunks = min(num_chunks, S // 64) if S >= 64 else 1
    # Cap at reasonable maximum
    num_chunks = min(num_chunks, 32)
    # Adjust if S is small
    num_chunks = min(num_chunks, S)
    return num_chunks


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
def qkv_phase1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    partial_max_ptr,  # (B * H * num_chunks, 3) - stores max for Q, K, V
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    S,
    D,
    chunk_size,  # Number of S positions per chunk
    num_chunks,
    H,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Compute partial max for each chunk.
    Grid: (B, H, num_chunks)
    Each block processes chunk_size * D elements.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    # Compute the S range for this chunk
    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    # Base pointer for this (batch, head)
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Find max for this chunk
    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset within chunk to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        q = tl.load(q_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        k = tl.load(k_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        v = tl.load(v_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )

        q_max = tl.maximum(q_max, tl.max(tl.abs(q)))
        k_max = tl.maximum(k_max, tl.max(tl.abs(k)))
        v_max = tl.maximum(v_max, tl.max(tl.abs(v)))

    # Store partial maxes: layout is (B * H * num_chunks, 3)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx * 3 + 0, q_max)
    tl.store(partial_max_ptr + chunk_idx * 3 + 1, k_max)
    tl.store(partial_max_ptr + chunk_idx * 3 + 2, v_max)


@triton.jit
def qkv_reduce_kernel(
    partial_max_ptr,  # (B * H * num_chunks, 3)
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
    Reduces partial maxes and computes scales in a single kernel.
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
    scale_idx = pid_b * H + pid_h

    q_scale = tl.where(q_max > 1e-12, 448.0 / q_max, 1.0)
    k_scale = tl.where(k_max > 1e-12, 448.0 / k_max, 1.0)
    v_scale = tl.where(v_max > 1e-12, 448.0 / v_max, 1.0)

    tl.store(q_scale_ptr + scale_idx, q_scale)
    tl.store(k_scale_ptr + scale_idx, k_scale)
    tl.store(v_scale_ptr + scale_idx, v_scale)
    tl.store(q_descale_ptr + scale_idx, tl.where(q_max > 1e-12, q_max / 448.0, 1.0))
    tl.store(k_descale_ptr + scale_idx, tl.where(k_max > 1e-12, k_max / 448.0, 1.0))
    tl.store(v_descale_ptr + scale_idx, tl.where(v_max > 1e-12, v_max / 448.0, 1.0))


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
def qkv_phase2_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_scale_ptr,  # (B, H) - precomputed scales
    k_scale_ptr,
    v_scale_ptr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    S,
    D,
    H,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Quantize using precomputed scales.
    Grid: (B, H, num_chunks)
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

    # Base pointer for this (batch, head)
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Quantize this chunk
    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        # Convert linear offset within chunk to (s, d) coordinates
        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        q = tl.load(q_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        k = tl.load(k_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        v = tl.load(v_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )

        q_fp8 = (q * q_scale).to(tl.float8e4nv)
        k_fp8 = (k * k_scale).to(tl.float8e4nv)
        v_fp8 = (v * v_scale).to(tl.float8e4nv)

        tl.store(q_out_ptr + base_offset + ptr_offset, q_fp8, mask=mask)
        tl.store(k_out_ptr + base_offset + ptr_offset, k_fp8, mask=mask)
        tl.store(v_out_ptr + base_offset + ptr_offset, v_fp8, mask=mask)


def qkv_quantize_func(
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
    Fused FP8 quantization for Q, K, V tensors when all have the same shape.
    Internal function - use fp8_per_head_quant_qkv_parallel as the entry point.
    """
    B, H, S, D = q.shape

    if num_chunks is None:
        num_chunks = _compute_num_chunks(q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks  # Ceiling division

    # Allocate output tensors
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)

    # Temporary buffer for partial maxes: (B * H * num_chunks, 3)
    partial_max = torch.empty(
        B * H * num_chunks, 3, device=q.device, dtype=torch.float32
    )

    # Allocate scale/descale tensors
    q_scale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    k_scale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    v_scale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    q_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    k_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    v_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)

    # Phase 1: Compute partial maxes
    grid_phase1 = (B, H, num_chunks)
    qkv_phase1_kernel[grid_phase1](
        q,
        k,
        v,
        partial_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S,
        D,
        chunk_size,
        num_chunks,
        H,
    )

    # Fused reduction and scale computation in Triton
    qkv_reduce_kernel[(B, H)](
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

    # Phase 2: Quantize with precomputed scales
    grid_phase2 = (B, H, num_chunks)
    qkv_phase2_kernel[grid_phase2](
        q,
        k,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        S,
        D,
        H,
        chunk_size,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


# =============================================================================
# Single tensor quantization (for Q when shapes differ)
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
def q_phase1_kernel(
    x_ptr,
    partial_max_ptr,  # (B * H * num_chunks,)
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    S,
    D,
    chunk_size,
    num_chunks,
    H,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Compute partial max for a single tensor.
    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    base_offset = pid_b * stride_b + pid_h * stride_h

    x_max = 0.0

    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        x = tl.load(x_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        x_max = tl.maximum(x_max, tl.max(tl.abs(x)))

    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max)


@triton.jit
def q_reduce_kernel(
    partial_max_ptr,  # (B * H * num_chunks,)
    scale_ptr,
    descale_ptr,
    H,
    num_chunks,
):
    """
    Reduce partial maxes and compute scale for a single tensor.
    Grid: (B, H)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    x_max = 0.0
    base_idx = (pid_b * H + pid_h) * num_chunks
    for c in range(num_chunks):
        x_max = tl.maximum(x_max, tl.load(partial_max_ptr + base_idx + c))

    scale_idx = pid_b * H + pid_h
    scale = tl.where(x_max > 1e-12, 448.0 / x_max, 1.0)
    descale = tl.where(x_max > 1e-12, x_max / 448.0, 1.0)

    tl.store(scale_ptr + scale_idx, scale)
    tl.store(descale_ptr + scale_idx, descale)


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
def q_phase2_kernel(
    x_ptr,
    x_out_ptr,
    scale_ptr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    S,
    D,
    H,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Quantize a single tensor using precomputed scale.
    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    scale_idx = pid_b * H + pid_h
    scale = tl.load(scale_ptr + scale_idx)

    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    base_offset = pid_b * stride_b + pid_h * stride_h

    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        x = tl.load(x_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        x_fp8 = (x * scale).to(tl.float8e4nv)
        tl.store(x_out_ptr + base_offset + ptr_offset, x_fp8, mask=mask)


def q_quantize_func(
    x: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parallelized FP8 quantization for a single tensor.

    Args:
        x: Input tensor of shape (B, H, S, D) in bf16/fp16
        num_chunks: Number of chunks to split S dimension into.

    Returns:
        x_fp8: Quantized tensor of shape (B, H, S, D) in fp8
        x_descale: Descale factors of shape (B, H) in fp32
    """
    B, H, S, D = x.shape

    if num_chunks is None:
        num_chunks = _compute_num_chunks(x, S)

    chunk_size = (S + num_chunks - 1) // num_chunks

    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    partial_max = torch.empty(B * H * num_chunks, device=x.device, dtype=torch.float32)
    scale = torch.empty(B, H, device=x.device, dtype=torch.float32)
    descale = torch.empty(B, H, device=x.device, dtype=torch.float32)

    # Phase 1: Compute partial maxes
    q_phase1_kernel[(B, H, num_chunks)](
        x,
        partial_max,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        S,
        D,
        chunk_size,
        num_chunks,
        H,
    )

    # Reduce and compute scales
    q_reduce_kernel[(B, H)](
        partial_max,
        scale,
        descale,
        H,
        num_chunks,
    )

    # Phase 2: Quantize
    q_phase2_kernel[(B, H, num_chunks)](
        x,
        x_fp8,
        scale,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        S,
        D,
        H,
        chunk_size,
    )

    return x_fp8, descale


# =============================================================================
# K/V pair quantization (for when Q shape differs from K/V)
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
def kv_phase1_kernel(
    k_ptr,
    v_ptr,
    partial_max_ptr,  # (B * H * num_chunks, 2) - stores max for K, V
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    S,
    D,
    chunk_size,
    num_chunks,
    H,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Compute partial max for K and V tensors.
    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    base_offset = pid_b * stride_b + pid_h * stride_h

    k_max = 0.0
    v_max = 0.0

    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        k = tl.load(k_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        v = tl.load(v_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )

        k_max = tl.maximum(k_max, tl.max(tl.abs(k)))
        v_max = tl.maximum(v_max, tl.max(tl.abs(v)))

    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx * 2 + 0, k_max)
    tl.store(partial_max_ptr + chunk_idx * 2 + 1, v_max)


@triton.jit
def kv_reduce_kernel(
    partial_max_ptr,  # (B * H * num_chunks, 2)
    k_scale_ptr,
    v_scale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    H,
    num_chunks,
):
    """
    Reduce partial maxes and compute scales for K and V.
    Grid: (B, H)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    k_max = 0.0
    v_max = 0.0

    base_idx = (pid_b * H + pid_h) * num_chunks * 2
    for c in range(num_chunks):
        idx = base_idx + c * 2
        k_max = tl.maximum(k_max, tl.load(partial_max_ptr + idx + 0))
        v_max = tl.maximum(v_max, tl.load(partial_max_ptr + idx + 1))

    scale_idx = pid_b * H + pid_h

    k_scale = tl.where(k_max > 1e-12, 448.0 / k_max, 1.0)
    v_scale = tl.where(v_max > 1e-12, 448.0 / v_max, 1.0)

    tl.store(k_scale_ptr + scale_idx, k_scale)
    tl.store(v_scale_ptr + scale_idx, v_scale)
    tl.store(k_descale_ptr + scale_idx, tl.where(k_max > 1e-12, k_max / 448.0, 1.0))
    tl.store(v_descale_ptr + scale_idx, tl.where(v_max > 1e-12, v_max / 448.0, 1.0))


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
def kv_phase2_kernel(
    k_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    S,
    D,
    H,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Quantize K and V using precomputed scales.
    Grid: (B, H, num_chunks)
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    scale_idx = pid_b * H + pid_h
    k_scale = tl.load(k_scale_ptr + scale_idx)
    v_scale = tl.load(v_scale_ptr + scale_idx)

    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    base_offset = pid_b * stride_b + pid_h * stride_h

    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

        local_s = offs // D
        d_idx = offs % D
        s_idx = s_start + local_s
        ptr_offset = s_idx * stride_s + d_idx * stride_d

        k = tl.load(k_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        v = tl.load(v_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )

        k_fp8 = (k * k_scale).to(tl.float8e4nv)
        v_fp8 = (v * v_scale).to(tl.float8e4nv)

        tl.store(k_out_ptr + base_offset + ptr_offset, k_fp8, mask=mask)
        tl.store(v_out_ptr + base_offset + ptr_offset, v_fp8, mask=mask)


def kv_quantize_func(
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parallelized FP8 quantization for K and V tensors.

    Args:
        k: Key tensor of shape (B, H, S, D) in bf16/fp16
        v: Value tensor of shape (B, H, S, D) in bf16/fp16
        num_chunks: Number of chunks to split S dimension into.

    Returns:
        k_fp8, v_fp8: Quantized tensors
        k_descale, v_descale: Descale factors
    """
    assert k.shape == v.shape, "K and V must have the same shape"

    B, H, S, D = k.shape

    if num_chunks is None:
        num_chunks = _compute_num_chunks(k, S)

    chunk_size = (S + num_chunks - 1) // num_chunks

    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    partial_max = torch.empty(
        B * H * num_chunks, 2, device=k.device, dtype=torch.float32
    )
    k_scale = torch.empty(B, H, device=k.device, dtype=torch.float32)
    v_scale = torch.empty(B, H, device=k.device, dtype=torch.float32)
    k_descale = torch.empty(B, H, device=k.device, dtype=torch.float32)
    v_descale = torch.empty(B, H, device=k.device, dtype=torch.float32)

    # Phase 1: Compute partial maxes
    kv_phase1_kernel[(B, H, num_chunks)](
        k,
        v,
        partial_max,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        S,
        D,
        chunk_size,
        num_chunks,
        H,
    )

    # Reduce and compute scales
    kv_reduce_kernel[(B, H)](
        partial_max,
        k_scale,
        v_scale,
        k_descale,
        v_descale,
        H,
        num_chunks,
    )

    # Phase 2: Quantize
    kv_phase2_kernel[(B, H, num_chunks)](
        k,
        v,
        k_fp8,
        v_fp8,
        k_scale,
        v_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        S,
        D,
        H,
        chunk_size,
    )

    return k_fp8, v_fp8, k_descale, v_descale


# =============================================================================
# Main entry point with shape-based dispatch
# =============================================================================


def fp8_sdpa_quantize_func(
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
    Parallelized FP8 quantization for Q, K, V tensors.
    Splits work across multiple thread blocks for better SM utilization.

    Supports both same-shape (fused kernel) and different-shape (separate kernels)
    cases. Q can have a different sequence length than K/V (e.g., cross-attention,
    KV-cache scenarios).

    Args:
        q: Query tensor of shape (B, H, Sq, D) in bf16/fp16
        k: Key tensor of shape (B, H, Skv, D) in bf16/fp16
        v: Value tensor of shape (B, H, Skv, D) in bf16/fp16
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8, k_fp8, v_fp8: Quantized tensors
        q_descale, k_descale, v_descale: Descale factors
    """
    assert q.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {v.dim()}D"
    assert k.shape == v.shape, "K and V must have the same shape"

    # Check that B, H, D match (only S can differ)
    assert q.shape[0] == k.shape[0], "Batch size must match"
    assert q.shape[1] == k.shape[1], "Number of heads must match"
    assert q.shape[3] == k.shape[3], "Head dimension must match"

    # Make contiguous if needed
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # Dispatch based on shape
    if q.shape == k.shape:
        # Use fused kernel for same shapes (optimal path)
        return qkv_quantize_func(q, k, v, num_chunks)
    else:
        # Use separate kernels for different shapes
        q_fp8, q_descale = q_quantize_func(q, num_chunks)
        k_fp8, v_fp8, k_descale, v_descale = kv_quantize_func(k, v, num_chunks)
        return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
