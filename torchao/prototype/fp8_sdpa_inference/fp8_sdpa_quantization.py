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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["total_elements"],
)
@triton.jit
def fp8_per_head_quant_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    total_elements,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
    H,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused quantization for Q, K, V tensors.
    Processes all three tensors in a single kernel launch.

    Input:  Q, K, V each (B, H, S, D) in bf16/fp16
    Output: Q, K, V each (B, H, S, D) in fp8, descales each (B, H) in fp32
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Compute base offset once
    base_offset = pid_b * stride_b + pid_h * stride_h

    # Pass 1: Find max for all three tensors
    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    for block_start in range(0, total_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements

        # Convert linear offset to (s, d) coordinates
        s_idx = offs // D
        d_idx = offs % D
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

    # Compute scales (multiply is faster than divide in the loop)
    q_scale = tl.where(q_max > 1e-12, 448.0 / q_max, 1.0)
    k_scale = tl.where(k_max > 1e-12, 448.0 / k_max, 1.0)
    v_scale = tl.where(v_max > 1e-12, 448.0 / v_max, 1.0)

    # Pass 2: Quantize all three tensors
    for block_start in range(0, total_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements

        # Convert linear offset to (s, d) coordinates
        s_idx = offs // D
        d_idx = offs % D
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

    # Store descales
    descale_idx = pid_b * H + pid_h
    tl.store(q_descale_ptr + descale_idx, tl.where(q_max > 1e-12, q_max / 448.0, 1.0))
    tl.store(k_descale_ptr + descale_idx, tl.where(k_max > 1e-12, k_max / 448.0, 1.0))
    tl.store(v_descale_ptr + descale_idx, tl.where(v_max > 1e-12, v_max / 448.0, 1.0))


def fp8_per_head_quant_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Fused quantization for Q, K, V tensors with per-head scaling.
    Single kernel launch for all three tensors.

    Args:
        q: Query tensor of shape (B, H, S, D) in bf16/fp16
        k: Key tensor of shape (B, H, S, D) in bf16/fp16
        v: Value tensor of shape (B, H, S, D) in bf16/fp16

    Returns:
        q_fp8: Quantized query tensor of shape (B, H, S, D) in fp8
        k_fp8: Quantized key tensor of shape (B, H, S, D) in fp8
        v_fp8: Quantized value tensor of shape (B, H, S, D) in fp8
        q_descale: Query descale factors of shape (B, H) in fp32
        k_descale: Key descale factors of shape (B, H) in fp32
        v_descale: Value descale factors of shape (B, H) in fp32
    """
    assert q.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {v.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
    assert q.is_contiguous(), "Q must be contiguous"
    assert k.is_contiguous(), "K must be contiguous"
    assert v.is_contiguous(), "V must be contiguous"

    B, H, S, D = q.shape

    # Allocate output tensors
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)
    q_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    k_descale = torch.empty(B, H, device=k.device, dtype=torch.float32)
    v_descale = torch.empty(B, H, device=v.device, dtype=torch.float32)

    grid = (B, H)
    total_elements = S * D

    fp8_per_head_quant_qkv_kernel[grid](
        q,
        k,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        total_elements,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        H,
        D,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


# =============================================================================
# Parallelized version: Better SM utilization by splitting work across chunks
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
def fp8_partial_max_kernel(
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
def fp8_quantize_with_scale_kernel(
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


@triton.jit
def reduce_partial_max_and_compute_scales_kernel(
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


def fp8_per_head_quant_qkv_parallel(
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

    Args:
        q: Query tensor of shape (B, H, S, D) in bf16/fp16
        k: Key tensor of shape (B, H, S, D) in bf16/fp16
        v: Value tensor of shape (B, H, S, D) in bf16/fp16
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8, k_fp8, v_fp8: Quantized tensors
        q_descale, k_descale, v_descale: Descale factors
    """
    assert q.dim() == 4, f"Expected 4D tensor (B, H, S, D), got {q.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    B, H, S, D = q.shape

    # Auto-select num_chunks based on GPU properties
    if num_chunks is None:
        props = torch.cuda.get_device_properties(q.device)
        num_sms = props.multi_processor_count
        base_parallelism = B * H
        # Target 2x SMs for good occupancy/latency hiding
        target_blocks = num_sms * 2
        num_chunks = max(1, target_blocks // base_parallelism)
        # Ensure each chunk has at least 64 S positions for efficiency
        num_chunks = min(num_chunks, S // 64) if S >= 64 else 1
        # Cap at reasonable maximum
        num_chunks = min(num_chunks, 32)

    # Adjust num_chunks if S is small
    num_chunks = min(num_chunks, S)
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
    fp8_partial_max_kernel[grid_phase1](
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
    reduce_partial_max_and_compute_scales_kernel[(B, H)](
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
    fp8_quantize_with_scale_kernel[grid_phase2](
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
