# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for fused per-head FP8 quantization of Q, K, V tensors.

Uses a 3-phase approach:
  1. Phase 1: Compute partial absmax values per chunk (parallelized across S)
  2. Reduce: Reduce partial maxes across chunks, compute scale/descale
  3. Phase 2: Apply scales and cast to FP8

Three variants are provided depending on whether Q, K, V share the same shape:
  - Fused QKV: all three share the same shape (optimal, single kernel set)
  - Single Q: quantize Q alone (for cross-attention where Q has different S)
  - Fused KV: quantize K and V together (they always share the same shape)
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


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


# =============================================================================
# Fused QKV quantization (when Q, K, V have the same shape)
# =============================================================================

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["chunk_size", "D"])
@triton.jit
def _qkv_phase1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    partial_max_ptr,  # (B * H * num_chunks, 3)
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
    """Phase 1: Compute partial absmax for Q, K, V per chunk."""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    s_start = pid_chunk * chunk_size
    s_end = tl.minimum(s_start + chunk_size, S)
    chunk_elements = (s_end - s_start) * D

    base_offset = pid_b * stride_b + pid_h * stride_h

    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    for block_start in range(0, chunk_elements, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_elements

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

    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx * 3 + 0, q_max)
    tl.store(partial_max_ptr + chunk_idx * 3 + 1, k_max)
    tl.store(partial_max_ptr + chunk_idx * 3 + 2, v_max)


@triton.jit
def _qkv_reduce_kernel(
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
    """Reduce partial maxes across chunks and compute scale/descale factors."""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    q_max = 0.0
    k_max = 0.0
    v_max = 0.0

    base_idx = (pid_b * H + pid_h) * num_chunks * 3
    for c in range(num_chunks):
        idx = base_idx + c * 3
        q_max = tl.maximum(q_max, tl.load(partial_max_ptr + idx + 0))
        k_max = tl.maximum(k_max, tl.load(partial_max_ptr + idx + 1))
        v_max = tl.maximum(v_max, tl.load(partial_max_ptr + idx + 2))

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


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["chunk_size", "D"])
@triton.jit
def _qkv_phase2_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_scale_ptr,
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
    """Phase 2: Quantize Q, K, V to FP8 using precomputed scales."""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    scale_idx = pid_b * H + pid_h
    q_scale = tl.load(q_scale_ptr + scale_idx)
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

        q = tl.load(q_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        k = tl.load(k_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        v = tl.load(v_ptr + base_offset + ptr_offset, mask=mask, other=0.0).to(
            tl.float32
        )

        tl.store(q_out_ptr + base_offset + ptr_offset, (q * q_scale).to(tl.float8e4nv), mask=mask)
        tl.store(k_out_ptr + base_offset + ptr_offset, (k * k_scale).to(tl.float8e4nv), mask=mask)
        tl.store(v_out_ptr + base_offset + ptr_offset, (v * v_scale).to(tl.float8e4nv), mask=mask)


def _triton_qkv_quantize(
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
    """Fused FP8 quantization for Q, K, V when all have the same shape."""
    B, H, S, D = q.shape

    if num_chunks is None:
        num_chunks = _compute_num_chunks(q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)

    partial_max = torch.empty(
        B * H * num_chunks, 3, device=q.device, dtype=torch.float32
    )

    q_scale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    k_scale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    v_scale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    q_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    k_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)
    v_descale = torch.empty(B, H, device=q.device, dtype=torch.float32)

    _qkv_phase1_kernel[(B, H, num_chunks)](
        q, k, v, partial_max,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        S, D, chunk_size, num_chunks, H,
    )

    _qkv_reduce_kernel[(B, H)](
        partial_max,
        q_scale, k_scale, v_scale,
        q_descale, k_descale, v_descale,
        H, num_chunks,
    )

    _qkv_phase2_kernel[(B, H, num_chunks)](
        q, k, v, q_fp8, k_fp8, v_fp8,
        q_scale, k_scale, v_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        S, D, H, chunk_size,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


# =============================================================================
# Single tensor quantization (for Q when shapes differ from K/V)
# =============================================================================


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["chunk_size", "D"])
@triton.jit
def _single_phase1_kernel(
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
    """Phase 1: Compute partial absmax for a single tensor."""
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
def _single_reduce_kernel(
    partial_max_ptr,  # (B * H * num_chunks,)
    scale_ptr,
    descale_ptr,
    H,
    num_chunks,
):
    """Reduce partial maxes and compute scale for a single tensor."""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    x_max = 0.0
    base_idx = (pid_b * H + pid_h) * num_chunks
    for c in range(num_chunks):
        x_max = tl.maximum(x_max, tl.load(partial_max_ptr + base_idx + c))

    scale_idx = pid_b * H + pid_h
    tl.store(scale_ptr + scale_idx, tl.where(x_max > 1e-12, 448.0 / x_max, 1.0))
    tl.store(descale_ptr + scale_idx, tl.where(x_max > 1e-12, x_max / 448.0, 1.0))


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["chunk_size", "D"])
@triton.jit
def _single_phase2_kernel(
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
    """Phase 2: Quantize a single tensor to FP8 using precomputed scale."""
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    scale = tl.load(scale_ptr + pid_b * H + pid_h)

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
        tl.store(x_out_ptr + base_offset + ptr_offset, (x * scale).to(tl.float8e4nv), mask=mask)


def _triton_single_quantize(
    x: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallelized FP8 quantization for a single tensor."""
    B, H, S, D = x.shape

    if num_chunks is None:
        num_chunks = _compute_num_chunks(x, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    partial_max = torch.empty(B * H * num_chunks, device=x.device, dtype=torch.float32)
    scale = torch.empty(B, H, device=x.device, dtype=torch.float32)
    descale = torch.empty(B, H, device=x.device, dtype=torch.float32)

    _single_phase1_kernel[(B, H, num_chunks)](
        x, partial_max,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        S, D, chunk_size, num_chunks, H,
    )

    _single_reduce_kernel[(B, H)](partial_max, scale, descale, H, num_chunks)

    _single_phase2_kernel[(B, H, num_chunks)](
        x, x_fp8, scale,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        S, D, H, chunk_size,
    )

    return x_fp8, descale


# =============================================================================
# K/V pair quantization (for cross-attention where Q has different S from K/V)
# =============================================================================


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["chunk_size", "D"])
@triton.jit
def _kv_phase1_kernel(
    k_ptr,
    v_ptr,
    partial_max_ptr,  # (B * H * num_chunks, 2)
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
    """Phase 1: Compute partial absmax for K and V."""
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
def _kv_reduce_kernel(
    partial_max_ptr,  # (B * H * num_chunks, 2)
    k_scale_ptr,
    v_scale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    H,
    num_chunks,
):
    """Reduce partial maxes and compute scales for K and V."""
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

    tl.store(k_scale_ptr + scale_idx, tl.where(k_max > 1e-12, 448.0 / k_max, 1.0))
    tl.store(v_scale_ptr + scale_idx, tl.where(v_max > 1e-12, 448.0 / v_max, 1.0))
    tl.store(k_descale_ptr + scale_idx, tl.where(k_max > 1e-12, k_max / 448.0, 1.0))
    tl.store(v_descale_ptr + scale_idx, tl.where(v_max > 1e-12, v_max / 448.0, 1.0))


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["chunk_size", "D"])
@triton.jit
def _kv_phase2_kernel(
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
    """Phase 2: Quantize K and V to FP8 using precomputed scales."""
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

        tl.store(k_out_ptr + base_offset + ptr_offset, (k * k_scale).to(tl.float8e4nv), mask=mask)
        tl.store(v_out_ptr + base_offset + ptr_offset, (v * v_scale).to(tl.float8e4nv), mask=mask)


def _triton_kv_quantize(
    k: torch.Tensor,
    v: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallelized FP8 quantization for K and V tensors."""
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

    _kv_phase1_kernel[(B, H, num_chunks)](
        k, v, partial_max,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        S, D, chunk_size, num_chunks, H,
    )

    _kv_reduce_kernel[(B, H)](
        partial_max,
        k_scale, v_scale,
        k_descale, v_descale,
        H, num_chunks,
    )

    _kv_phase2_kernel[(B, H, num_chunks)](
        k, v, k_fp8, v_fp8,
        k_scale, v_scale,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        S, D, H, chunk_size,
    )

    return k_fp8, v_fp8, k_descale, v_descale


# =============================================================================
# Main entry point with shape-based dispatch
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
    Fused per-head FP8 quantization for Q, K, V using Triton kernels.

    Supports both same-shape (fused kernel) and different-shape (separate kernels)
    cases. Q can have a different sequence length than K/V (cross-attention).

    Args:
        q: Query tensor of shape (B, H, Sq, D) in bf16/fp16
        k: Key tensor of shape (B, H, Skv, D) in bf16/fp16
        v: Value tensor of shape (B, H, Skv, D) in bf16/fp16
        num_chunks: Number of chunks to split S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8, k_fp8, v_fp8: Quantized tensors in float8_e4m3fn
        q_descale, k_descale, v_descale: Descale factors of shape (B, H)
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if q.shape == k.shape:
        return _triton_qkv_quantize(q, k, v, num_chunks)
    else:
        q_fp8, q_descale = _triton_single_quantize(q, num_chunks)
        k_fp8, v_fp8, k_descale, v_descale = _triton_kv_quantize(k, v, num_chunks)
        return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
