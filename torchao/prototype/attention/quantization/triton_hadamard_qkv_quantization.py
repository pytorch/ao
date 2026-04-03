# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Hadamard + FP8 quantization kernels for Q, K, V.

Input/output format: [B, H, S, D].
Supports GQA (different head counts for Q vs K/V) and cross-attention
(different sequence lengths for Q vs K/V).

The Hadamard transform spreads outliers across the head dimension,
improving per-head FP8 quantization quality.

Phase 1 uses D threads per block (one per head-dim element) to apply
the butterfly-based Hadamard transform.  Phase 2 and reduce kernels
are reused from triton_qkv_quantization.py.
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
from torchao.prototype.attention.quantization.triton_qkv_quantization import (
    group_reduce_kernel,
    single_phase2_kernel,
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
def hadamard_single_phase1_kernel(
    # Input tensor [B, H, S, D]
    x_ptr,
    # Intermediate output tensor [B, H, S, D] - Hadamard'd
    x_had_ptr,
    # Temp buffer for Hadamard butterfly [B, H, num_chunks, D]
    temp_ptr,
    # Output: partial max values [B * H * num_chunks]
    partial_max_ptr,
    # Input strides (for [B, H, S, D] layout)
    stride_b,
    stride_h,
    stride_s,
    stride_d,
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
    Phase 1 for a single tensor: Apply Hadamard transform, store to
    intermediate buffer, compute partial absmax.

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
    base = pid_b * stride_b + pid_h * stride_h

    x_max = tl.zeros([D], dtype=tl.float32)

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        offset = base + s_idx * stride_s + d_idx * stride_d
        x = tl.load(x_ptr + offset, mask=s_mask, other=0.0).to(tl.float32)

        # Apply Hadamard transform with 1/sqrt(D) normalization
        x = _apply_hadamard(x, temp_ptr, temp_base, d_idx, D, LOG2_D)

        # Store to intermediate buffer in input dtype
        if USE_BFLOAT16:
            tl.store(x_had_ptr + offset, x.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(x_had_ptr + offset, x.to(tl.float16), mask=s_mask)

        x_max = tl.maximum(x_max, tl.abs(x))

    x_max_scalar = tl.max(x_max)
    chunk_idx = pid_b * (H * num_chunks) + pid_h * num_chunks + pid_chunk
    tl.store(partial_max_ptr + chunk_idx, x_max_scalar)


def triton_fp8_hadamard_sdpa_quantize(
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
    Hadamard + FP8 quantization for Q, K, V tensors.

    Applies Hadamard transform then quantizes to FP8 with per-head scaling.
    Each of Q, K, V is processed with independent kernel launches,
    supporting GQA where Q has more heads than K/V (H_q = groups * H_kv)
    and cross-attention where Q and K/V have different sequence lengths.

    For GQA, Q is quantized with per-KV-group scaling so that q_descale
    has shape [B, H_kv] as required by FA3.

    The caller must apply inverse Hadamard to the attention output:
        output = inverse_hadamard_transform(attention_output)

    Args:
        q: Query tensor of shape [B, H_q, S_q, D] in bf16/fp16
        k: Key tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        v: Value tensor of shape [B, H_kv, S_kv, D] in bf16/fp16
        num_chunks: Number of chunks to split the S dimension into.
                    If None, automatically selects based on GPU SM count.

    Returns:
        q_fp8: Quantized query with Hadamard, shape [B, H_q, S_q, D] in fp8
        k_fp8: Quantized key with Hadamard, shape [B, H_kv, S_kv, D] in fp8
        v_fp8: Quantized value with Hadamard, shape [B, H_kv, S_kv, D] in fp8
        q_descale: Query descale factors, shape [B, H_kv] in fp32
        k_descale: Key descale factors, shape [B, H_kv] in fp32
        v_descale: Value descale factors, shape [B, H_kv] in fp32

    Note:
        D must be a power of 2 and <= 256 for the Hadamard transform.
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
    assert q.shape[3] == k.shape[3], f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}"
    assert q.shape[1] % k.shape[1] == 0, (
        f"Q heads ({q.shape[1]}) must be a multiple of K heads ({k.shape[1]})"
    )

    B, H_q, S_q, D = q.shape
    H_kv = k.shape[1]
    S_kv = k.shape[2]
    groups = H_q // H_kv

    LOG2_D = _get_log2_d(D)
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"

    # Make tensors contiguous if needed
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    use_bfloat16 = q.dtype == torch.bfloat16

    # Compute number of chunks independently for Q and KV
    if num_chunks is None:
        q_num_chunks = _compute_num_chunks(q.device, B, H_q, S_q)
        kv_num_chunks = _compute_num_chunks(k.device, B, H_kv, S_kv)
    else:
        q_num_chunks = num_chunks
        kv_num_chunks = num_chunks
    q_chunk_size = (S_q + q_num_chunks - 1) // q_num_chunks
    kv_chunk_size = (S_kv + kv_num_chunks - 1) // kv_num_chunks

    # Allocate output tensors
    q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v, dtype=torch.float8_e4m3fn)

    # Intermediate buffers for Hadamard'd values (same layout/dtype as input)
    q_had = torch.empty_like(q)
    k_had = torch.empty_like(k)
    v_had = torch.empty_like(v)

    # Temp buffers for Hadamard butterfly (one D-vector per (b, h, chunk) triple)
    q_temp = torch.empty(B, H_q, q_num_chunks, D, dtype=torch.float32, device=q.device)
    kv_temp = torch.empty(
        B, H_kv, kv_num_chunks, D, dtype=torch.float32, device=q.device
    )

    # Partial max buffers
    q_partial_max = torch.empty(
        B * H_q * q_num_chunks, dtype=torch.float32, device=q.device
    )
    k_partial_max = torch.empty(
        B * H_kv * kv_num_chunks, dtype=torch.float32, device=q.device
    )
    v_partial_max = torch.empty(
        B * H_kv * kv_num_chunks, dtype=torch.float32, device=q.device
    )

    # Scale/descale tensors (per KV group for Q, per head for K/V)
    q_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(B, H_kv, dtype=torch.float32, device=q.device)

    q_grid = (B, H_q, q_num_chunks)
    kv_grid = (B, H_kv, kv_num_chunks)

    # ---- Phase 1: Hadamard + max for Q ----
    hadamard_single_phase1_kernel[q_grid](
        q,
        q_had,
        q_temp,
        q_partial_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q_temp.stride(0),
        q_temp.stride(1),
        q_temp.stride(2),
        q_temp.stride(3),
        S_q,
        H_q,
        q_chunk_size,
        q_num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
    )

    # ---- Phase 1: Hadamard + max for K ----
    hadamard_single_phase1_kernel[kv_grid](
        k,
        k_had,
        kv_temp,
        k_partial_max,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        kv_temp.stride(0),
        kv_temp.stride(1),
        kv_temp.stride(2),
        kv_temp.stride(3),
        S_kv,
        H_kv,
        kv_chunk_size,
        kv_num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
    )

    # ---- Phase 1: Hadamard + max for V ----
    # kv_temp reused from K: safe because both launches are on the same CUDA
    # stream, so K's kernel fully completes before V's starts.
    hadamard_single_phase1_kernel[kv_grid](
        v,
        v_had,
        kv_temp,
        v_partial_max,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        kv_temp.stride(0),
        kv_temp.stride(1),
        kv_temp.stride(2),
        kv_temp.stride(3),
        S_kv,
        H_kv,
        kv_chunk_size,
        kv_num_chunks,
        D=D,
        LOG2_D=LOG2_D,
        USE_BFLOAT16=use_bfloat16,
    )

    # ---- Reduce ----
    # Q: group reduce across `groups` Q heads per KV head
    group_reduce_kernel[(B, H_kv)](
        q_partial_max, q_scale, q_descale, H_q, H_kv, groups, q_num_chunks
    )
    # K, V: per-head reduce
    single_reduce_kernel[(B, H_kv)](
        k_partial_max, k_scale, k_descale, H_kv, kv_num_chunks
    )
    single_reduce_kernel[(B, H_kv)](
        v_partial_max, v_scale, v_descale, H_kv, kv_num_chunks
    )

    # ---- Phase 2: Quantize Q from Hadamard'd intermediate ----
    single_phase2_kernel[q_grid](
        q_had,
        q_fp8,
        q_scale,
        q_had.stride(0),
        q_had.stride(1),
        q_had.stride(2),
        q_had.stride(3),
        S_q,
        D,
        H_q,
        q_chunk_size,
        H_kv,
        groups,
    )

    # ---- Phase 2: Quantize K ----
    single_phase2_kernel[kv_grid](
        k_had,
        k_fp8,
        k_scale,
        k_had.stride(0),
        k_had.stride(1),
        k_had.stride(2),
        k_had.stride(3),
        S_kv,
        D,
        H_kv,
        kv_chunk_size,
        H_kv,
        1,
    )

    # ---- Phase 2: Quantize V ----
    single_phase2_kernel[kv_grid](
        v_had,
        v_fp8,
        v_scale,
        v_had.stride(0),
        v_had.stride(1),
        v_had.stride(2),
        v_had.stride(3),
        S_kv,
        D,
        H_kv,
        kv_chunk_size,
        H_kv,
        1,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
