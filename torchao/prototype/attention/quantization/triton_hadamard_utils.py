# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared Hadamard transform utilities for FP8 quantization kernels.

Provides the Hadamard butterfly helper used by both the RoPE-fused and
plain Hadamard quantization kernels, plus the inverse Hadamard transform
applied to attention output.

The Hadamard transform H/sqrt(D) is orthogonal and self-inverse:
    H/sqrt(D) @ H/sqrt(D) = I
so the same butterfly + normalization is used for both forward and inverse.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


def _get_log2_d(D: int) -> int:
    """Get log2(D), asserting D is a power of 2."""
    assert D > 0 and (D & (D - 1)) == 0, f"D must be a power of 2, got {D}"
    log2_d = 0
    temp = D
    while temp > 1:
        temp >>= 1
        log2_d += 1
    return log2_d


def _compute_num_chunks(device: torch.device, B: int, H: int, S: int) -> int:
    """Compute optimal number of chunks for parallelizing over the S dimension.

    Layout-agnostic: callers extract B and H from whichever tensor layout
    they use ([B, H, S, D] or [B, S, H, D]) and pass the scalars directly.
    """
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    base_parallelism = B * H
    target_blocks = num_sms * 4
    num_chunks = max(1, target_blocks // base_parallelism)
    num_chunks = min(num_chunks, S // 32) if S >= 32 else 1
    num_chunks = min(num_chunks, 64)
    num_chunks = min(num_chunks, S)
    return num_chunks


@triton.jit
def _hadamard_butterfly_stage(
    x,
    temp_ptr,
    temp_base,
    d_idx,
    stage: tl.constexpr,
    D: tl.constexpr,
):
    """One stage of the Hadamard butterfly transform.

    Uses global memory temp buffer as shuffle buffer with barriers.
    Each thread stores its value, barrier, loads its partner's value,
    barrier, then computes the butterfly sum/difference.

    Args:
        x: Current D-element vector (vectorized across threads)
        temp_ptr: Pointer to temp buffer base
        temp_base: Offset to this block's region in temp buffer
        d_idx: Vectorized index tensor (tl.arange(0, D))
        stage: Butterfly stage (0 to log2(D)-1), must be constexpr
        D: Head dimension (compile-time constant)
    """
    stride = 1 << stage
    partner_d = d_idx ^ stride
    is_left = (d_idx & stride) == 0

    tl.store(temp_ptr + temp_base + d_idx, x)
    tl.debug_barrier()
    x_partner = tl.load(temp_ptr + temp_base + partner_d)
    tl.debug_barrier()

    return tl.where(is_left, x + x_partner, x_partner - x)


@triton.jit
def _apply_hadamard(
    x,
    temp_ptr,
    temp_base,
    d_idx,
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """Apply full Hadamard butterfly transform with 1/sqrt(D) normalization.

    Uses tl.static_range so each stage index is a compile-time constant.
    Supports D up to 256 (LOG2_D up to 8).
    """
    for stage in tl.static_range(LOG2_D):
        x = _hadamard_butterfly_stage(x, temp_ptr, temp_base, d_idx, stage, D)
    inv_sqrt_d = 1.0 / tl.sqrt(float(D))
    return x * inv_sqrt_d


# =============================================================================
# Inverse Hadamard transform kernel
# Applied to attention output to recover correct results after V was transformed
# =============================================================================


@triton.jit
def _inverse_hadamard_kernel(
    # Input tensor [B, H, S, D]
    input_ptr,
    # Output tensor [B, H, S, D]
    output_ptr,
    # Temp buffer for Hadamard [B, H, num_chunks, D]
    temp_ptr,
    # Input strides [B, H, S, D] (may be non-contiguous)
    stride_in_b,
    stride_in_h,
    stride_in_s,
    stride_in_d,
    # Output strides [B, H, S, D] (contiguous)
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
    """Apply inverse Hadamard transform along D dimension.

    Grid: (B, H, num_chunks)
    Block: D threads, each handles one d index across S positions in chunk.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_chunk = tl.program_id(axis=2)

    d_idx = tl.arange(0, D)
    temp_base = (
        pid_b * stride_temp_b + pid_h * stride_temp_h + pid_chunk * stride_temp_c
    )
    s_start = pid_chunk * chunk_size

    in_base = pid_b * stride_in_b + pid_h * stride_in_h
    out_base = pid_b * stride_out_b + pid_h * stride_out_h

    for s_offset in range(chunk_size):
        s_idx = s_start + s_offset
        s_mask = s_idx < S

        in_offset = in_base + s_idx * stride_in_s + d_idx * stride_in_d
        out_offset = out_base + s_idx * stride_out_s + d_idx * stride_out_d

        x = tl.load(input_ptr + in_offset, mask=s_mask, other=0.0).to(tl.float32)
        x = _apply_hadamard(x, temp_ptr, temp_base, d_idx, D, LOG2_D)

        if USE_BFLOAT16:
            tl.store(output_ptr + out_offset, x.to(tl.bfloat16), mask=s_mask)
        else:
            tl.store(output_ptr + out_offset, x.to(tl.float16), mask=s_mask)


def inverse_hadamard_transform(
    x: torch.Tensor,
    num_chunks: Optional[int] = None,
) -> torch.Tensor:
    """Apply inverse Hadamard transform along the last dimension.

    Input shape: [B, H, S, D] where D must be a power of 2 and <= 256.
    Output: same shape and dtype, always contiguous.

    The Hadamard transform is self-inverse up to normalization, so this
    applies the same butterfly + 1/sqrt(D) as the forward transform.
    """
    assert x.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {x.dim()}D"

    B, H, S, D = x.shape
    LOG2_D = _get_log2_d(D)
    assert D <= 256, f"D must be <= 256 for Hadamard transform, got {D}"
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"Expected bf16 or fp16, got {x.dtype}"
    )

    if num_chunks is None:
        num_chunks = _compute_num_chunks(x.device, B, H, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    output = torch.empty(B, H, S, D, dtype=x.dtype, device=x.device)
    temp_buffer = torch.empty(B, H, num_chunks, D, dtype=torch.float32, device=x.device)

    grid = (B, H, num_chunks)
    use_bfloat16 = x.dtype == torch.bfloat16

    _inverse_hadamard_kernel[grid](
        x,
        output,
        temp_buffer,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
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
        num_warps=4,
    )

    return output
