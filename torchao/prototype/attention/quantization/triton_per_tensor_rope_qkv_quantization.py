# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused RoPE + per-tensor FP8 quantization kernels for Q, K, V.

Input: [B, S, H, D], output: [B, H, S, D].
Produces a single scalar descale per tensor (shape [1, 1, 1, 1]) for cuDNN.
Supports GQA (different head counts for Q vs K/V).

Uses atomic max reduction in Phase 1, eliminating the separate reduce
kernel and partial-max buffer needed by per-head quantization.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from torchao.prototype.attention.quantization.triton_hadamard_utils import (
    _compute_num_chunks,
)
from torchao.prototype.attention.quantization.triton_per_tensor_qkv_quantization import (
    compute_scale_descale_kernel,
    per_tensor_phase1_kernel,
)
from torchao.prototype.attention.quantization.triton_rope_qkv_quantization import (
    rope_single_phase2_kernel,
    v_phase2_kernel,
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["D_HALF"],
)
@triton.jit
def rope_per_tensor_phase1_kernel(
    # Input tensor [B, S, H, D]
    x_ptr,
    # RoPE frequency tensors [S, D]
    cos_ptr,
    sin_ptr,
    # Intermediate output tensor [B, H, S, D] - stores RoPE'd result
    x_rope_ptr,
    # Output: single global max [1], atomically updated
    global_max_ptr,
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
    B,
    H,
    S,
    D,
    D_HALF,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    ROPE_INTERLEAVED: tl.constexpr,
):
    """
    Phase 1 for Q/K: Apply RoPE, store to intermediate, compute absmax
    with atomic reduction.

    Grid: (num_blocks,) — flat grid tuned for GPU occupancy.
    Each block processes a strided portion of B*H*S*D_HALF RoPE pairs.
    """
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    total_pairs = B * H * S * D_HALF

    x_max = 0.0

    for idx_start in range(pid * BLOCK_SIZE, total_pairs, num_blocks * BLOCK_SIZE):
        offs = idx_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_pairs

        # Convert flat index to (b, h, s, pair_idx) coordinates
        pair_idx = offs % D_HALF
        remaining = offs // D_HALF
        s_idx = remaining % S
        remaining = remaining // S
        h_idx = remaining % H
        b_idx = remaining // H

        # Compute element indices for this pair based on RoPE variant
        if ROPE_INTERLEAVED:
            # FLUX/GPT-J interleaved: pair (2i, 2i+1)
            d_first = pair_idx * 2
            d_second = pair_idx * 2 + 1
        else:
            # NeoX/LLaMA half-split: pair (j, j+D/2)
            d_first = pair_idx
            d_second = pair_idx + D_HALF

        # Input offsets [B, S, H, D]
        in_offset_first = (
            b_idx * stride_in_b
            + s_idx * stride_in_s
            + h_idx * stride_in_h
            + d_first * stride_in_d
        )
        in_offset_second = (
            b_idx * stride_in_b
            + s_idx * stride_in_s
            + h_idx * stride_in_h
            + d_second * stride_in_d
        )

        # Output offsets [B, H, S, D]
        out_offset_first = (
            b_idx * stride_out_b
            + h_idx * stride_out_h
            + s_idx * stride_out_s
            + d_first * stride_out_d
        )
        out_offset_second = (
            b_idx * stride_out_b
            + h_idx * stride_out_h
            + s_idx * stride_out_s
            + d_second * stride_out_d
        )

        # Load input pairs
        x_first = tl.load(x_ptr + in_offset_first, mask=mask, other=0.0).to(tl.float32)
        x_second = tl.load(x_ptr + in_offset_second, mask=mask, other=0.0).to(
            tl.float32
        )

        # Load cos/sin
        cos_offset = s_idx * D + d_first
        cos_val = tl.load(cos_ptr + cos_offset, mask=mask, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_ptr + cos_offset, mask=mask, other=0.0).to(tl.float32)

        # Apply RoPE rotation
        x_rope_first = tl.math.fma(x_first, cos_val, -(x_second * sin_val))
        x_rope_second = tl.math.fma(x_second, cos_val, x_first * sin_val)

        # Store RoPE'd result to intermediate buffer [B, H, S, D]
        tl.store(
            x_rope_ptr + out_offset_first, x_rope_first.to(x_first.dtype), mask=mask
        )
        tl.store(
            x_rope_ptr + out_offset_second, x_rope_second.to(x_first.dtype), mask=mask
        )

        # Update max
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_rope_first)))
        x_max = tl.maximum(x_max, tl.max(tl.abs(x_rope_second)))

    # Single atomic update per block after local reduction
    tl.atomic_max(global_max_ptr, x_max)


def triton_fp8_per_tensor_rope_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_chunks: Optional[int] = None,
    rope_interleaved: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Fused RoPE + per-tensor FP8 quantization for Q, K, V tensors.

    Applies RoPE to Q and K, then quantizes all tensors to FP8 with
    per-tensor scaling. Also performs layout transformation from
    [B, S, H, D] to [B, H, S, D].

    Args:
        q: Query tensor of shape [B, S, H_q, D] in bf16/fp16
        k: Key tensor of shape [B, S, H_kv, D] in bf16/fp16
        v: Value tensor of shape [B, S, H_kv, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Number of chunks to split S dimension into (for Phase 2).
        rope_interleaved: If True, use interleaved RoPE pairing (2i, 2i+1).

    Returns:
        q_fp8: Quantized query with RoPE, shape [B, H_q, S, D] in fp8
        k_fp8: Quantized key with RoPE, shape [B, H_kv, S, D] in fp8
        v_fp8: Quantized value, shape [B, H_kv, S, D] in fp8
        q_descale: Query descale factor, shape [1, 1, 1, 1] in fp32
        k_descale: Key descale factor, shape [1, 1, 1, 1] in fp32
        v_descale: Value descale factor, shape [1, 1, 1, 1] in fp32
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert k.shape == v.shape
    assert q.shape[0] == k.shape[0]
    assert q.shape[1] == k.shape[1]
    assert q.shape[3] == k.shape[3]
    assert q.shape[2] % k.shape[2] == 0
    assert cos.dim() == 2 and sin.dim() == 2

    B, S, H_q, D = q.shape
    H_kv = k.shape[2]

    assert D % 2 == 0
    assert cos.shape == (S, D) and sin.shape == (S, D)

    D_HALF = D // 2

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Flat grid for Phase 1: num_blocks tuned to GPU occupancy
    props = torch.cuda.get_device_properties(q.device)
    num_blocks = props.multi_processor_count * 4

    # Chunk computation for Phase 2 (still uses (B, H, chunks) grid)
    if num_chunks is None:
        num_chunks = _compute_num_chunks(q.device, B, H_q, S)
    chunk_size = (S + num_chunks - 1) // num_chunks

    # Output in [B, H, S, D] layout
    q_fp8 = torch.empty(B, H_q, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(B, H_kv, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    v_fp8 = torch.empty(B, H_kv, S, D, dtype=torch.float8_e4m3fn, device=q.device)

    # Intermediate buffers for RoPE'd Q, K in [B, H, S, D]
    q_rope = torch.empty(B, H_q, S, D, dtype=q.dtype, device=q.device)
    k_rope = torch.empty(B, H_kv, S, D, dtype=k.dtype, device=q.device)

    # Global max buffers: zero-initialized for atomic_max
    q_global_max = torch.zeros(1, dtype=torch.float32, device=q.device)
    k_global_max = torch.zeros(1, dtype=torch.float32, device=q.device)
    v_global_max = torch.zeros(1, dtype=torch.float32, device=q.device)

    # One scale/descale per tensor
    q_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(1, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(1, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(1, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(1, dtype=torch.float32, device=q.device)

    # Phase 1: RoPE + absmax for Q (flat grid, atomic reduction)
    rope_per_tensor_phase1_kernel[(num_blocks,)](
        q,
        cos,
        sin,
        q_rope,
        q_global_max,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q_rope.stride(0),
        q_rope.stride(1),
        q_rope.stride(2),
        q_rope.stride(3),
        B,
        H_q,
        S,
        D,
        D_HALF,
        ROPE_INTERLEAVED=rope_interleaved,
    )

    # Phase 1: RoPE + absmax for K (flat grid, atomic reduction)
    rope_per_tensor_phase1_kernel[(num_blocks,)](
        k,
        cos,
        sin,
        k_rope,
        k_global_max,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k_rope.stride(0),
        k_rope.stride(1),
        k_rope.stride(2),
        k_rope.stride(3),
        B,
        H_kv,
        S,
        D,
        D_HALF,
        ROPE_INTERLEAVED=rope_interleaved,
    )

    # Phase 1: absmax for V (flat grid, atomic reduction, no RoPE)
    # Reuse per_tensor_phase1_kernel with [B,S,H,D] strides remapped
    per_tensor_phase1_kernel[(num_blocks,)](
        v,
        v_global_max,
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v.stride(3),  # remap: B,H,S,D <- B,S,H,D
        B,
        H_kv,
        S,
        D,
    )

    # Compute scale/descale from global max
    compute_scale_descale_kernel[(1,)](q_global_max, q_scale, q_descale)
    compute_scale_descale_kernel[(1,)](k_global_max, k_scale, k_descale)
    compute_scale_descale_kernel[(1,)](v_global_max, v_scale, v_descale)

    q_grid = (B, H_q, num_chunks)
    kv_grid = (B, H_kv, num_chunks)

    # Phase 2: quantize Q from RoPE'd intermediate
    # H_scale=0 so all blocks read scale_ptr[0]
    rope_single_phase2_kernel[q_grid](
        q_rope,
        q_fp8,
        q_scale,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        S,
        D,
        H_q,
        chunk_size,
        0,
        H_q,  # H_scale=0, groups=H_q: index = pid_b*0 + pid_h//H_q = 0
    )

    # Phase 2: quantize K from RoPE'd intermediate
    rope_single_phase2_kernel[kv_grid](
        k_rope,
        k_fp8,
        k_scale,
        k_fp8.stride(0),
        k_fp8.stride(1),
        k_fp8.stride(2),
        k_fp8.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
        0,
        H_kv,  # H_scale=0, groups=H_kv: index = 0
    )

    # Phase 2: transpose + quantize V
    # v_phase2_kernel hardcodes scale indexing as scale_ptr[pid_b * H + pid_h],
    # so we broadcast the scalar to [B, H_kv]
    v_scale_bh = v_scale.expand(B, H_kv).contiguous()
    v_phase2_kernel[kv_grid](
        v,
        v_fp8,
        v_scale_bh,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        S,
        D,
        H_kv,
        chunk_size,
    )

    # Reshape descales to [1, 1, 1, 1] for cuDNN per-tensor format
    q_descale = q_descale.view(1, 1, 1, 1)
    k_descale = k_descale.view(1, 1, 1, 1)
    v_descale = v_descale.view(1, 1, 1, 1)

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
