# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Fused RoPE + FP8 Quantization kernels using Helion (Optimized version).

This module provides Helion kernels that fuse:
- RoPE (Rotary Position Embeddings) for Q and K
- FP8 quantization for Q, K, V
- Layout transformation from [B, S, H, D] (FLUX) to [B, H, S, D] (SDPA)

The layout transformation is fused directly into the kernels, avoiding
separate transpose+contiguous memory copies that would otherwise add ~70%
overhead to the overall kernel execution time.

The 3-kernel structure parallelizes over (B, H, S) with nested D loop:
- Phase 1: RoPE + partial max (reads [B,S,H,D], writes [B,H,S,D])
- Reduce: Aggregate maxes per head (parallel over B * H)
- Phase 2: Quantize (reads V from [B,S,H,D], writes [B,H,S,D])

Input format: [B, S, H, D] (FLUX-style)
Output format: [B, H, S, D] (SDPA-style)

"""

from typing import Optional, Tuple

import torch

import helion
import helion.language as hl


# =============================================================================
# Phase 1: RoPE + Max computation
# Reads from [B, S, H, D] (FLUX layout), writes to [B, H, S, D] (SDPA layout)
# Fuses layout transformation with RoPE computation to avoid separate copy
# =============================================================================


@helion.kernel(static_shapes=True)
def rope_qkv_phase1_helion(
    q: torch.Tensor,  # [B, S, H, D] - FLUX input layout
    k: torch.Tensor,  # [B, S, H, D] - FLUX input layout
    v: torch.Tensor,  # [B, S, H, D] - FLUX input layout
    cos: torch.Tensor,  # [S, D]
    sin: torch.Tensor,  # [S, D]
    q_rope_out: torch.Tensor,  # [B, H, S, D] - output in SDPA layout
    k_rope_out: torch.Tensor,  # [B, H, S, D] - output in SDPA layout
    partial_max: torch.Tensor,  # [B, H, num_s_blocks, 3] - output
) -> None:
    """
    Phase 1: Apply RoPE to Q and K, store results, compute partial max.

    Reads from [B, S, H, D] (FLUX layout) and writes to [B, H, S, D] (SDPA layout),
    fusing the layout transformation with the RoPE computation.

    Uses 3D tiling over (B, H, S) with block_size=[1, 1, block_s], plus
    a nested inner loop over D with block_size=D (single iteration).
    """
    B, S, H, D = q.size()
    D_HALF = hl.specialize(D // 2)

    block_size_s = hl.register_block_size(S)

    for tile_b, tile_h, tile_s in hl.tile([B, H, S], block_size=[1, 1, block_size_s]):
        for tile_d in hl.tile(D, block_size=D):
            # Load from [B, S, H, D] input layout
            q_tile = q[tile_b.begin, tile_s, tile_h.begin, tile_d].to(torch.float32)
            k_tile = k[tile_b.begin, tile_s, tile_h.begin, tile_d].to(torch.float32)
            v_tile = v[tile_b.begin, tile_s, tile_h.begin, tile_d].to(torch.float32)

            cos_tile = cos[tile_s, tile_d].to(torch.float32)
            sin_tile = sin[tile_s, tile_d].to(torch.float32)

            # Split into real/imag components
            q_tile_ri = q_tile.reshape(-1, D_HALF, 2)
            k_tile_ri = k_tile.reshape(-1, D_HALF, 2)
            q_real, q_imag = hl.split(q_tile_ri)
            k_real, k_imag = hl.split(k_tile_ri)

            cos_tile_ri = cos_tile.reshape(-1, D_HALF, 2)
            sin_tile_ri = sin_tile.reshape(-1, D_HALF, 2)
            cos_real, cos_imag = hl.split(cos_tile_ri)
            sin_real, sin_imag = hl.split(sin_tile_ri)

            # Apply RoPE
            q_rope_real = q_real * cos_real - q_imag * sin_real
            q_rope_imag = q_real * sin_imag + q_imag * cos_imag
            k_rope_real = k_real * cos_real - k_imag * sin_real
            k_rope_imag = k_real * sin_imag + k_imag * cos_imag

            q_rope = hl.join(q_rope_real, q_rope_imag).reshape(-1, D)
            k_rope = hl.join(k_rope_real, k_rope_imag).reshape(-1, D)

            # Store RoPE'd Q, K to [B, H, S, D] output layout
            q_rope_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = q_rope.to(q.dtype)
            k_rope_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = k_rope.to(k.dtype)

            # Compute partial max for this block
            q_max_tile = torch.amax(torch.abs(q_rope), dim=-1)
            k_max_tile = torch.amax(torch.abs(k_rope), dim=-1)
            v_max_tile = torch.amax(torch.abs(v_tile), dim=-1)

            q_max = torch.amax(q_max_tile)
            k_max = torch.amax(k_max_tile)
            v_max = torch.amax(v_max_tile)

            partial_max[tile_b.begin, tile_h.begin, tile_s.id, 0] = q_max
            partial_max[tile_b.begin, tile_h.begin, tile_s.id, 1] = k_max
            partial_max[tile_b.begin, tile_h.begin, tile_s.id, 2] = v_max


# =============================================================================
# Reduce kernel: Aggregate partial maxes and compute scales
# Parallelizes over (B, H) using hl.tile([...]) with block_size=[1, 1]
# =============================================================================


@helion.kernel(static_shapes=True)
def rope_qkv_reduce_helion(
    partial_max: torch.Tensor,  # [B, H, num_s_blocks, 3]
    q_scale: torch.Tensor,  # [B, H] - output
    k_scale: torch.Tensor,  # [B, H] - output
    v_scale: torch.Tensor,  # [B, H] - output
    q_descale: torch.Tensor,  # [B, H] - output
    k_descale: torch.Tensor,  # [B, H] - output
    v_descale: torch.Tensor,  # [B, H] - output
) -> None:
    """
    Reduce partial maxes across S-blocks and compute scales/descales.

    Uses 2D tiling over (B, H) with block_size=[1, 1].
    - tile_b.begin, tile_h.begin are scalar indices
    - Sequential reduction over S blocks using vectorized access
    """
    FP8_MAX: float = 448.0
    eps: float = 1e-12
    B, H = q_scale.size()
    num_s_blocks = partial_max.size(2)

    for tile_b, tile_h in hl.tile([B, H], block_size=[1, 1]):
        q_partial = partial_max[tile_b.begin, tile_h.begin, :, 0]
        k_partial = partial_max[tile_b.begin, tile_h.begin, :, 1]
        v_partial = partial_max[tile_b.begin, tile_h.begin, :, 2]

        q_max = torch.amax(q_partial)
        k_max = torch.amax(k_partial)
        v_max = torch.amax(v_partial)

        q_scale_val = torch.where(q_max > eps, FP8_MAX / q_max, torch.ones_like(q_max))
        k_scale_val = torch.where(k_max > eps, FP8_MAX / k_max, torch.ones_like(k_max))
        v_scale_val = torch.where(v_max > eps, FP8_MAX / v_max, torch.ones_like(v_max))

        q_scale[tile_b.begin, tile_h.begin] = q_scale_val
        k_scale[tile_b.begin, tile_h.begin] = k_scale_val
        v_scale[tile_b.begin, tile_h.begin] = v_scale_val

        q_descale[tile_b.begin, tile_h.begin] = torch.where(
            q_max > eps, q_max / FP8_MAX, torch.ones_like(q_max)
        )
        k_descale[tile_b.begin, tile_h.begin] = torch.where(
            k_max > eps, k_max / FP8_MAX, torch.ones_like(k_max)
        )
        v_descale[tile_b.begin, tile_h.begin] = torch.where(
            v_max > eps, v_max / FP8_MAX, torch.ones_like(v_max)
        )


# =============================================================================
# Phase 2: Quantize
# Reads V from [B, S, H, D] (FLUX layout), writes to [B, H, S, D] (SDPA layout)
# Fuses layout transformation with quantization to avoid separate copy
# =============================================================================


@helion.kernel(static_shapes=True)
def rope_qkv_phase2_helion(
    q_rope: torch.Tensor,  # [B, H, S, D] - intermediate RoPE'd Q (SDPA layout)
    k_rope: torch.Tensor,  # [B, H, S, D] - intermediate RoPE'd K (SDPA layout)
    v: torch.Tensor,  # [B, S, H, D] - original V (FLUX layout)
    q_out: torch.Tensor,  # [B, H, S, D] - FP8 output
    k_out: torch.Tensor,  # [B, H, S, D] - FP8 output
    v_out: torch.Tensor,  # [B, H, S, D] - FP8 output
    q_scale: torch.Tensor,  # [B, H] - precomputed scales
    k_scale: torch.Tensor,  # [B, H]
    v_scale: torch.Tensor,  # [B, H]
) -> None:
    """
    Phase 2: Quantize pre-computed RoPE'd Q, K and V.

    Q and K are read from intermediate buffers in [B, H, S, D] (SDPA layout).
    V is read from original input in [B, S, H, D] (FLUX layout).
    All outputs are written in [B, H, S, D] (SDPA layout).

    Uses 3D tiling over (B, H, S) with block_size=[1, 1, block_s], plus
    a nested inner loop over D with block_size=D (single iteration).
    """
    B, H, S, D = q_rope.size()

    block_size_s = hl.register_block_size(S)

    for tile_b, tile_h, tile_s in hl.tile([B, H, S], block_size=[1, 1, block_size_s]):
        for tile_d in hl.tile(D, block_size=D):
            q_sc = q_scale[tile_b.begin, tile_h.begin]
            k_sc = k_scale[tile_b.begin, tile_h.begin]
            v_sc = v_scale[tile_b.begin, tile_h.begin]

            # Load Q, K from [B, H, S, D] intermediate buffers
            q_val = q_rope[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)
            k_val = k_rope[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)

            # Load V from [B, S, H, D] input layout
            v_val = v[tile_b.begin, tile_s, tile_h.begin, tile_d].to(torch.float32)

            # Quantize to FP8
            q_fp8 = (q_val * q_sc).to(torch.float8_e4m3fn)
            k_fp8 = (k_val * k_sc).to(torch.float8_e4m3fn)
            v_fp8 = (v_val * v_sc).to(torch.float8_e4m3fn)

            q_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = q_fp8
            k_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = k_fp8
            v_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = v_fp8


# =============================================================================
# Main entry point (same API as Triton version)
# =============================================================================


def fp8_rope_quantize_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_chunks: Optional[int] = None,  # Ignored - block sizes are autotuned
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

    The layout transformation is fused into the kernels themselves, avoiding
    the need for separate transpose+contiguous memory copies.

    Uses 3-kernel structure with full parallelization:
    - Phase 1: RoPE + partial max (parallel over B * H * S_blocks)
    - Reduce: Aggregate maxes per head (parallel over B * H)
    - Phase 2: Quantize (parallel over B * H * S_blocks)

    Note: The num_chunks parameter is ignored. Block sizes are autotuned by Helion.

    Args:
        q: Query tensor of shape [B, S, H, D] in bf16/fp16
        k: Key tensor of shape [B, S, H, D] in bf16/fp16
        v: Value tensor of shape [B, S, H, D] in bf16/fp16
        cos: Cosine frequencies for RoPE, shape [S, D]
        sin: Sine frequencies for RoPE, shape [S, D]
        num_chunks: Ignored (kept for API compatibility)

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

    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    # Upper bound for S blocks (block_size_s is autotuned)
    max_s_blocks = S

    # Allocate intermediate buffers
    q_rope_intermediate = torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)
    k_rope_intermediate = torch.empty(B, H, S, D, dtype=k.dtype, device=q.device)

    partial_max = torch.zeros(
        B, H, max_s_blocks, 3, dtype=torch.float32, device=q.device
    )

    q_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)

    # Phase 1: RoPE + partial max
    rope_qkv_phase1_helion(
        q,
        k,
        v,
        cos,
        sin,
        q_rope_intermediate,
        k_rope_intermediate,
        partial_max,
    )

    # Reduce: aggregate maxes per head
    rope_qkv_reduce_helion(
        partial_max,
        q_scale,
        k_scale,
        v_scale,
        q_descale,
        k_descale,
        v_descale,
    )

    # Allocate FP8 output buffers
    q_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    k_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)
    v_fp8 = torch.empty(B, H, S, D, dtype=torch.float8_e4m3fn, device=q.device)

    # Phase 2: quantize
    rope_qkv_phase2_helion(
        q_rope_intermediate,
        k_rope_intermediate,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
