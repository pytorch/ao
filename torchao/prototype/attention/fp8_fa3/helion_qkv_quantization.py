# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 Quantization kernels using Helion.

This module provides Helion kernels that perform per-head FP8 quantization
for Q, K, V tensors.

The 3-kernel structure parallelizes over (B, H, S) with nested D loop:
- Phase 1: Compute partial absmax values per S-block
- Reduce: Aggregate maxes per head and compute scale/descale factors
- Phase 2: Apply scales and cast to FP8

Input/output format: [B, H, S, D]
"""

from typing import Optional, Tuple

import torch

import helion
import helion.language as hl


# =============================================================================
# Phase 1: Partial absmax computation
# =============================================================================


@helion.kernel(static_shapes=True)
def qkv_phase1_helion(
    q: torch.Tensor,  # [B, H, S, D]
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    partial_max: torch.Tensor,  # [B, H, num_s_blocks, 3] - output
) -> None:
    """
    Phase 1: Compute partial absmax for Q, K, V per S-block.

    Uses 3D tiling over (B, H, S) with block_size=[1, 1, block_s], plus
    a nested inner loop over D with block_size=D (single iteration).
    """
    B, H, S, D = q.size()

    block_size_s = hl.register_block_size(S)

    for tile_b, tile_h, tile_s in hl.tile([B, H, S], block_size=[1, 1, block_size_s]):
        for tile_d in hl.tile(D, block_size=D):
            q_tile = q[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)
            k_tile = k[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)
            v_tile = v[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)

            q_max_tile = torch.amax(torch.abs(q_tile), dim=-1)
            k_max_tile = torch.amax(torch.abs(k_tile), dim=-1)
            v_max_tile = torch.amax(torch.abs(v_tile), dim=-1)

            q_max = torch.amax(q_max_tile)
            k_max = torch.amax(k_max_tile)
            v_max = torch.amax(v_max_tile)

            partial_max[tile_b.begin, tile_h.begin, tile_s.id, 0] = q_max
            partial_max[tile_b.begin, tile_h.begin, tile_s.id, 1] = k_max
            partial_max[tile_b.begin, tile_h.begin, tile_s.id, 2] = v_max


# =============================================================================
# Reduce kernel: Aggregate partial maxes and compute scales
# =============================================================================


@helion.kernel(static_shapes=True)
def qkv_reduce_helion(
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
    """
    FP8_MAX: float = 448.0
    eps: float = 1e-12
    B, H = q_scale.size()

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
# Phase 2: Quantize to FP8
# =============================================================================


@helion.kernel(static_shapes=True)
def qkv_phase2_helion(
    q: torch.Tensor,  # [B, H, S, D]
    k: torch.Tensor,  # [B, H, S, D]
    v: torch.Tensor,  # [B, H, S, D]
    q_out: torch.Tensor,  # [B, H, S, D] - FP8 output
    k_out: torch.Tensor,  # [B, H, S, D] - FP8 output
    v_out: torch.Tensor,  # [B, H, S, D] - FP8 output
    q_scale: torch.Tensor,  # [B, H] - precomputed scales
    k_scale: torch.Tensor,  # [B, H]
    v_scale: torch.Tensor,  # [B, H]
) -> None:
    """
    Phase 2: Quantize Q, K, V to FP8 using precomputed scales.

    Uses 3D tiling over (B, H, S) with block_size=[1, 1, block_s], plus
    a nested inner loop over D with block_size=D (single iteration).
    """
    B, H, S, D = q.size()

    block_size_s = hl.register_block_size(S)

    for tile_b, tile_h, tile_s in hl.tile([B, H, S], block_size=[1, 1, block_size_s]):
        for tile_d in hl.tile(D, block_size=D):
            q_sc = q_scale[tile_b.begin, tile_h.begin]
            k_sc = k_scale[tile_b.begin, tile_h.begin]
            v_sc = v_scale[tile_b.begin, tile_h.begin]

            q_val = q[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)
            k_val = k[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)
            v_val = v[tile_b.begin, tile_h.begin, tile_s, tile_d].to(torch.float32)

            q_fp8 = (q_val * q_sc).to(torch.float8_e4m3fn)
            k_fp8 = (k_val * k_sc).to(torch.float8_e4m3fn)
            v_fp8 = (v_val * v_sc).to(torch.float8_e4m3fn)

            q_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = q_fp8
            k_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = k_fp8
            v_out[tile_b.begin, tile_h.begin, tile_s, tile_d] = v_fp8


# =============================================================================
# Main entry point
# =============================================================================


def helion_fp8_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
    Fused per-head FP8 quantization for Q, K, V using Helion kernels.

    Uses 3-kernel structure with full parallelization:
    - Phase 1: Partial absmax (parallel over B * H * S_blocks)
    - Reduce: Aggregate maxes per head (parallel over B * H)
    - Phase 2: Quantize (parallel over B * H * S_blocks)

    Note: The num_chunks parameter is ignored. Block sizes are autotuned by Helion.

    Args:
        q: Query tensor of shape [B, H, S, D] in bf16/fp16
        k: Key tensor of shape [B, H, S, D] in bf16/fp16
        v: Value tensor of shape [B, H, S, D] in bf16/fp16
        num_chunks: Ignored (kept for API compatibility)

    Returns:
        q_fp8, k_fp8, v_fp8: Quantized tensors in float8_e4m3fn, shape [B, H, S, D]
        q_descale, k_descale, v_descale: Descale factors of shape [B, H] in fp32
    """
    assert q.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {q.dim()}D"
    assert k.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {k.dim()}D"
    assert v.dim() == 4, f"Expected 4D tensor [B, H, S, D], got {v.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape"

    B, H, S, D = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Upper bound for S blocks (block_size_s is autotuned)
    max_s_blocks = S

    partial_max = torch.zeros(
        B, H, max_s_blocks, 3, dtype=torch.float32, device=q.device
    )

    q_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    k_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    v_scale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    q_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    k_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)
    v_descale = torch.empty(B, H, dtype=torch.float32, device=q.device)

    # Phase 1: partial absmax
    qkv_phase1_helion(q, k, v, partial_max)

    # Reduce: aggregate maxes per head
    qkv_reduce_helion(
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
    qkv_phase2_helion(
        q, k, v,
        q_fp8, k_fp8, v_fp8,
        q_scale, k_scale, v_scale,
    )

    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
