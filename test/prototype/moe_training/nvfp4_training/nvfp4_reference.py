# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Plain-PyTorch NVFP4 reference, transcribed from TransformerEngine.

The Triton and CuteDSL kernels in this package are a port of TransformerEngine's NVFP4 +
Randomized Hadamard Transform kernels, so the oracle they are tested against has to be
TE's arithmetic rather than a merely equivalent formulation. The scale chain here follows
``compute_global_encode_scaling_factor_FP4`` and ``compute_decoding_scaling_factor`` in
``transformer_engine/common/cast/nvfp4/core_nvfp4.cuh`` operation for operation:

    S_enc  = min(448 * 6 / amax, FLT_MAX);  if amax == 0 or S_enc == 0: S_enc = 1
    S_dec  = e4m3( min(block_amax * (S_enc * (1/6)), 448) )
    encode = min(1 / (f32(S_dec) * (1 / S_enc)), FLT_MAX)

Two details are load-bearing and easy to "simplify" wrongly:

* ``block_amax * (S_enc * (1/6))`` rounds once. ``(block_amax / 6) * S_enc`` rounds twice
  and disagrees on a fraction of blocks.
* There is no lower clamp on the block scale. TE emits a literal zero E4M3 scale for a
  zero block and for one that underflows E4M3, and so do the kernels; an ``E4M3_EPS``
  floor (as in ``mx_formats.nvfp4_quantize``) is a different recipe.

This module deliberately imports nothing that needs Triton, CuteDSL or TransformerEngine,
so it is importable on CPU and usable from out-of-tree comparison scripts.

**FP4 encode.** ``mx_formats.kernels.f32_to_f4_unpacked`` reproduces the hardware
``cvt.rn.satfinite.e2m1x2.f32`` exactly, so it is reused rather than reimplemented. The
property it is being relied on for: E2M1 magnitudes are ``[0, .5, 1, 1.5, 2, 3, 4, 6]`` at
magnitude codes 0..7, so the mantissa LSB is zero for codes {0,2,4,6}; round-to-nearest-even
therefore resolves every midpoint to the neighbour with the *even* code, which alternates
down/up along the grid:

    0.25 -> 0    0.75 -> 1.0    1.25 -> 1.0    1.75 -> 2.0
    2.5  -> 2.0  3.5  -> 4.0    5.0  -> 4.0    |x| > 6 -> 6

``pack_uint4`` puts the even element in the low nibble, which is the kernels' order.

**What can be asserted.** Scales, amaxes, and RTNE FP4 codes are bitwise: the kernels use
correctly rounded FP32 division, matching both this PyTorch transcription and
TransformerEngine's default numeric path.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
    get_rht_matrix,
)
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
from torchao.prototype.mx_formats.utils import to_blocked, to_blocked_grouped

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
_FP32_MAX = torch.finfo(torch.float32).max

__all__ = [
    "NVFP4ReferenceOutput",
    "global_encode_scale",
    "nvfp4_reference_quantize",
    "reference_group_rht_amax",
    "reference_group_rht_quantize_row_col",
    "reference_group_weight_quantize_2d",
    "reference_rht",
    "reference_rht_amax",
    "reference_rht_quantize_row_col",
    "reference_weight_quantize_2d",
]


# ---------------------------------------------------------------------------
# Core arithmetic (core_nvfp4.cuh)
# ---------------------------------------------------------------------------


def global_encode_scale(global_amax: torch.Tensor) -> torch.Tensor:
    """``compute_global_encode_scaling_factor_FP4``: 2688 / amax, guarded at both ends."""
    amax = global_amax.to(torch.float32)
    candidate = torch.full_like(amax, FP8_E4M3_MAX * FP4_E2M1_MAX) / amax
    candidate = candidate.clamp(max=_FP32_MAX)
    # amax == 0 gives inf; an enormous amax underflows the scale to zero. Both -> identity.
    return torch.where(
        (amax == 0.0) | (candidate == 0.0), torch.ones_like(candidate), candidate
    )


def _block_scale(block_amax: torch.Tensor, s_enc: torch.Tensor) -> torch.Tensor:
    """``compute_decoding_scaling_factor``: one rounding, upper clamp only."""
    scale = block_amax * (s_enc * (1.0 / FP4_E2M1_MAX))
    return scale.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def _encode_scale(block_scale_fp8: torch.Tensor, s_enc: torch.Tensor) -> torch.Tensor:
    """Correctly rounded reciprocal of the effective decode scale."""
    denom = block_scale_fp8.to(torch.float32) * (1.0 / s_enc)
    return (1.0 / denom).clamp(max=_FP32_MAX)


def pack_fp4(scaled: torch.Tensor) -> torch.Tensor:
    """(R, C) f32 -> (R, C//2) uint8, low nibble = even element."""
    clamped = scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    return pack_uint4(f32_to_f4_unpacked(clamped.contiguous()))


# ---------------------------------------------------------------------------
# The quantize primitive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVFP4ReferenceOutput:
    """Everything a caller might assert on, including the intermediates.

    Intermediates remain exposed for diagnostics of the TE scale chain.
    """

    codes: torch.Tensor  # (R, C//2) uint8
    scales: torch.Tensor  # float8_e4m3fn, plain (R, C//16) or swizzled
    block_scale: torch.Tensor  # (R//bm, C//16) float8_e4m3fn, pre-expansion
    encode_scale: torch.Tensor  # (R//bm, C//16) float32
    values: torch.Tensor  # (R, C) float32, the quantizer's input
    scaled: torch.Tensor  # (R, C) float32, values * encode_scale, pre-round
    block_rows: int  # 1 or 16, the tile height the scales were reduced over


def _block_amax(x: torch.Tensor, block_rows: int) -> torch.Tensor:
    """Per-tile amax over ``(block_rows, 16)`` tiles -> (R//block_rows, C//16) f32."""
    rows, cols = x.shape
    tiles = x.abs().reshape(rows // block_rows, block_rows, cols // 16, 16)
    return tiles.amax(dim=(1, 3))


def nvfp4_reference_quantize(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    block: str = "1x16",
    layout: str = "plain",
) -> NVFP4ReferenceOutput:
    """NVFP4 quantize a 2-D tensor with TE's arithmetic.

    Args:
        x: (R, C) bfloat16 or float32. Upcast to float32, which is lossless from bf16 and
            is what the kernels do before the block reduction.
        global_amax: scalar float32 tensor-wide amax.
        block: ``"1x16"`` (activations) or ``"16x16"`` (2D weight scaling).
        layout: ``"plain"`` returns (R, C//16) scales; ``"swizzled"`` returns the
            ``to_blocked`` byte sequence the kernels emit.
    """
    if block not in ("1x16", "16x16"):
        raise ValueError(f"block must be '1x16' or '16x16', got {block!r}")
    if layout not in ("plain", "swizzled"):
        raise ValueError(f"layout must be 'plain' or 'swizzled', got {layout!r}")
    block_rows = 1 if block == "1x16" else 16

    xf = x.float()
    s_enc = global_encode_scale(global_amax)
    block_scale = _block_scale(_block_amax(xf, block_rows), s_enc)
    enc = _encode_scale(block_scale, s_enc)

    # Broadcast the per-tile scale back over its elements.
    expanded = enc.repeat_interleave(block_rows, dim=0).repeat_interleave(16, dim=1)
    scaled = xf * expanded

    scales_plain = block_scale.repeat_interleave(block_rows, dim=0)
    scales = to_blocked(scales_plain) if layout == "swizzled" else scales_plain
    return NVFP4ReferenceOutput(
        codes=pack_fp4(scaled),
        scales=scales,
        block_scale=block_scale,
        encode_scale=enc,
        values=xf,
        scaled=scaled,
        block_rows=block_rows,
    )


# ---------------------------------------------------------------------------
# Per-op wrappers. Each returns exactly the tuple its kernel returns -- the linear
# RHT op is column-first, the grouped one row-first, and weight_quantize_2d reorders.
# ---------------------------------------------------------------------------


def reference_rht(
    A: torch.Tensor, sign_vector: Sequence[int] = DEFAULT_SIGN_VECTOR
) -> torch.Tensor:
    """``RHT(A.t())``: (M, N) -> (N, M) bfloat16.

    The bf16 downcast is not incidental -- TE's RHT output tensor is bf16, and both
    kernels round their fp32 Hadamard accumulator to bf16 before consuming it.
    """
    m, n = A.shape
    B = get_rht_matrix(tuple(sign_vector), A.device, torch.bfloat16, 16)
    return (A.t().reshape(-1, 16) @ B).reshape(n, m).to(torch.bfloat16)


def reference_rht_amax(
    A: torch.Tensor, sign_vector: Sequence[int] = DEFAULT_SIGN_VECTOR
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(col_amax, row_amax)`` = ``(max|RHT(A.t())|, max|A|)``, both scalar f32."""
    return (
        reference_rht(A, sign_vector).float().abs().max(),
        A.float().abs().max(),
    )


def reference_rht_quantize_row_col(
    A: torch.Tensor,
    col_global_amax: torch.Tensor,
    row_global_amax: torch.Tensor,
    sign_vector: Sequence[int] = DEFAULT_SIGN_VECTOR,
    *,
    layout: str = "swizzled",
) -> Tuple[
    NVFP4ReferenceOutput,
    NVFP4ReferenceOutput,
]:
    """``(col, row)`` references for ``*_rht_quantize_row_col``.

    Columnwise quantizes ``RHT(A.t())``, rowwise quantizes raw ``A``; both 1x16. Returns
    the full reference objects rather than a flat tuple for code and scale assertions.
    """
    col = nvfp4_reference_quantize(
        reference_rht(A, sign_vector), col_global_amax, block="1x16", layout=layout
    )
    row = nvfp4_reference_quantize(A, row_global_amax, block="1x16", layout=layout)
    return col, row


def reference_weight_quantize_2d(
    W: torch.Tensor, global_amax: torch.Tensor, *, layout: str = "swizzled"
) -> Tuple[NVFP4ReferenceOutput, NVFP4ReferenceOutput]:
    """``(rowwise, colwise)`` references for ``*_weight_quantize_2d`` (16x16, no RHT).

    Colwise is the same recipe applied to ``W.t()``; ``max|W.t()| == max|W|``, so both
    take the same global amax.
    """
    rowwise = nvfp4_reference_quantize(W, global_amax, block="16x16", layout=layout)
    colwise = nvfp4_reference_quantize(
        W.t().contiguous(), global_amax, block="16x16", layout=layout
    )
    return rowwise, colwise


# ---------------------------------------------------------------------------
# Grouped wrappers
# ---------------------------------------------------------------------------


def _group_sizes(offsets: torch.Tensor, num_tensors: int) -> list:
    """Cumulative row-end offsets -> per-group row counts."""
    ends = offsets[:num_tensors].tolist()
    starts = [0] + ends[:-1]
    return [e - s for s, e in zip(starts, ends)]


def reference_group_rht_amax(
    A: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    sign_vector: Sequence[int] = DEFAULT_SIGN_VECTOR,
    *,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``(col_amax, row_amax)``, each ``(num_tensors,)`` float32.

    Rows at or past ``logical_packed_length`` are capacity padding and must not reach the
    reduction -- the kernels mask them out.
    """
    valid = A.shape[0] if logical_packed_length is None else int(logical_packed_length)
    cols, rows = [], []
    start = 0
    for size in _group_sizes(offsets, num_tensors):
        end = min(start + size, valid)
        group = A[start:end]
        cols.append(reference_rht(group, sign_vector).float().abs().max())
        rows.append(group.float().abs().max())
        start += size
    return torch.stack(cols), torch.stack(rows)


def reference_group_rht_quantize_row_col(
    A: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    col_global_amax: torch.Tensor,
    row_global_amax: torch.Tensor,
    sign_vector: Sequence[int] = DEFAULT_SIGN_VECTOR,
    *,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(row_codes, row_sf, col_codes, col_sf)`` for ``*_group_rht_quantize_row_col``.

    Both scale buffers are returned in the kernels' logical 2-D shapes holding swizzled
    bytes. The two axes need different swizzles: rowwise has the grouped axis on the outer
    (128-blocked) side, where a group is already contiguous, so a whole-extent
    ``to_blocked`` is correct. Columnwise has it on the inner (64-blocked) side, so each
    group is blocked on its own extent and the buffers concatenated -- a whole-extent
    ``to_blocked`` would scatter every group's tiles to the wrong offsets.
    """
    psl, hidden = A.shape
    valid = psl if logical_packed_length is None else int(logical_packed_length)
    sizes = _group_sizes(offsets, num_tensors)
    if any(size % 128 for size in sizes):
        raise ValueError(f"group row counts must be 128-aligned, got {sizes}")

    row_codes = A.new_zeros((psl, hidden // 2), dtype=torch.uint8)
    row_sf_plain = A.new_zeros((psl, hidden // 16), dtype=torch.float8_e4m3fn)
    col_codes = A.new_zeros((hidden, psl // 2), dtype=torch.uint8)
    col_sf_blocks = []

    start = 0
    for g, size in enumerate(sizes):
        end = min(start + size, valid)
        # Capacity rows keep the zeros allocated above, matching the kernels' zero-fill
        # of the codes and of the columnwise scale tiles.
        block = A.new_zeros((hidden, size // 16), dtype=torch.float8_e4m3fn)
        if end > start:
            group = A[start:end]
            row = nvfp4_reference_quantize(
                group, row_global_amax[g], block="1x16", layout="plain"
            )
            row_codes[start:end] = row.codes
            row_sf_plain[start:end] = row.scales
            col = nvfp4_reference_quantize(
                reference_rht(group, sign_vector),
                col_global_amax[g],
                block="1x16",
                layout="plain",
            )
            col_codes[:, start // 2 : end // 2] = col.codes
            block[:, : (end - start) // 16] = col.scales
        col_sf_blocks.append(block)
        start += size

    row_sf = to_blocked(row_sf_plain).view(psl, hidden // 16)
    col_sf = to_blocked_grouped(torch.cat(col_sf_blocks, dim=1), sizes).view(
        hidden, psl // 16
    )
    return row_codes, row_sf, col_codes, col_sf


def reference_group_weight_quantize_2d(
    W: torch.Tensor,
    global_amax: torch.Tensor,
    num_tensors: int,
    *,
    layout: str = "swizzled",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(codes, sf, t_codes, t_sf)`` for ``*_group_weight_quantize_2d``.

    Every expert is an independent 16x16 quantize with its own global amax, stacked -- the
    grouped kernel is a launch optimization, not a different recipe.
    """
    if W.shape[0] != num_tensors:
        raise ValueError(f"expected {num_tensors} experts, got {W.shape[0]}")
    _, m, n = W.shape
    per_expert = [
        reference_weight_quantize_2d(W[e], global_amax[e], layout=layout)
        for e in range(num_tensors)
    ]

    def _stack(outs, rows, cols):
        # to_blocked returns a flat buffer; the kernels return it shaped
        # (E, rows//128, cols//64, 32, 16).
        scales = torch.stack([o.scales for o in outs])
        if layout == "swizzled":
            scales = scales.view(num_tensors, rows // 128, cols // 64, 32, 16)
        return torch.stack([o.codes for o in outs]), scales

    row_codes, row_scales = _stack([row for row, _ in per_expert], m, n)
    col_codes, col_scales = _stack([col for _, col in per_expert], n, m)
    return row_codes, row_scales, col_codes, col_scales
