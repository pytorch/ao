# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MoE dispatch + MXFP8 quantization + blocked-scale rearrangement kernel.

This module implements Triton kernels that fuse, in one GPU pass:

  * per-row gather from a source tensor via a permutation index vector
    (MoE token dispatch / per-group padding),
  * per-row MXFP8 quantization (1x32 scaling granularity), and
  * direct write of scales in the tcgen05 blocked layout consumed by
    ``torch._scaled_grouped_mm``.

Motivation (ao#4184). The existing MoE forward pipeline on SM 10.x does three
sequential memory passes: pad token groups -> MXFP8 quantize -> rearrange
scales into the blocked layout. For realistic batch sizes (total_M in the
4k-32k range on DeepSeek-V3-like shapes) this is memory-bandwidth bound, and
the padding copy alone burns ~1/3 of the budget. Fusing these three steps into
one kernel lets us touch global memory exactly once per token.

The same core kernel handles both the non-EP path (source rows are in
expert-major order, we only pad) and the EP path (``src_indices`` is an
arbitrary permutation produced by ``generate_permute_indices``): both reduce
to "for each output row r, gather x[src_indices[r], :] or emit zeros if
src_indices[r] == -1".

Layout contract (v1, alignment = 128):

  * Per-group sizes are multiples of 128 so that groups align with the
    128-row blocks of the blocked scale layout. This lets each CTA write
    exactly one (128, 4) blocked tile with no cross-group bookkeeping.
  * The blocked scale output matches the layout produced by
    ``triton_mx_block_rearrange_2d_M_groups`` for alignment=128 (identical
    bytewise on non-gap cells).
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.utils._triton import has_triton

from torchao.utils import ceil_div, torch_version_at_least

# v1 tuning parameters (see module docstring).
_ALIGNMENT = 128
_BLOCK_ROWS = 128
_SCALE_BLOCK_COLS = 4
_SCALE_BLOCK_SIZE = 32
_COL_TILE_SIZE = 128  # == _SCALE_BLOCK_COLS * _SCALE_BLOCK_SIZE


def _validate_src_indices(
    x: Tensor,
    src_indices: Tensor,
) -> Tuple[int, int, int, int]:
    assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.ndim == 2, f"x must be 2D, got {x.ndim}D"
    assert src_indices.dtype == torch.int32, (
        f"src_indices must be int32, got {src_indices.dtype}"
    )
    assert src_indices.ndim == 1, "src_indices must be 1D"
    unpermuted_M, K = x.shape
    padded_M = src_indices.shape[0]
    assert K % _SCALE_BLOCK_SIZE == 0, (
        f"K must be divisible by {_SCALE_BLOCK_SIZE}, got K={K}"
    )
    assert padded_M % _ALIGNMENT == 0, (
        f"src_indices length must be a multiple of {_ALIGNMENT}, got {padded_M}"
    )
    return unpermuted_M, K, padded_M, K // _SCALE_BLOCK_SIZE


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.kernels import (
        _triton_calculate_scale_floor,
        _triton_calculate_scale_rceil,
    )

    def _dispatch_quantize_autotune_configs():
        # BLOCK_ROWS and COL_TILE_SIZE are locked by the blocked scale layout
        # in v1; we only sweep over warp/stage counts. A broader tuning pass
        # can come once we have more shapes to profile against.
        configs = []
        for num_warps in (4, 8):
            for num_stages in (2, 3, 4):
                configs.append(
                    triton.Config({}, num_warps=num_warps, num_stages=num_stages)
                )
        return configs

    @triton.jit
    def _quantize_row_tile_and_store(
        x_block,
        m_offs,
        col_offs,
        row_mask,
        col_mask,
        qdata_ptr,
        scales_ptr,
        K,
        padded_scale_cols,
        pid_m,
        pid_k,
        BLOCK_ROWS: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
        SCALE_BLOCK_COLS: tl.constexpr,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
    ):
        """Given a (BLOCK_ROWS x COL_TILE_SIZE) bf16 tile, MXFP8-quantize it
        rowwise and write fp8 row-major + one (128,4) scale tile in the
        tcgen05 blocked layout."""
        x_block_r = x_block.reshape(
            BLOCK_ROWS * SCALE_BLOCK_COLS, SCALE_BLOCK_SIZE
        )
        x_block_abs_r = tl.abs(x_block_r)

        if SCALING_MODE == "rceil":
            descale_fp32_r, scale_e8m0_r = _triton_calculate_scale_rceil(
                x_block_abs_r, axis=1, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale_fp32_r, scale_e8m0_r = _triton_calculate_scale_floor(
                x_block_abs_r, axis=1
            )

        scaled_r = x_block_r * descale_fp32_r[:, None]
        fp8_2d = tl.reshape(scaled_r, BLOCK_ROWS, COL_TILE_SIZE).to(tl.float8e4nv)

        fp8_mask = row_mask[:, None] & col_mask[None, :]
        fp8_offs = m_offs[:, None].to(tl.int64) * K + col_offs[None, :]
        tl.store(qdata_ptr + fp8_offs, fp8_2d, mask=fp8_mask)

        scale_e8m0_2d = scale_e8m0_r.reshape(BLOCK_ROWS, SCALE_BLOCK_COLS)

        tile_row_offs = tl.arange(0, BLOCK_ROWS)[:, None]
        tile_col_offs = tl.arange(0, SCALE_BLOCK_COLS)[None, :]
        r_div_32 = tile_row_offs // 32
        r_mod_32 = tile_row_offs % 32
        dest_indices = r_mod_32 * 16 + r_div_32 * 4 + tile_col_offs
        dest_indices_flat = tl.reshape(
            dest_indices, (BLOCK_ROWS * SCALE_BLOCK_COLS)
        )
        scales_flat = tl.reshape(
            scale_e8m0_2d, (BLOCK_ROWS * SCALE_BLOCK_COLS)
        )

        tile_block_base = (
            pid_m * BLOCK_ROWS * padded_scale_cols
            + pid_k * BLOCK_ROWS * SCALE_BLOCK_COLS
        )
        tl.store(
            scales_ptr + tile_block_base + dest_indices_flat,
            scales_flat,
        )

    @triton.autotune(
        configs=_dispatch_quantize_autotune_configs(),
        key=["K", "SCALING_MODE"],
    )
    @triton.jit
    def _dispatch_and_quantize_kernel(
        x_ptr,
        src_indices_ptr,
        qdata_ptr,
        scales_ptr,
        padded_M,
        K,
        padded_scale_cols,
        BLOCK_ROWS: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
        SCALE_BLOCK_COLS: tl.constexpr,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
    ):
        """EP path: read src_indices from a precomputed tensor. -1 marks
        padding slots (the same sentinel convention used by
        ``generate_permute_indices`` and ``pad_token_groups``)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_offs = pid_m * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = m_offs < padded_M
        src_idx = tl.load(src_indices_ptr + m_offs, mask=row_mask, other=-1)
        is_valid = src_idx >= 0
        safe_src = tl.where(is_valid, src_idx, 0).to(tl.int64)

        col_offs = pid_k * COL_TILE_SIZE + tl.arange(0, COL_TILE_SIZE)
        col_mask = col_offs < K

        gather_offs = safe_src[:, None] * K + col_offs[None, :]
        load_mask = is_valid[:, None] & col_mask[None, :] & row_mask[:, None]
        x_block = tl.load(x_ptr + gather_offs, mask=load_mask, other=0.0)

        _quantize_row_tile_and_store(
            x_block,
            m_offs,
            col_offs,
            row_mask,
            col_mask,
            qdata_ptr,
            scales_ptr,
            K,
            padded_scale_cols,
            pid_m,
            pid_k,
            BLOCK_ROWS=BLOCK_ROWS,
            COL_TILE_SIZE=COL_TILE_SIZE,
            SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
            SCALE_BLOCK_COLS=SCALE_BLOCK_COLS,
            SCALING_MODE=SCALING_MODE,
            USE_PTX=USE_PTX,
        )

    @triton.autotune(
        configs=_dispatch_quantize_autotune_configs(),
        key=["K", "SCALING_MODE", "NUM_GROUPS"],
    )
    @triton.jit
    def _pad_and_quantize_kernel(
        x_ptr,
        group_offsets_ptr,  # (E,) int32 unpadded cumulative end offsets
        padded_group_end_offsets_ptr,  # (E,) int32 padded cumulative end offsets
        qdata_ptr,
        scales_ptr,
        padded_M,
        K,
        padded_scale_cols,
        NUM_GROUPS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
        SCALE_BLOCK_COLS: tl.constexpr,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
    ):
        """Non-EP path: compute src_idx per row on-the-fly from group offsets,
        avoiding a separate src_indices allocation + kernel launch.

        Because alignment=128 and BLOCK_ROWS=128, every row in this tile
        belongs to the same group (boundaries land on tile edges). We can
        therefore resolve the group once per CTA via a tiny linear scan over
        NUM_GROUPS (typically <= 16)."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_offs = pid_m * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = m_offs < padded_M

        # Resolve the group index g for this 128-row tile: g is the smallest
        # group index such that tile_start < padded_end[g]. Because
        # alignment=128 and BLOCK_ROWS=128 every row in this tile sits in
        # the same group, so one resolution per CTA is enough.
        #
        # Uses a compile-time-unrolled scan (NUM_GROUPS is typically <=32)
        # to avoid the power-of-2 constraint that tl.arange would impose.
        tile_start = pid_m * BLOCK_ROWS
        g = 0
        for i in tl.static_range(NUM_GROUPS):
            next_end = tl.load(padded_group_end_offsets_ptr + i)
            g = tl.where(tile_start >= next_end, i + 1, g)
        # Clamp: padding tiles beyond the last real group use g=NUM_GROUPS-1
        # and the (within < orig_size) check below maps them all to -1.
        g_clamped = tl.minimum(g, NUM_GROUPS - 1)

        # Gather per-group values without going out of bounds at g=0.
        safe_prev = tl.maximum(g_clamped - 1, 0)
        padded_prev = tl.load(padded_group_end_offsets_ptr + safe_prev)
        orig_prev = tl.load(group_offsets_ptr + safe_prev)
        padded_start = tl.where(g_clamped == 0, 0, padded_prev)
        orig_start = tl.where(g_clamped == 0, 0, orig_prev)
        orig_end = tl.load(group_offsets_ptr + g_clamped)
        orig_size = orig_end - orig_start

        within = m_offs - padded_start
        is_valid = (within < orig_size) & (within >= 0)
        src_idx = tl.where(is_valid, orig_start + within, 0).to(tl.int64)

        col_offs = pid_k * COL_TILE_SIZE + tl.arange(0, COL_TILE_SIZE)
        col_mask = col_offs < K

        gather_offs = src_idx[:, None] * K + col_offs[None, :]
        load_mask = is_valid[:, None] & col_mask[None, :] & row_mask[:, None]
        x_block = tl.load(x_ptr + gather_offs, mask=load_mask, other=0.0)

        _quantize_row_tile_and_store(
            x_block,
            m_offs,
            col_offs,
            row_mask,
            col_mask,
            qdata_ptr,
            scales_ptr,
            K,
            padded_scale_cols,
            pid_m,
            pid_k,
            BLOCK_ROWS=BLOCK_ROWS,
            COL_TILE_SIZE=COL_TILE_SIZE,
            SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
            SCALE_BLOCK_COLS=SCALE_BLOCK_COLS,
            SCALING_MODE=SCALING_MODE,
            USE_PTX=USE_PTX,
        )

    @triton.jit
    def _compute_padded_group_offsets_kernel(
        group_offsets_ptr,  # (E,) int32 cumulative end offsets (unpadded)
        padded_start_ptr,  # (E,) int32 out
        padded_end_ptr,  # (E,) int32 out
        NUM_GROUPS: tl.constexpr,
        NUM_GROUPS_PO2: tl.constexpr,
        ALIGNMENT: tl.constexpr,
    ):
        """Compute padded_group_start_offsets and padded_group_end_offsets in
        a single kernel launch. One CTA; NUM_GROUPS_PO2 threads do a prefix
        sum of the aligned group sizes (slots past NUM_GROUPS contribute
        zero so the cumsum stays correct)."""
        i = tl.arange(0, NUM_GROUPS_PO2)
        valid = i < NUM_GROUPS
        offsets = tl.load(group_offsets_ptr + i, mask=valid, other=0)
        prev_offsets = tl.load(
            group_offsets_ptr + i - 1, mask=valid & (i > 0), other=0
        )
        group_sizes = offsets - prev_offsets
        padded_sizes = ((group_sizes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        padded_sizes = tl.where(valid, padded_sizes, 0)
        padded_end = tl.cumsum(padded_sizes, axis=0)
        padded_start = padded_end - padded_sizes
        tl.store(padded_start_ptr + i, padded_start.to(tl.int32), mask=valid)
        tl.store(padded_end_ptr + i, padded_end.to(tl.int32), mask=valid)

    def triton_mxfp8_dispatch_and_quantize(
        x: Tensor,
        src_indices: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor]:
        """Fused gather + MXFP8 quantize + blocked-scale rearrange (EP path).

        For each output row r in ``[0, src_indices.numel())``:

          * ``src = src_indices[r]``;
          * if ``src >= 0``, emit MXFP8(``x[src, :]``);
          * if ``src == -1``, emit an all-zero row (data *and* scales), which
            is the sentinel used by ``permute_and_pad`` / ``pad_token_groups``.

        Args:
            x: bfloat16 tensor of shape ``(unpermuted_M, K)``, row-major.
            src_indices: int32 tensor of shape ``(padded_M,)``. ``padded_M``
                must be a multiple of 128 and each per-group slice must also
                be a multiple of 128 (i.e. alignment=128).
            scaling_mode: ``"rceil"`` (default) or ``"floor"``.

        Returns:
            qdata: ``float8_e4m3fn``, shape ``(padded_M, K)``, row-major.
            blocked_scales: ``float8_e8m0fnu`` flat tensor of shape
                ``(padded_M * padded_scale_cols,)``, in tcgen05 blocked layout
                (same byte layout as ``triton_mx_block_rearrange_2d_M_groups``
                for alignment=128).
        """
        assert scaling_mode in ("rceil", "floor"), (
            f"scaling_mode must be 'rceil' or 'floor', got {scaling_mode}"
        )
        _, K, padded_M, k_blocks = _validate_src_indices(x, src_indices)
        assert K % _COL_TILE_SIZE == 0, (
            f"K must be a multiple of {_COL_TILE_SIZE} in v1, got K={K}"
        )
        padded_scale_cols = ceil_div(k_blocks, _SCALE_BLOCK_COLS) * _SCALE_BLOCK_COLS

        qdata = torch.empty(
            (padded_M, K), dtype=torch.float8_e4m3fn, device=x.device
        )
        blocked_scales_u8 = torch.empty(
            (padded_M * padded_scale_cols,),
            dtype=torch.uint8,
            device=x.device,
        )

        grid = (padded_M // _BLOCK_ROWS, K // _COL_TILE_SIZE)
        _dispatch_and_quantize_kernel[grid](
            x,
            src_indices,
            qdata,
            blocked_scales_u8,
            padded_M,
            K,
            padded_scale_cols,
            BLOCK_ROWS=_BLOCK_ROWS,
            COL_TILE_SIZE=_COL_TILE_SIZE,
            SCALE_BLOCK_SIZE=_SCALE_BLOCK_SIZE,
            SCALE_BLOCK_COLS=_SCALE_BLOCK_COLS,
            SCALING_MODE=scaling_mode,
            USE_PTX=x.device.type != "hip",
        )

        return qdata, blocked_scales_u8.view(torch.float8_e8m0fnu)

    def triton_mxfp8_pad_and_quantize(
        x: Tensor,
        group_offsets: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Non-EP drop-in for ``pad_token_groups`` +
        ``triton_to_mxfp8_dim0`` + ``triton_mx_block_rearrange_2d_M_groups``,
        all in one Triton pass. Uses alignment=128.

        Uses upper-bound output sizing (``num_tokens + num_groups * 128``),
        same as ``fused_pad_token_groups_cuda``, so the whole forward is
        free of D2H ``.item()`` syncs -- the actual padded ends live in
        ``padded_group_end_offsets`` on-device, which is what the downstream
        ``torch._scaled_grouped_mm`` consumes.

        Args:
            x: bfloat16 ``(M, K)``, row-major.
            group_offsets: int32 ``(E,)`` cumulative unpadded end indices.
            scaling_mode: ``"rceil"`` or ``"floor"``.

        Returns:
            ``(qdata, blocked_scales, padded_group_start_offsets,
            padded_group_end_offsets)``. Shapes / dtypes match the three
            existing kernels' combined outputs.
        """
        assert group_offsets.dtype == torch.int32
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        assert x.ndim == 2
        assert scaling_mode in ("rceil", "floor"), (
            f"scaling_mode must be 'rceil' or 'floor', got {scaling_mode}"
        )
        num_tokens, K = x.shape
        assert K % _COL_TILE_SIZE == 0, (
            f"K must be a multiple of {_COL_TILE_SIZE} in v1, got K={K}"
        )
        num_groups = int(group_offsets.shape[0])
        # Upper-bound sizing (host-computable, no sync): each of the
        # num_groups groups needs at most 128 bytes of padding beyond its
        # unpadded rows. Round up to BLOCK_ROWS so the main kernel's grid is
        # static (no tail handling).
        padded_M = num_tokens + num_groups * _ALIGNMENT
        padded_M = ((padded_M + _BLOCK_ROWS - 1) // _BLOCK_ROWS) * _BLOCK_ROWS

        k_blocks = K // _SCALE_BLOCK_SIZE
        padded_scale_cols = ceil_div(k_blocks, _SCALE_BLOCK_COLS) * _SCALE_BLOCK_COLS

        # Single tiny kernel emits both offset tensors together -- replaces a
        # chain of ~8 small torch ops (diff, cdiv, cumsum, sub, cat, ...) each
        # of which costs a ~10us dispatch overhead on B200.
        padded_group_start_offsets = torch.empty(
            (num_groups,), dtype=torch.int32, device=x.device
        )
        padded_group_end_offsets = torch.empty(
            (num_groups,), dtype=torch.int32, device=x.device
        )
        num_groups_po2 = triton.next_power_of_2(num_groups)
        _compute_padded_group_offsets_kernel[(1,)](
            group_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            NUM_GROUPS=num_groups,
            NUM_GROUPS_PO2=num_groups_po2,
            ALIGNMENT=_ALIGNMENT,
        )

        qdata = torch.empty(
            (padded_M, K), dtype=torch.float8_e4m3fn, device=x.device
        )
        blocked_scales_u8 = torch.empty(
            (padded_M * padded_scale_cols,),
            dtype=torch.uint8,
            device=x.device,
        )

        grid = (padded_M // _BLOCK_ROWS, K // _COL_TILE_SIZE)
        _pad_and_quantize_kernel[grid](
            x,
            group_offsets,
            padded_group_end_offsets,
            qdata,
            blocked_scales_u8,
            padded_M,
            K,
            padded_scale_cols,
            NUM_GROUPS=num_groups,
            BLOCK_ROWS=_BLOCK_ROWS,
            COL_TILE_SIZE=_COL_TILE_SIZE,
            SCALE_BLOCK_SIZE=_SCALE_BLOCK_SIZE,
            SCALE_BLOCK_COLS=_SCALE_BLOCK_COLS,
            SCALING_MODE=scaling_mode,
            USE_PTX=x.device.type != "hip",
        )

        return (
            qdata,
            blocked_scales_u8.view(torch.float8_e8m0fnu),
            padded_group_start_offsets,
            padded_group_end_offsets,
        )

else:

    def triton_mxfp8_dispatch_and_quantize(  # type: ignore[no-redef]
        x: Tensor,
        src_indices: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "triton_mxfp8_dispatch_and_quantize requires torch >= 2.7 and triton"
        )

    def triton_mxfp8_pad_and_quantize(  # type: ignore[no-redef]
        x: Tensor,
        group_offsets: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError(
            "triton_mxfp8_pad_and_quantize requires torch >= 2.7 and triton"
        )


__all__ = [
    "triton_mxfp8_dispatch_and_quantize",
    "triton_mxfp8_pad_and_quantize",
]
