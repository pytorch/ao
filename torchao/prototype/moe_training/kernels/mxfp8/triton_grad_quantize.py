# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP8 quantization of ``grad_out`` into both row-major and
transpose variants, with e8m0 scales written directly into the tcgen05
blocked layout.

Motivation. The MoE backward pass needs two MXFP8 operands of ``grad_out``:

  * ``dgrad = grad_out @ weight``    -> needs MXFP8(``grad_out``) with
    rowwise (along-N) scales, ``(total_M, N)`` row-major.
  * ``wgrad = grad_out.T @ x``       -> needs MXFP8(``grad_out.T``) with
    rowwise scales along the new reduction axis (which is M), ``(N,
    total_M)`` row-major.

Today this takes two separate Triton passes plus two separate blocked-
layout rearrangements, so the bf16 ``grad_out`` gets read twice and every
scale tensor is written twice.

This module implements the combined "read once, emit four outputs" kernel
as a **single fused Triton pass**:

  ``_dim0_dim1_fused_kernel`` reads one ``(BLOCK_M, BLOCK_N) = (128, 128)``
  bf16 tile, computes rowwise and colwise scales in-register (the colwise
  reduction runs through a single bf16 register transpose, so both
  reductions stay warp-local), and emits:

    * ``qdata_dim0 [M, N]`` e4m3 row-major + blocked e8m0 scales
    * ``qdata_dim1_t [N, M]`` e4m3 row-major + blocked e8m0 scales

  The two fp8 stores target disjoint output tensors so the memory
  controller can issue them concurrently; in practice this means the
  slow ``(N, M)`` scattered-stride store is overlapped with the fast
  ``(M, N)`` coalesced store, rather than the two passes serializing
  on HBM as they did in the split-kernel version.

Layout contract (v1, matches ``triton_dispatch_quantize.py``):

  * ``BLOCK_M = BLOCK_N = 128`` and ``inner_block_size = 32``, so each CTA
    writes exactly one ``(128, 4)`` blocked scale tile per output. This
    matches ``triton_mx_block_rearrange`` for alignment=128.
  * ``M % 128 == 0`` and ``N % 128 == 0`` required in v1.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.utils._triton import has_triton

from torchao.utils import ceil_div, torch_version_at_least

_BLOCK_M = 128
_BLOCK_N = 128
_SCALE_BLOCK_SIZE = 32
_SCALE_BLOCK_COLS = 4  # == BLOCK_M // SCALE_BLOCK_SIZE == BLOCK_N // SCALE_BLOCK_SIZE


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.kernels import (
        _triton_calculate_scale_floor,
        _triton_calculate_scale_rceil,
    )

    def _fused_autotune_configs():
        # Fused kernel does one bf16 load + one register transpose + two
        # rowwise reductions + two coalesced fp8 stores per CTA, so the
        # working set is roughly 2x the dispatch kernel's. num_warps=8
        # gives us enough threads to hide the transpose while staying
        # under the register pressure that would cap occupancy at 1 CTA
        # per SM. num_warps=4 wins at narrow N where we're already
        # occupancy-limited by CTA count.
        configs = []
        for num_warps in (4, 8):
            for num_stages in (2, 3, 4):
                configs.append(
                    triton.Config({}, num_warps=num_warps, num_stages=num_stages)
                )
        return configs

    @triton.autotune(
        configs=_fused_autotune_configs(),
        key=["M", "N", "SCALING_MODE"],
    )
    @triton.jit
    def _dim0_dim1_fused_kernel(
        x_ptr,  # (M, N) bf16 row-major input
        qdata_dim0_ptr,  # (M, N) e4m3 row-major out (rowwise scales)
        qdata_dim1_t_ptr,  # (N, M) e4m3 row-major out (colwise scales; transpose)
        scales_dim0_ptr,  # flat (M * padded_scale_cols_n,) e8m0 out (blocked)
        scales_dim1_ptr,  # flat (N * padded_scale_cols_m,) e8m0 out (blocked)
        M,
        N,
        padded_scale_cols_n,
        padded_scale_cols_m,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
        SCALE_BLOCK_COLS: tl.constexpr,
    ):
        """Single-pass fused MXFP8 quantize of a ``(BLOCK_M, BLOCK_N)`` bf16 tile.

        Traffic per CTA (128 x 128 tile):
            read  : 2 * BLOCK_M * BLOCK_N bf16 = 32 KB (single HBM load,
                    consumed by BOTH dim0 and dim1 scale paths)
            write : BLOCK_M * BLOCK_N fp8 dim0    = 16 KB  (coalesced (M,N))
                    BLOCK_M * BLOCK_N fp8 dim1_t  = 16 KB  (coalesced (N,M))
                    tiny scale stores (<= 1 KB total)

        Why single-pass wins vs two-pass:
          * bf16 input is read exactly once per CTA, not twice.
          * The ``(M, N)`` and ``(N, M)`` output tensors are disjoint
            allocations so the memory controller can issue both fp8
            stores concurrently (different HBM channels / banks). The
            scattered-stride ``(N, M)`` store that bottlenecks the
            dim1-alone kernel gets overlapped with the coalesced
            ``(M, N)`` dim0 store, rather than running back-to-back.
          * Only one bf16 register transpose (BLOCK_M=BLOCK_N=128, 32 KB)
            is materialized, and it is reused for the colwise scale
            reduction AND the dim1_t fp8 store -- the compiler elides
            the second transpose because the stored tile shape already
            matches ``(BLOCK_N, BLOCK_M)``.

        Scale layouts. Both scale outputs use the same tcgen05 super-tile
        pattern as ``triton_mx_block_rearrange`` for alignment=128:

            dest_within_super_tile = (r % 32) * 16 + (r // 32) * 4 + c

        so each CTA writes one ``(BLOCK_M, SCALE_BLOCK_COLS)`` dim0 tile
        and one ``(BLOCK_N, SCALE_BLOCK_COLS)`` dim1 tile using that
        permutation.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x_offs = m_offs[:, None].to(tl.int64) * N + n_offs[None, :]
        x_mn = tl.load(x_ptr + x_offs)  # (BLOCK_M, BLOCK_N) bf16

        # --- dim0 (rowwise, along N) ----------------------------------
        x_r0 = x_mn.reshape(BLOCK_M * SCALE_BLOCK_COLS, SCALE_BLOCK_SIZE)
        x_abs_r0 = tl.abs(x_r0)
        if SCALING_MODE == "rceil":
            descale_r0, scale_e8m0_r0 = _triton_calculate_scale_rceil(
                x_abs_r0, axis=1, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale_r0, scale_e8m0_r0 = _triton_calculate_scale_floor(
                x_abs_r0, axis=1
            )
        scaled_r0 = x_r0 * descale_r0[:, None]
        fp8_mn = tl.reshape(scaled_r0, BLOCK_M, BLOCK_N).to(tl.float8e4nv)
        tl.store(qdata_dim0_ptr + x_offs, fp8_mn)

        # --- dim1 (colwise, along M via single bf16 transpose) --------
        x_nm = tl.trans(x_mn)  # (BLOCK_N, BLOCK_M) bf16, one register shuffle
        x_r1 = x_nm.reshape(BLOCK_N * SCALE_BLOCK_COLS, SCALE_BLOCK_SIZE)
        x_abs_r1 = tl.abs(x_r1)
        if SCALING_MODE == "rceil":
            descale_r1, scale_e8m0_r1 = _triton_calculate_scale_rceil(
                x_abs_r1, axis=1, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale_r1, scale_e8m0_r1 = _triton_calculate_scale_floor(
                x_abs_r1, axis=1
            )
        scaled_r1 = x_r1 * descale_r1[:, None]
        fp8_nm = tl.reshape(scaled_r1, BLOCK_N, BLOCK_M).to(tl.float8e4nv)
        out_offs_dim1 = n_offs[:, None].to(tl.int64) * M + m_offs[None, :]
        tl.store(qdata_dim1_t_ptr + out_offs_dim1, fp8_nm)

        # --- dim0 scales: one (BLOCK_M, SCALE_BLOCK_COLS) super-tile --
        scale0_2d = scale_e8m0_r0.reshape(BLOCK_M, SCALE_BLOCK_COLS)
        tile_row0 = tl.arange(0, BLOCK_M)[:, None]
        tile_col0 = tl.arange(0, SCALE_BLOCK_COLS)[None, :]
        dest0 = (tile_row0 % 32) * 16 + (tile_row0 // 32) * 4 + tile_col0
        dest0_flat = tl.reshape(dest0, BLOCK_M * SCALE_BLOCK_COLS)
        scale0_flat = tl.reshape(scale0_2d, BLOCK_M * SCALE_BLOCK_COLS)
        tile_base_dim0 = (
            pid_m * BLOCK_M * padded_scale_cols_n
            + pid_n * BLOCK_M * SCALE_BLOCK_COLS
        )
        tl.store(scales_dim0_ptr + tile_base_dim0 + dest0_flat, scale0_flat)

        # --- dim1 scales: one (BLOCK_N, SCALE_BLOCK_COLS) super-tile --
        scale1_2d = scale_e8m0_r1.reshape(BLOCK_N, SCALE_BLOCK_COLS)
        tile_row1 = tl.arange(0, BLOCK_N)[:, None]
        tile_col1 = tl.arange(0, SCALE_BLOCK_COLS)[None, :]
        dest1 = (tile_row1 % 32) * 16 + (tile_row1 // 32) * 4 + tile_col1
        dest1_flat = tl.reshape(dest1, BLOCK_N * SCALE_BLOCK_COLS)
        scale1_flat = tl.reshape(scale1_2d, BLOCK_N * SCALE_BLOCK_COLS)
        tile_base_dim1 = (
            pid_n * BLOCK_N * padded_scale_cols_m
            + pid_m * BLOCK_N * SCALE_BLOCK_COLS
        )
        tl.store(scales_dim1_ptr + tile_base_dim1 + dest1_flat, scale1_flat)

    def triton_mxfp8_quantize_dim0_dim1(
        x: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fused dim0 + dim1 MXFP8 quantization of ``x`` with blocked scales.

        Drop-in replacement for the sequence

            qdata0, scales0 = triton_to_mxfp8_dim0(x, 32, mode)
            qdata1_t, scales1 = triton_to_mxfp8_dim1(x, 32, mode)
            scales0_blocked = triton_mx_block_rearrange(scales0)
            scales1_blocked = triton_mx_block_rearrange(scales1)

        Implemented as a single Triton kernel that reads ``x`` once and
        emits all four outputs (two fp8 tensors + two blocked-layout
        scale tensors). Total HBM traffic = ``2MN + 2MN + scale bytes``,
        which matches the memcpy-equivalent lower bound.

        Args:
            x: bfloat16 tensor of shape ``(M, N)``, row-major.
                ``M`` and ``N`` must both be multiples of 128 in v1.
            scaling_mode: ``"rceil"`` (default) or ``"floor"``.

        Returns:
            ``qdata_dim0``: ``float8_e4m3fn`` tensor of shape ``(M, N)``,
                row-major. Scales live along dim 1.
            ``qdata_dim1_t``: ``float8_e4m3fn`` tensor of shape ``(N, M)``,
                row-major. Scales live along dim 1 (= along M of the original
                input).
            ``scales_dim0_blocked``: ``float8_e8m0fnu`` flat tensor of
                length ``M * padded_scale_cols_n`` in tcgen05 blocked
                layout.
            ``scales_dim1_blocked``: ``float8_e8m0fnu`` flat tensor of
                length ``N * padded_scale_cols_m`` in tcgen05 blocked
                layout.
        """
        assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
        assert x.is_contiguous(), "x must be contiguous"
        assert x.ndim == 2, f"x must be 2D, got {x.ndim}D"
        assert scaling_mode in ("rceil", "floor"), (
            f"scaling_mode must be 'rceil' or 'floor', got {scaling_mode}"
        )
        M, N = x.shape
        assert M % _BLOCK_M == 0, (
            f"M must be a multiple of {_BLOCK_M}, got M={M}"
        )
        assert N % _BLOCK_N == 0, (
            f"N must be a multiple of {_BLOCK_N}, got N={N}"
        )

        padded_scale_cols_n = ceil_div(N // _SCALE_BLOCK_SIZE, _SCALE_BLOCK_COLS) * _SCALE_BLOCK_COLS
        padded_scale_cols_m = ceil_div(M // _SCALE_BLOCK_SIZE, _SCALE_BLOCK_COLS) * _SCALE_BLOCK_COLS

        qdata_dim0 = torch.empty(
            (M, N), dtype=torch.float8_e4m3fn, device=x.device
        )
        qdata_dim1_t = torch.empty(
            (N, M), dtype=torch.float8_e4m3fn, device=x.device
        )
        scales_dim0_blocked_u8 = torch.empty(
            (M * padded_scale_cols_n,), dtype=torch.uint8, device=x.device
        )
        scales_dim1_blocked_u8 = torch.empty(
            (N * padded_scale_cols_m,), dtype=torch.uint8, device=x.device
        )

        grid = (M // _BLOCK_M, N // _BLOCK_N)
        _dim0_dim1_fused_kernel[grid](
            x,
            qdata_dim0,
            qdata_dim1_t,
            scales_dim0_blocked_u8,
            scales_dim1_blocked_u8,
            M,
            N,
            padded_scale_cols_n,
            padded_scale_cols_m,
            SCALING_MODE=scaling_mode,
            USE_PTX=x.device.type != "hip",
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            SCALE_BLOCK_SIZE=_SCALE_BLOCK_SIZE,
            SCALE_BLOCK_COLS=_SCALE_BLOCK_COLS,
        )

        return (
            qdata_dim0,
            qdata_dim1_t,
            scales_dim0_blocked_u8.view(torch.float8_e8m0fnu),
            scales_dim1_blocked_u8.view(torch.float8_e8m0fnu),
        )

else:

    def triton_mxfp8_quantize_dim0_dim1(  # type: ignore[no-redef]
        x: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError(
            "triton_mxfp8_quantize_dim0_dim1 requires torch >= 2.7 and triton"
        )


__all__ = ["triton_mxfp8_quantize_dim0_dim1"]
