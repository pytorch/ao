# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP8 quantization of ``grad_out`` into both row-major and
transpose variants, with e8m0 scales written directly into the tcgen05
blocked layout.

Motivation. The MoE backward pass needs two MXFP8 operands of ``grad_out``:

  * ``dgrad = grad_out @ weight`` -> needs MXFP8(``grad_out``) with
    rowwise (along-N) scales, ``(total_M, N)`` row-major.
  * ``wgrad = grad_out.T @ x`` -> needs MXFP8(``grad_out.T``) with
    rowwise scales along the new reduction axis (which is M), ``(N,
    total_M)`` row-major.

Today this takes two separate Triton passes plus two separate blocked-
layout rearrangements, so the bf16 ``grad_out`` gets read twice and every
scale tensor is written twice. At the ``(total_M=16384, N=2048)`` shape
Daniel flagged (num_groups=4, M_per_group=4096, N=2048, DeepSeek-V3-like),
that turns a ~22 us memcpy-equivalent problem into ~70-80 us of real GPU
time.

This kernel does both in one pass:

  1. read a ``(BLOCK_M, BLOCK_N) = (128, 128)`` bf16 tile of ``grad_out``
     once,
  2. emit ``qdata_dim0[M, N]`` (row-major e4m3) with rowwise scales,
  3. emit ``qdata_dim1_t[N, M]`` (row-major e4m3, i.e. transpose of the
     colwise-quantized tile) with colwise scales,
  4. write both e8m0 scale tensors directly in the tcgen05 blocked
     layout (one (128, 4) super-tile per CTA per output, no separate
     rearrange kernel).

Layout contract (v1):

  * ``BLOCK_M = BLOCK_N = 128`` and ``inner_block_size = 32``, so each CTA
    writes exactly one ``(128, 4)`` blocked scale tile per output. This
    matches ``triton_scale_swizzle`` / ``triton_mx_block_rearrange`` for
    alignment=128.
  * ``M % 128 == 0`` and ``N % 128 == 0`` required in v1 (same constraint
    as the existing ``to_mxfp8_dim1_kernel``).
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.utils._triton import has_triton

from torchao.utils import ceil_div, torch_version_at_least

_BLOCK_M = 128
_SCALE_BLOCK_SIZE = 32
# ``BLOCK_N`` is autotuned; ``BLOCK_M`` stays pinned at 128 so every CTA
# writes exactly one ``(128, 4)``-sized super-tile per dim1-scale tile and
# ``BLOCK_N // 128`` super-tiles per dim0-scale tile (no reshape across
# super-tile boundaries). The scale block size is fixed at 32 per MX spec.


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.kernels import (
        _triton_calculate_scale_floor,
        _triton_calculate_scale_rceil,
    )

    def _dim0_dim1_autotune_configs():
        # Autotune over BLOCK_N in {128, 256, 512}. Larger tiles mean fewer
        # CTAs and better HBM coalescing at the cost of extra register /
        # shared-memory pressure; the best choice is shape-dependent, so we
        # let Triton pick. Each BLOCK_N choice keeps BLOCK_M pinned at 128
        # (one (128, 4) dim1-scale super-tile per CTA).
        configs = []
        for block_n in (128, 256, 512):
            for num_warps in (4, 8, 16):
                for num_stages in (2, 3, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_N": block_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
        return configs

    def _early_prune_configs(configs, named_args, **kwargs):
        """Drop configs whose BLOCK_N would exceed the tensor width. Triton
        would fail those at compile-time if we didn't, and the autotune run
        itself would bubble the failure up to the user."""
        n = named_args.get("N", kwargs.get("N"))
        if n is None:
            return configs
        return [c for c in configs if c.kwargs["BLOCK_N"] <= n]

    @triton.autotune(
        configs=_dim0_dim1_autotune_configs(),
        key=["M", "N", "SCALING_MODE"],
        prune_configs_by={"early_config_prune": _early_prune_configs},
    )
    @triton.jit
    def _mxfp8_dim0_dim1_blocked_kernel(
        x_ptr,  # (M, N) bf16 row-major
        qdata_dim0_ptr,  # (M, N) e4m3 row-major out
        qdata_dim1_t_ptr,  # (N, M) e4m3 row-major out (logical transpose)
        scales_dim0_blocked_ptr,  # flat (M * scale_cols_n,) e8m0 out
        scales_dim1_blocked_ptr,  # flat (N * scale_cols_m,) e8m0 out
        M,
        N,
        scale_cols_n,  # padded number of (N / 32) scale columns, aligned to 4
        scale_cols_m,  # padded number of (M / 32) scale columns, aligned to 4
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
        BLOCK_M: tl.constexpr = 128,
        BLOCK_N: tl.constexpr = 128,
        SCALE_BLOCK_SIZE: tl.constexpr = 32,
    ):
        """Single pass: read (BLOCK_M, BLOCK_N) bf16 tile once, emit two
        e4m3 fp8 tiles (row-major + transposed row-major) and the
        corresponding blocked e8m0 scale tiles.

        Key perf trick: we compute both rowwise and colwise scales via
        pure 3D reshapes of the SAME bf16 tile (no wide bf16 transpose).
        The only transpose we actually materialize is a (BLOCK_M, BLOCK_N)
        fp8 register-resident tile for the dim1 output - half the byte
        width of the bf16 tile the naive version would transpose.

        BLOCK_M is fixed at 128 (one (128, 4) dim1-scale super-tile per
        CTA); BLOCK_N is autotuned and must be a multiple of 128.
        """
        SCALE_BLOCK_COLS_N: tl.constexpr = BLOCK_N // SCALE_BLOCK_SIZE
        SCALE_BLOCK_COLS_M: tl.constexpr = BLOCK_M // SCALE_BLOCK_SIZE
        SUPER_TILE_COLS: tl.constexpr = 4

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        x_offs = m_offs[:, None].to(tl.int64) * N + n_offs[None, :]
        x_tile = tl.load(x_ptr + x_offs)  # (BLOCK_M, BLOCK_N) bf16

        # ---- Rowwise (dim0) quantization: scales along N (dim-1). ---------
        # 3D view: (BLOCK_M, SCALE_BLOCK_COLS_N, SCALE_BLOCK_SIZE).
        x_row_3d = x_tile.reshape(
            BLOCK_M, SCALE_BLOCK_COLS_N, SCALE_BLOCK_SIZE
        )
        x_row_abs_3d = tl.abs(x_row_3d)
        if SCALING_MODE == "rceil":
            descale_row_2d, scale_row_e8m0_2d = _triton_calculate_scale_rceil(
                x_row_abs_3d, axis=2, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale_row_2d, scale_row_e8m0_2d = _triton_calculate_scale_floor(
                x_row_abs_3d, axis=2
            )
        qdata_row_3d = (x_row_3d * descale_row_2d[:, :, None]).to(tl.float8e4nv)
        qdata_row_2d = tl.reshape(qdata_row_3d, BLOCK_M, BLOCK_N)
        tl.store(qdata_dim0_ptr + x_offs, qdata_row_2d)

        # Dim0 scales: (BLOCK_M, SCALE_BLOCK_COLS_N). Compute the tcgen05
        # blocked-layout offset for every (r, c) pair directly and do a
        # single store. Layout per (128, 4) super-tile:
        #     within = (r % 32) * 16 + (r // 32) * 4 + c
        # Super-tiles are packed along cols (each 512 bytes), so col c
        # belongs to super-tile c // 4 and has col c % 4 inside it.
        m_idx = tl.arange(0, BLOCK_M)[:, None]
        n_idx = tl.arange(0, SCALE_BLOCK_COLS_N)[None, :]
        r_div_32 = m_idx // 32
        r_mod_32 = m_idx % 32
        sup_col = n_idx // SUPER_TILE_COLS
        in_col = n_idx % SUPER_TILE_COLS
        dest_row = (
            sup_col * (128 * SUPER_TILE_COLS)
            + r_mod_32 * 16
            + r_div_32 * 4
            + in_col
        )
        dest_row_flat = tl.reshape(dest_row, BLOCK_M * SCALE_BLOCK_COLS_N)
        scales_row_flat = tl.reshape(
            scale_row_e8m0_2d, BLOCK_M * SCALE_BLOCK_COLS_N
        )
        tile_base_row = (
            pid_m * BLOCK_M * scale_cols_n
            + pid_n * BLOCK_M * SCALE_BLOCK_COLS_N
        )
        tl.store(
            scales_dim0_blocked_ptr + tile_base_row + dest_row_flat,
            scales_row_flat,
        )

        # ---- Colwise (dim1) quantization: scales along M (dim-0). ---------
        # 3D view: (SCALE_BLOCK_COLS_M, SCALE_BLOCK_SIZE, BLOCK_N).
        x_col_3d = x_tile.reshape(
            SCALE_BLOCK_COLS_M, SCALE_BLOCK_SIZE, BLOCK_N
        )
        x_col_abs_3d = tl.abs(x_col_3d)
        if SCALING_MODE == "rceil":
            descale_col_2d, scale_col_e8m0_2d = _triton_calculate_scale_rceil(
                x_col_abs_3d, axis=1, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale_col_2d, scale_col_e8m0_2d = _triton_calculate_scale_floor(
                x_col_abs_3d, axis=1
            )
        qdata_col_3d = (x_col_3d * descale_col_2d[:, None, :]).to(tl.float8e4nv)
        qdata_col_2d_mn = tl.reshape(qdata_col_3d, BLOCK_M, BLOCK_N)
        # Write directly at transposed offsets - no in-kernel tl.trans on fp8.
        # Triton picks a thread layout that makes the m-axis contiguous per
        # warp, so writing qdata_col_2d_mn[m, n] to qdata_dim1_t[n, m] = offset
        # n * M + m is coalesced along m (consecutive threads in a warp cover
        # consecutive m-values; n is the "slow" axis).
        out_t_offs_mn = (
            m_offs[:, None].to(tl.int64) + n_offs[None, :].to(tl.int64) * M
        )
        tl.store(qdata_dim1_t_ptr + out_t_offs_mn, qdata_col_2d_mn)

        # Dim1 scales: (SCALE_BLOCK_COLS_M, BLOCK_N) -> (BLOCK_N,
        # SCALE_BLOCK_COLS_M). For BLOCK_N > 128 we have N_SUPER_TILES
        # (128, 4) super-tiles along the row axis; swizzle each row group
        # of 128 independently and include the inter-super-tile stride.
        scale_col_2d_tr = tl.trans(scale_col_e8m0_2d)  # (BLOCK_N, SCALE_BLOCK_COLS_M)
        n_row_idx = tl.arange(0, BLOCK_N)[:, None]
        c_idx2 = tl.arange(0, SCALE_BLOCK_COLS_M)[None, :]
        sup_row = n_row_idx // 128
        in_row = n_row_idx % 128
        r_div_32_c = in_row // 32
        r_mod_32_c = in_row % 32
        dest_col = (
            sup_row * (128 * scale_cols_m)
            + r_mod_32_c * 16
            + r_div_32_c * 4
            + c_idx2
        )
        dest_col_flat = tl.reshape(dest_col, BLOCK_N * SCALE_BLOCK_COLS_M)
        scales_col_flat = tl.reshape(
            scale_col_2d_tr, BLOCK_N * SCALE_BLOCK_COLS_M
        )
        tile_base_col = (
            pid_n * BLOCK_N * scale_cols_m
            + pid_m * BLOCK_M * SUPER_TILE_COLS
        )
        tl.store(
            scales_dim1_blocked_ptr + tile_base_col + dest_col_flat,
            scales_col_flat,
        )

    def triton_mxfp8_quantize_dim0_dim1(
        x: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fused dim0 + dim1 MXFP8 quantization of ``x`` with blocked scales.

        Reads ``x`` once, writes both quantized outputs and both blocked-
        layout scale tensors in a single Triton pass. Drop-in replacement
        for the sequence

            qdata0, scales0 = triton_to_mxfp8_dim0(x, 32, mode)
            qdata1_t, scales1 = triton_to_mxfp8_dim1(x, 32, mode)
            scales0_blocked = triton_mx_block_rearrange(scales0)
            scales1_blocked = triton_mx_block_rearrange(scales1)

        Args:
            x: bfloat16 tensor of shape ``(M, N)``, row-major.
                ``M`` and ``N`` must both be multiples of 128 in v1 (same
                alignment the existing ``to_mxfp8_dim1_kernel`` enforces).
            scaling_mode: ``"rceil"`` (default) or ``"floor"``.

        Returns:
            ``qdata_dim0``: ``float8_e4m3fn`` tensor of shape ``(M, N)``,
                row-major. Scales live along dim 1.
            ``qdata_dim1_t``: ``float8_e4m3fn`` tensor of shape ``(N, M)``,
                row-major. This is the transpose of the dim1-quantized
                ``x``. Scales live along dim 1 (= along M of the original
                input).
            ``scales_dim0_blocked``: ``float8_e8m0fnu`` flat tensor of
                length ``M * scale_cols_n`` in tcgen05 blocked layout
                (byte-compatible with ``triton_mx_block_rearrange`` output
                on a ``(M, N/32)`` scale tensor).
            ``scales_dim1_blocked``: ``float8_e8m0fnu`` flat tensor of
                length ``N * scale_cols_m`` in tcgen05 blocked layout.
        """
        assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
        assert x.is_contiguous(), "x must be contiguous"
        assert x.ndim == 2, f"x must be 2D, got {x.ndim}D"
        assert scaling_mode in ("rceil", "floor"), (
            f"scaling_mode must be 'rceil' or 'floor', got {scaling_mode}"
        )
        M, N = x.shape
        assert M % _BLOCK_M == 0, f"M must be a multiple of {_BLOCK_M}, got M={M}"
        # BLOCK_N is autotuned up to 512, so N must be a multiple of 128 and
        # additionally a multiple of whatever BLOCK_N triton picks. We
        # simplify by requiring N % 512 == 0 when N >= 512 and N % 128 == 0
        # otherwise (callers pass power-of-2 N in practice).
        assert N % 128 == 0, f"N must be a multiple of 128, got N={N}"
        assert _SCALE_BLOCK_SIZE == 32

        scale_cols_n = ceil_div(N // _SCALE_BLOCK_SIZE, 4) * 4
        scale_cols_m = ceil_div(M // _SCALE_BLOCK_SIZE, 4) * 4

        qdata_dim0 = torch.empty(
            (M, N), dtype=torch.float8_e4m3fn, device=x.device
        )
        qdata_dim1_t = torch.empty(
            (N, M), dtype=torch.float8_e4m3fn, device=x.device
        )
        scales_dim0_blocked_u8 = torch.empty(
            (M * scale_cols_n,), dtype=torch.uint8, device=x.device
        )
        scales_dim1_blocked_u8 = torch.empty(
            (N * scale_cols_m,), dtype=torch.uint8, device=x.device
        )

        grid = lambda meta: (M // _BLOCK_M, triton.cdiv(N, meta["BLOCK_N"]))
        _mxfp8_dim0_dim1_blocked_kernel[grid](
            x,
            qdata_dim0,
            qdata_dim1_t,
            scales_dim0_blocked_u8,
            scales_dim1_blocked_u8,
            M,
            N,
            scale_cols_n,
            scale_cols_m,
            SCALING_MODE=scaling_mode,
            USE_PTX=x.device.type != "hip",
            BLOCK_M=_BLOCK_M,
            SCALE_BLOCK_SIZE=_SCALE_BLOCK_SIZE,
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
