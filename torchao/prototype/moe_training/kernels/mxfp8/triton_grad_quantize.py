# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP8 quantization of ``grad_out`` into both row-major and
transposed variants, with e8m0 scales emitted directly into the tcgen05
blocked layout.

The MoE backward pass needs two MXFP8 operands of ``grad_out``:

  * ``dgrad = grad_out @ weight``    -> MXFP8(``grad_out``) with rowwise
    (along-N) scales, ``(total_M, N)`` row-major.
  * ``wgrad = grad_out.T @ x``       -> MXFP8(``grad_out.T``) with
    rowwise scales along the new reduction axis (which is M),
    ``(N, total_M)`` row-major.

Today the production path takes two separate Triton quant passes plus
two separate blocked-layout rearrangement kernels (4 kernels total), so
the bf16 ``grad_out`` is read twice and every scale tensor is written
and then read+rewritten in the rearrange pass.

This module implements two direct alternatives and dispatches between
them by problem size:

* **Fused single-CTA path** (``_dim0_dim1_fused_kernel``). Reads bf16
  once and emits all four outputs from the same CTA (two fp8 tensors +
  two tcgen05 blocked scale tensors). Minimises HBM traffic (~130 MB on
  16k x 2k) and launch overhead, so it wins on small shapes.

* **Split two-stream path** (``_dim0_blocked_kernel`` +
  ``_dim1_blocked_kernel`` on separate CUDA streams). Each kernel has
  its own register budget like production's standalone dim0/dim1
  kernels, so occupancy stays high despite the doubled bf16 read; the
  two kernels then overlap on the HBM controller. Wins on large shapes
  where the overlap outweighs the extra bf16 read.

Both paths write scales directly in the tcgen05 ``(128, 4)`` blocked
layout so no downstream rearrange kernel is needed.

Layout contract (v1):

  * ``inner_block_size = 32`` and both ``BLOCK_M`` and ``BLOCK_N`` are
    multiples of 128, so each CTA emits an integer number of
    ``(128, 4)`` blocked scale super-tiles per output.
  * ``M % 128 == 0`` and ``N % 128 == 0`` required in v1.
"""

from typing import Tuple

import torch
from torch import Tensor
from torch.utils._triton import has_triton

from torchao.utils import ceil_div, torch_version_at_least

_SCALE_BLOCK_SIZE = 32
_SCALE_SUPERTILE_ROWS = 128
_SCALE_SUPERTILE_COLS = 4

# Empirically, on B200 the 2-stream split wins when the per-direction HBM
# traffic is large enough that the stream overlap > extra bf16 read cost
# + 2x kernel launch overhead. Below this threshold the fused single-CTA
# path wins.
_SPLIT_NUMEL_THRESHOLD = 32 * 1024 * 1024


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    from torchao.prototype.mx_formats.kernels import (
        _triton_calculate_scale_floor,
        _triton_calculate_scale_rceil,
    )

    def _tile_autotune_configs():
        # Mirror the production dim0/dim1 autotune sweep
        # (torchao/prototype/mx_formats/kernels.py::_get_mxfp8_quant_autotune_configs).
        # 512 COL_TILE is skipped per the known triton bug #3362 on mx_formats.
        configs = []
        tile_shapes = [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
            (512, 128),
            (512, 256),
        ]
        for BM, BN in tile_shapes:
            for num_warps in (4, 8):
                for num_stages in (2, 3, 4):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
        return configs

    def _prune_tile_configs(configs, named_args, **kwargs):
        M = named_args["M"]
        N = named_args["N"]
        kept = [
            c
            for c in configs
            if c.kwargs["BLOCK_M"] <= M and c.kwargs["BLOCK_N"] <= N
        ]
        return kept if kept else configs

    # --------------------------------------------------------------------
    # Fused path: single CTA emits all 4 outputs (wins on small shapes).
    # --------------------------------------------------------------------
    @triton.autotune(
        configs=_tile_autotune_configs(),
        key=["M", "N", "SCALING_MODE"],
        prune_configs_by={"early_config_prune": _prune_tile_configs},
    )
    @triton.jit
    def _dim0_dim1_fused_kernel(
        x_ptr,  # (M, N) bf16 row-major input
        qdata_dim0_ptr,  # (M, N) e4m3 row-major out (rowwise scales)
        qdata_dim1_t_ptr,  # (N, M) e4m3 row-major out (colwise scales; transpose of x)
        scales_dim0_ptr,  # flat (M * padded_scale_cols_n,) e8m0 tcgen05-blocked
        scales_dim1_ptr,  # flat (N * padded_scale_cols_m,) e8m0 tcgen05-blocked
        M,
        N,
        padded_scale_cols_n,
        padded_scale_cols_m,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
    ):
        """Single-pass fused MXFP8 quantize of a ``(BLOCK_M, BLOCK_N)`` bf16 tile.

        Per-CTA memory traffic:
            read  : BLOCK_M * BLOCK_N bf16 (coalesced (M, N) load)
            write : BLOCK_M * BLOCK_N fp8 dim0   (coalesced (M, N) store)
                    BLOCK_M * BLOCK_N fp8 dim1_t (transposed (N, M) store)
                    scale bytes (blocked-layout stores, both outputs)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x_offs = m_offs[:, None].to(tl.int64) * N + n_offs[None, :]
        x_mn = tl.load(x_ptr + x_offs)

        # --- dim0 (rowwise, along N) ----------------------------------
        x_r0 = x_mn.reshape(
            BLOCK_M * (BLOCK_N // SCALE_BLOCK_SIZE), SCALE_BLOCK_SIZE
        )
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
        fp8_d0 = tl.reshape(scaled_r0, BLOCK_M, BLOCK_N).to(tl.float8e4nv)
        tl.store(qdata_dim0_ptr + x_offs, fp8_d0)

        # --- dim1 (colwise, along M via single bf16 transpose) --------
        x_nm = tl.trans(x_mn)
        x_r1 = x_nm.reshape(
            BLOCK_N * (BLOCK_M // SCALE_BLOCK_SIZE), SCALE_BLOCK_SIZE
        )
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
        fp8_d1 = tl.reshape(scaled_r1, BLOCK_N, BLOCK_M).to(tl.float8e4nv)
        out_offs_dim1 = n_offs[:, None].to(tl.int64) * M + m_offs[None, :]
        tl.store(qdata_dim1_t_ptr + out_offs_dim1, fp8_d1)

        # --- blocked scale stores -------------------------------------
        # Each CTA covers BLOCK_M/128 * BLOCK_N/128 blocked super-tiles
        # per output with the (128, 4) tcgen05 swizzle
        #   dest_within_supertile = (r % 32) * 16 + (r // 32) * 4 + c
        # inside each super-tile. The fused formula below handles the
        # multi-super-tile case by adding per-super-tile base offsets.
        SCALE_N_COLS: tl.constexpr = BLOCK_N // SCALE_BLOCK_SIZE
        scale0_2d = scale_e8m0_r0.reshape(BLOCK_M, SCALE_N_COLS)
        tile_row0 = tl.arange(0, BLOCK_M)[:, None]
        tile_col0 = tl.arange(0, SCALE_N_COLS)[None, :]
        r_global0 = pid_m * BLOCK_M + tile_row0
        c_global0 = pid_n * SCALE_N_COLS + tile_col0
        dest0 = (
            (r_global0 // 128) * (128 * padded_scale_cols_n)
            + (c_global0 // 4) * (128 * 4)
            + ((r_global0 % 128) % 32) * 16
            + ((r_global0 % 128) // 32) * 4
            + (c_global0 % 4)
        )
        tl.store(scales_dim0_ptr + dest0, scale0_2d)

        SCALE_M_COLS: tl.constexpr = BLOCK_M // SCALE_BLOCK_SIZE
        scale1_2d = scale_e8m0_r1.reshape(BLOCK_N, SCALE_M_COLS)
        tile_row1 = tl.arange(0, BLOCK_N)[:, None]
        tile_col1 = tl.arange(0, SCALE_M_COLS)[None, :]
        r_global1 = pid_n * BLOCK_N + tile_row1
        c_global1 = pid_m * SCALE_M_COLS + tile_col1
        dest1 = (
            (r_global1 // 128) * (128 * padded_scale_cols_m)
            + (c_global1 // 4) * (128 * 4)
            + ((r_global1 % 128) % 32) * 16
            + ((r_global1 % 128) // 32) * 4
            + (c_global1 % 4)
        )
        tl.store(scales_dim1_ptr + dest1, scale1_2d)

    # --------------------------------------------------------------------
    # Split two-stream path (wins on large shapes).
    # --------------------------------------------------------------------
    @triton.autotune(
        configs=_tile_autotune_configs(),
        key=["M", "N", "SCALING_MODE"],
        prune_configs_by={"early_config_prune": _prune_tile_configs},
    )
    @triton.jit
    def _dim0_blocked_kernel(
        x_ptr,  # (M, N) bf16 row-major input
        qdata_dim0_ptr,  # (M, N) e4m3 row-major out
        scales_dim0_ptr,  # flat (M * padded_scale_cols_n,) e8m0 tcgen05-blocked
        M,
        N,
        padded_scale_cols_n,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
    ):
        """Rowwise (along-N) MXFP8 quantize, coalesced ``(M, N)`` stores.

        Scales are written directly into the tcgen05 ``(128, 4)`` blocked
        layout so no downstream rearrange kernel is needed.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x_offs = m_offs[:, None].to(tl.int64) * N + n_offs[None, :]
        x_mn = tl.load(x_ptr + x_offs)

        x_r = x_mn.reshape(BLOCK_M * (BLOCK_N // SCALE_BLOCK_SIZE), SCALE_BLOCK_SIZE)
        x_abs = tl.abs(x_r)
        if SCALING_MODE == "rceil":
            descale, scale_e8m0 = _triton_calculate_scale_rceil(
                x_abs, axis=1, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale, scale_e8m0 = _triton_calculate_scale_floor(x_abs, axis=1)

        scaled = x_r * descale[:, None]
        fp8_out = tl.reshape(scaled, BLOCK_M, BLOCK_N).to(tl.float8e4nv)
        tl.store(qdata_dim0_ptr + x_offs, fp8_out)

        SCALE_N_COLS: tl.constexpr = BLOCK_N // SCALE_BLOCK_SIZE
        scale_2d = scale_e8m0.reshape(BLOCK_M, SCALE_N_COLS)
        tile_row = tl.arange(0, BLOCK_M)[:, None]
        tile_col = tl.arange(0, SCALE_N_COLS)[None, :]
        r_global = pid_m * BLOCK_M + tile_row
        c_global = pid_n * SCALE_N_COLS + tile_col
        dest = (
            (r_global // 128) * (128 * padded_scale_cols_n)
            + (c_global // 4) * (128 * 4)
            + ((r_global % 128) % 32) * 16
            + ((r_global % 128) // 32) * 4
            + (c_global % 4)
        )
        tl.store(scales_dim0_ptr + dest, scale_2d)

    @triton.autotune(
        configs=_tile_autotune_configs(),
        key=["M", "N", "SCALING_MODE"],
        prune_configs_by={"early_config_prune": _prune_tile_configs},
    )
    @triton.jit
    def _dim1_blocked_kernel(
        x_ptr,  # (M, N) bf16 row-major input
        qdata_dim1_t_ptr,  # (N, M) e4m3 row-major out (transpose of x)
        scales_dim1_ptr,  # flat (N * padded_scale_cols_m,) e8m0 tcgen05-blocked
        M,
        N,
        padded_scale_cols_m,
        SCALING_MODE: tl.constexpr,
        USE_PTX: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
    ):
        """Colwise (along-M) MXFP8 quantize, transposed ``(N, M)`` stores.

        Uses the same in-register bf16 transpose as the production
        ``to_mxfp8_dim1_kernel`` for maximum HBM utilization on the
        scattered row stores.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        x_offs = m_offs[:, None].to(tl.int64) * N + n_offs[None, :]
        x_mn = tl.load(x_ptr + x_offs)

        x_nm = tl.trans(x_mn)
        x_r = x_nm.reshape(BLOCK_N * (BLOCK_M // SCALE_BLOCK_SIZE), SCALE_BLOCK_SIZE)
        x_abs = tl.abs(x_r)
        if SCALING_MODE == "rceil":
            descale, scale_e8m0 = _triton_calculate_scale_rceil(
                x_abs, axis=1, USE_PTX=USE_PTX
            )
        else:
            tl.static_assert(SCALING_MODE == "floor")
            descale, scale_e8m0 = _triton_calculate_scale_floor(x_abs, axis=1)

        scaled = x_r * descale[:, None]
        fp8_out = tl.reshape(scaled, BLOCK_N, BLOCK_M).to(tl.float8e4nv)
        out_offs = n_offs[:, None].to(tl.int64) * M + m_offs[None, :]
        tl.store(qdata_dim1_t_ptr + out_offs, fp8_out)

        SCALE_M_COLS: tl.constexpr = BLOCK_M // SCALE_BLOCK_SIZE
        scale_2d = scale_e8m0.reshape(BLOCK_N, SCALE_M_COLS)
        tile_row = tl.arange(0, BLOCK_N)[:, None]
        tile_col = tl.arange(0, SCALE_M_COLS)[None, :]
        r_global = pid_n * BLOCK_N + tile_row
        c_global = pid_m * SCALE_M_COLS + tile_col
        dest = (
            (r_global // 128) * (128 * padded_scale_cols_m)
            + (c_global // 4) * (128 * 4)
            + ((r_global % 128) % 32) * 16
            + ((r_global % 128) // 32) * 4
            + (c_global % 4)
        )
        tl.store(scales_dim1_ptr + dest, scale_2d)

    def triton_mxfp8_quantize_dim0_dim1(
        x: Tensor,
        scaling_mode: str = "rceil",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dim0 + dim1 MXFP8 quantization of ``x`` with tcgen05 blocked scales.

        Drop-in replacement for the sequence

            qdata0, scales0 = triton_to_mxfp8_dim0(x, 32, mode)
            qdata1_t, scales1 = triton_to_mxfp8_dim1(x, 32, mode)
            scales0_blocked = triton_mx_block_rearrange(scales0)
            scales1_blocked = triton_mx_block_rearrange(scales1)

        Dispatches between two Triton implementations by problem size:

        * **Fused** (``M * N < ~32M elements``): one CTA reads bf16 and
          emits all four outputs. Minimises HBM traffic and launch
          overhead. Wins on small shapes.
        * **Split two-stream**: a dedicated rowwise kernel and a
          dedicated colwise kernel launched on separate CUDA streams, so
          they overlap on the HBM controller. Each kernel has its own
          register budget like production's standalone dim0/dim1
          kernels. Wins on large shapes.

        Both paths write scales directly in the tcgen05 blocked layout,
        so no rearrange kernel is needed.

        Args:
            x: bfloat16 tensor of shape ``(M, N)``, row-major.
                ``M`` and ``N`` must both be multiples of 128 in v1.
            scaling_mode: ``"rceil"`` (default) or ``"floor"``.

        Returns:
            ``qdata_dim0``: ``float8_e4m3fn`` tensor of shape ``(M, N)``,
                row-major. Scales live along dim 1.
            ``qdata_dim1_t``: ``float8_e4m3fn`` tensor of shape ``(N, M)``,
                row-major. Scales live along dim 1 (= along M of the
                original input).
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
        assert M % 128 == 0, f"M must be a multiple of 128, got M={M}"
        assert N % 128 == 0, f"N must be a multiple of 128, got N={N}"

        padded_scale_cols_n = (
            ceil_div(N // _SCALE_BLOCK_SIZE, _SCALE_SUPERTILE_COLS)
            * _SCALE_SUPERTILE_COLS
        )
        padded_scale_cols_m = (
            ceil_div(M // _SCALE_BLOCK_SIZE, _SCALE_SUPERTILE_COLS)
            * _SCALE_SUPERTILE_COLS
        )

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

        use_ptx = x.device.type != "hip"
        use_split = (M * N) >= _SPLIT_NUMEL_THRESHOLD

        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

        if use_split:
            # Launch both direction kernels on separate CUDA streams so
            # they overlap on the HBM controller. Each stream waits on
            # the current stream's allocations, then the current stream
            # joins both before returning.
            cur_stream = torch.cuda.current_stream(device=x.device)
            stream_dim0 = torch.cuda.Stream(device=x.device)
            stream_dim1 = torch.cuda.Stream(device=x.device)
            stream_dim0.wait_stream(cur_stream)
            stream_dim1.wait_stream(cur_stream)

            with torch.cuda.stream(stream_dim0):
                _dim0_blocked_kernel[grid](
                    x,
                    qdata_dim0,
                    scales_dim0_blocked_u8,
                    M,
                    N,
                    padded_scale_cols_n,
                    SCALING_MODE=scaling_mode,
                    USE_PTX=use_ptx,
                    SCALE_BLOCK_SIZE=_SCALE_BLOCK_SIZE,
                )

            with torch.cuda.stream(stream_dim1):
                _dim1_blocked_kernel[grid](
                    x,
                    qdata_dim1_t,
                    scales_dim1_blocked_u8,
                    M,
                    N,
                    padded_scale_cols_m,
                    SCALING_MODE=scaling_mode,
                    USE_PTX=use_ptx,
                    SCALE_BLOCK_SIZE=_SCALE_BLOCK_SIZE,
                )

            cur_stream.wait_stream(stream_dim0)
            cur_stream.wait_stream(stream_dim1)
        else:
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
                USE_PTX=use_ptx,
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
