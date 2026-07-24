# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unified FP4 (NVFP4 + MXFP4) + optional RHT CuTeDSL quantize cast.

A single no-smem streaming kernel that supersedes the separate ``nvfp4_rht``
and ``mxfp4_rht`` maxbw casts: it serves both FP4 formats and all three scale
layouts the GEMM consumers use, with optional fused RHT.

* ``fmt="nvfp4"`` -- block 16, two-level E4M3 block scale + per-tensor global
  scale (``float8_e4m3fn`` scales). Supports arbitrary ``K % 16`` (a masked
  remainder handles an odd number of 16-blocks).
* ``fmt="mxfp4"`` -- block 32, single-level E8M0 block scale
  (``float8_e8m0fnu`` scales), ``"floor"`` or ``"rceil"``. Requires ``K % 32``.
* ``scale_layout in {"linear", "cublas_blocked", "mma_tiled"}`` is selected at
  compile time. ``cublas_blocked`` is the to_blocked padded swizzle consumed by
  the f4f4bf16 GEMM; ``mma_tiled`` is the SM100 blockscaled-GEMM atom layout
  (no separate scale-conversion pass); ``linear`` is the plain ``(M, K//blk)``.
* optional fused RHT (register FWHT16/32 + sign) is applied per block before
  amax / scale / pack; an empty ``sign_vector`` skips it via a compile-time
  constexpr (no FWHT overhead on the plain path).

A "group" is 32 input elements = one 128-bit store = two NVFP4 blocks or one
MXFP4 block; the per-format scale recipe, FWHT size, and MMA row-atom are
compile-time ``FORMAT``-selected so a single kernel body covers both formats.

Two byte-identical thread mappings are exposed via ``mapping=``:
* ``"striped"`` -- threads stripe a row's groups; grid-strided rows. Best at
  very large N.
* ``"wpr"`` -- warp-per-row: warp ``w`` owns contiguous row ``bidy*WARPS+w``,
  with the 32 lanes + a ``grid.x`` column split + ILP covering the columns
  (replicates the dense-GEMM grid). Best at small / mid N; requires ``K % 32``.

Gated behind a Blackwell (SM 10.x) GPU, CUDA >= 12.8, and the CuTeDSL runtime
(see ``cutedsl/__init__.py``). Output is byte-exact vs the eager torchao FP4
casts (and vs the per-format ``{nvfp4,mxfp4}_rht`` CuTeDSL kernels).
"""

from typing import Optional, Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    _cvt_rn_satfinite_e2m1x2_f32,
    compute_amax,
    compute_nvfp4_scale_e4m3,
    compute_scale_byte_fp4,
)
from .fwht import fwht16_sign, fwht32_sign

# NVFP4 two-level global-scale numerator: F8E4M3_MAX * F4_E2M1_MAX = 448 * 6.
_GLOBAL_SCALE_NUMERATOR = 2688.0
_LAYOUTS = {"linear": 0, "cublas_blocked": 1, "mma_tiled": 2}
_E8M0_NEUTRAL = 127  # 2**0
_E4M3_NEUTRAL = 0x38  # 1.0

# Compiled-launcher + per-shape kernel caches (populated lazily).
_LAUNCH_CACHE = {}
_JIT_CACHE = {}


def _get_launches():
    """Define and cache the (uncompiled) striped + warp-per-row ``@cute.jit``
    launchers. ``cutlass`` is imported here (not at module scope) so importing
    this module is safe without the CuTeDSL runtime."""
    if "v" in _LAUNCH_CACHE:
        return _LAUNCH_CACHE["v"]

    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu as nv

    def _scale_offset(row, kb, LAYOUT, ATOM_M0, pad_cols, rest_m, rest_k, K, NBLK):
        if LAYOUT == 2:  # mma_tiled (ATOM_M0=128 NVFP4 / 32 MXFP4)
            atom_m0 = row % cutlass.Int32(ATOM_M0)
            atom_m1 = (row // cutlass.Int32(ATOM_M0)) % cutlass.Int32(4)
            rest_m_idx = row // cutlass.Int32(ATOM_M0 * 4)
            atom_k = kb % cutlass.Int32(4)
            rest_k_idx = kb // cutlass.Int32(4)
            stride_rest_m = rest_k * cutlass.Int32(4)
            stride_m1 = stride_rest_m * rest_m
            stride_m0 = stride_m1 * cutlass.Int32(4)
            return (
                atom_m0 * stride_m0
                + atom_m1 * stride_m1
                + rest_m_idx * stride_rest_m
                + atom_k * rest_k
                + rest_k_idx
            )
        elif LAYOUT == 1:  # cublas_blocked (format-independent)
            r128 = row // cutlass.Int32(128)
            r32 = row % cutlass.Int32(32)
            r32_4 = (row // cutlass.Int32(32)) % cutlass.Int32(4)
            kb4 = kb // cutlass.Int32(4)
            kbm = kb % cutlass.Int32(4)
            return (
                r128 * cutlass.Int32(128) * pad_cols
                + kb4 * cutlass.Int32(512)
                + r32 * cutlass.Int32(16)
                + r32_4 * cutlass.Int32(4)
                + kbm
            )
        else:  # linear
            return row * (K // cutlass.Int32(NBLK)) + kb

    @cute.kernel
    def _striped_kernel(  # noqa: C901
        gx: cute.Tensor,
        gq: cute.Tensor,
        gscale: cute.Tensor,
        gsign: cute.Tensor,
        M: cutlass.Int32,
        K: cutlass.Int32,
        GPR: cutlass.Int32,
        pad_cols: cutlass.Int32,
        rest_m: cutlass.Int32,
        rest_k: cutlass.Int32,
        qstride: cutlass.Int32,
        global_scale: cutlass.Float32,
        ILP: cutlass.Constexpr[int],
        APPLY_RHT: cutlass.Constexpr[bool],
        LAYOUT: cutlass.Constexpr[int],
        FORMAT: cutlass.Constexpr[int],
        USE_RCEIL: cutlass.Constexpr[bool],
        LDWIDTH: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy_init, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        gid = bidx * bdim + tidx
        nthreads_x = cute.arch.grid_dim()[0] * bdim
        NBLK = 16 if FORMAT == 0 else 32
        NB = 32 // NBLK
        HALF = NBLK // 2
        ATOM_M0 = 128 if FORMAT == 0 else 32
        in_dt = gx.element_type
        ld = cute.make_copy_atom(nv.CopyUniversalOp(), in_dt, num_bits_per_copy=128)
        st = cute.make_copy_atom(
            nv.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=128
        )
        st64 = cute.make_copy_atom(
            nv.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
        )
        NLD = 32 // LDWIDTH

        sign_reg = cute.make_rmem_tensor((NBLK,), cutlass.Float32)
        if cutlass.const_expr(APPLY_RHT):
            for j in cutlass.range_constexpr(NBLK):
                sign_reg[j] = cutlass.Float32(gsign[j])

        row = bidy_init
        while row < M:
            base = gid
            while base < GPR:
                fragbuf = cute.make_rmem_tensor((ILP * 32,), in_dt)
                for jj in cutlass.range_constexpr(ILP):
                    gc = base + jj * nthreads_x
                    if gc < GPR:
                        off = cute.assume(row * K + gc * cutlass.Int32(32), divby=32)
                        for w in cutlass.range_constexpr(NLD):
                            s = cute.make_tensor(
                                gx.iterator + off + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            f = cute.make_tensor(
                                fragbuf.iterator + jj * 32 + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            cute.copy(ld, s, f)
                for jj in cutlass.range_constexpr(ILP):
                    gc = base + jj * nthreads_x
                    if gc < GPR:
                        vals = cute.make_rmem_tensor((32,), cutlass.Float32)
                        for i in cutlass.range_constexpr(32):
                            vals[i] = cutlass.Float32(fragbuf[jj * 32 + i])
                        packed = cute.make_rmem_tensor((16,), cutlass.Uint8)
                        for b in cutlass.range_constexpr(NB):
                            blk = cute.make_tensor(
                                vals.iterator + b * NBLK,
                                cute.make_layout((NBLK,), stride=(1,)),
                            )
                            if cutlass.const_expr(APPLY_RHT):
                                if cutlass.const_expr(FORMAT == 0):
                                    fwht16_sign(blk, sign_reg)
                                else:
                                    fwht32_sign(blk, sign_reg)
                            amax = compute_amax(blk)
                            if cutlass.const_expr(FORMAT == 0):
                                scb, inv = compute_nvfp4_scale_e4m3(amax, global_scale)
                            else:
                                sbi, inv = compute_scale_byte_fp4(amax, USE_RCEIL)
                                scb = cutlass.Uint8(sbi & cutlass.Int32(0xFF))
                            kb = gc * cutlass.Int32(NB) + b
                            gscale[
                                _scale_offset(row, kb, LAYOUT, ATOM_M0, pad_cols,
                                              rest_m, rest_k, K, NBLK)
                            ] = scb
                            for p in cutlass.range_constexpr(HALF):
                                lo = blk[2 * p] * inv
                                hi = blk[2 * p + 1] * inv
                                if cutlass.const_expr(FORMAT == 1 and not USE_RCEIL):
                                    if lo > cutlass.Float32(6.0):
                                        lo = cutlass.Float32(6.0)
                                    if lo < cutlass.Float32(-6.0):
                                        lo = cutlass.Float32(-6.0)
                                    if hi > cutlass.Float32(6.0):
                                        hi = cutlass.Float32(6.0)
                                    if hi < cutlass.Float32(-6.0):
                                        hi = cutlass.Float32(-6.0)
                                packed[b * HALF + p] = _cvt_rn_satfinite_e2m1x2_f32(
                                    hi, lo
                                )
                        offq = cute.assume(
                            row * qstride + gc * cutlass.Int32(16), divby=16
                        )
                        d = cute.make_tensor(
                            gq.iterator + offq, cute.make_layout((16,), stride=(1,))
                        )
                        cute.copy(st, packed, d)
                base = base + nthreads_x * cutlass.Int32(ILP)
            # masked remainder (NVFP4 only): leftover 16-block when k_blocks odd
            if cutlass.const_expr(FORMAT == 0):
                rem = K // cutlass.Int32(16) - cutlass.Int32(2) * GPR
                if gid == cutlass.Int32(0):
                    if rem > cutlass.Int32(0):
                        kb = cutlass.Int32(2) * GPR
                        offr = cute.assume(row * K + kb * cutlass.Int32(16), divby=16)
                        rbuf = cute.make_rmem_tensor((16,), in_dt)
                        for w in cutlass.range_constexpr(16 // LDWIDTH):
                            s = cute.make_tensor(
                                gx.iterator + offr + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            f = cute.make_tensor(
                                rbuf.iterator + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            cute.copy(ld, s, f)
                        blkv = cute.make_rmem_tensor((16,), cutlass.Float32)
                        for i in cutlass.range_constexpr(16):
                            blkv[i] = cutlass.Float32(rbuf[i])
                        if cutlass.const_expr(APPLY_RHT):
                            fwht16_sign(blkv, sign_reg)
                        amax = compute_amax(blkv)
                        e4m3, inv = compute_nvfp4_scale_e4m3(amax, global_scale)
                        gscale[
                            _scale_offset(row, kb, LAYOUT, 128, pad_cols, rest_m,
                                          rest_k, K, 16)
                        ] = e4m3
                        rpacked = cute.make_rmem_tensor((8,), cutlass.Uint8)
                        for p in cutlass.range_constexpr(8):
                            lo = blkv[2 * p] * inv
                            hi = blkv[2 * p + 1] * inv
                            rpacked[p] = _cvt_rn_satfinite_e2m1x2_f32(hi, lo)
                        offrq = cute.assume(
                            row * qstride + kb * cutlass.Int32(8), divby=8
                        )
                        dr = cute.make_tensor(
                            gq.iterator + offrq, cute.make_layout((8,), stride=(1,))
                        )
                        cute.copy(st64, rpacked, dr)
            row = row + cute.arch.grid_dim()[1]

    @cute.jit
    def _striped_launch(
        gx, gq, gscale, gsign, M, K, GPR, pad_cols, rest_m, rest_k, qstride, gs,
        threads, ncta_x, ncta_y, stream,
        ILP: cutlass.Constexpr[int],
        APPLY_RHT: cutlass.Constexpr[bool],
        LAYOUT: cutlass.Constexpr[int],
        FORMAT: cutlass.Constexpr[int],
        USE_RCEIL: cutlass.Constexpr[bool],
        LDWIDTH: cutlass.Constexpr[int],
    ):
        _striped_kernel(
            gx, gq, gscale, gsign, M, K, GPR, pad_cols, rest_m, rest_k, qstride, gs,
            ILP, APPLY_RHT, LAYOUT, FORMAT, USE_RCEIL, LDWIDTH,
        ).launch(
            grid=(ncta_x, ncta_y, 1),
            block=(threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _wpr_kernel(  # noqa: C901
        gx: cute.Tensor,
        gq: cute.Tensor,
        gscale: cute.Tensor,
        gsign: cute.Tensor,
        M: cutlass.Int32,
        K: cutlass.Int32,
        GPR: cutlass.Int32,
        pad_cols: cutlass.Int32,
        rest_m: cutlass.Int32,
        rest_k: cutlass.Int32,
        qstride: cutlass.Int32,
        global_scale: cutlass.Float32,
        ILP: cutlass.Constexpr[int],
        APPLY_RHT: cutlass.Constexpr[bool],
        LAYOUT: cutlass.Constexpr[int],
        FORMAT: cutlass.Constexpr[int],
        USE_RCEIL: cutlass.Constexpr[bool],
        WARPS: cutlass.Constexpr[int],
        LDWIDTH: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp_id = tidx // cutlass.Int32(32)
        lane = tidx % cutlass.Int32(32)
        NBLK = 16 if FORMAT == 0 else 32
        NB = 32 // NBLK
        HALF = NBLK // 2
        ATOM_M0 = 128 if FORMAT == 0 else 32
        in_dt = gx.element_type
        ld = cute.make_copy_atom(nv.CopyUniversalOp(), in_dt, num_bits_per_copy=128)
        st = cute.make_copy_atom(
            nv.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=128
        )
        NLD = 32 // LDWIDTH
        sign_reg = cute.make_rmem_tensor((NBLK,), cutlass.Float32)
        if cutlass.const_expr(APPLY_RHT):
            for j in cutlass.range_constexpr(NBLK):
                sign_reg[j] = cutlass.Float32(gsign[j])

        row = bidy * cutlass.Int32(WARPS) + warp_id
        if row < M:
            nthreads_row = cute.arch.grid_dim()[0] * cutlass.Int32(32)
            base = bidx * cutlass.Int32(32) + lane
            while base < GPR:
                fragbuf = cute.make_rmem_tensor((ILP * 32,), in_dt)
                for jj in cutlass.range_constexpr(ILP):
                    gc = base + jj * nthreads_row
                    if gc < GPR:
                        off = cute.assume(row * K + gc * cutlass.Int32(32), divby=32)
                        for w in cutlass.range_constexpr(NLD):
                            s = cute.make_tensor(
                                gx.iterator + off + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            f = cute.make_tensor(
                                fragbuf.iterator + jj * 32 + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            cute.copy(ld, s, f)
                for jj in cutlass.range_constexpr(ILP):
                    gc = base + jj * nthreads_row
                    if gc < GPR:
                        vals = cute.make_rmem_tensor((32,), cutlass.Float32)
                        for i in cutlass.range_constexpr(32):
                            vals[i] = cutlass.Float32(fragbuf[jj * 32 + i])
                        packed = cute.make_rmem_tensor((16,), cutlass.Uint8)
                        for b in cutlass.range_constexpr(NB):
                            blk = cute.make_tensor(
                                vals.iterator + b * NBLK,
                                cute.make_layout((NBLK,), stride=(1,)),
                            )
                            if cutlass.const_expr(APPLY_RHT):
                                if cutlass.const_expr(FORMAT == 0):
                                    fwht16_sign(blk, sign_reg)
                                else:
                                    fwht32_sign(blk, sign_reg)
                            amax = compute_amax(blk)
                            if cutlass.const_expr(FORMAT == 0):
                                scb, inv = compute_nvfp4_scale_e4m3(amax, global_scale)
                            else:
                                sbi, inv = compute_scale_byte_fp4(amax, USE_RCEIL)
                                scb = cutlass.Uint8(sbi & cutlass.Int32(0xFF))
                            kb = gc * cutlass.Int32(NB) + b
                            gscale[
                                _scale_offset(row, kb, LAYOUT, ATOM_M0, pad_cols,
                                              rest_m, rest_k, K, NBLK)
                            ] = scb
                            for p in cutlass.range_constexpr(HALF):
                                lo = blk[2 * p] * inv
                                hi = blk[2 * p + 1] * inv
                                if cutlass.const_expr(FORMAT == 1 and not USE_RCEIL):
                                    if lo > cutlass.Float32(6.0):
                                        lo = cutlass.Float32(6.0)
                                    if lo < cutlass.Float32(-6.0):
                                        lo = cutlass.Float32(-6.0)
                                    if hi > cutlass.Float32(6.0):
                                        hi = cutlass.Float32(6.0)
                                    if hi < cutlass.Float32(-6.0):
                                        hi = cutlass.Float32(-6.0)
                                packed[b * HALF + p] = _cvt_rn_satfinite_e2m1x2_f32(
                                    hi, lo
                                )
                        offq = cute.assume(
                            row * qstride + gc * cutlass.Int32(16), divby=16
                        )
                        d = cute.make_tensor(
                            gq.iterator + offq, cute.make_layout((16,), stride=(1,))
                        )
                        cute.copy(st, packed, d)
                base = base + nthreads_row * cutlass.Int32(ILP)

    @cute.jit
    def _wpr_launch(
        gx, gq, gscale, gsign, M, K, GPR, pad_cols, rest_m, rest_k, qstride, gs,
        block_threads, ncta_x, ncta_y, stream,
        ILP: cutlass.Constexpr[int],
        APPLY_RHT: cutlass.Constexpr[bool],
        LAYOUT: cutlass.Constexpr[int],
        FORMAT: cutlass.Constexpr[int],
        USE_RCEIL: cutlass.Constexpr[bool],
        WARPS: cutlass.Constexpr[int],
        LDWIDTH: cutlass.Constexpr[int],
    ):
        _wpr_kernel(
            gx, gq, gscale, gsign, M, K, GPR, pad_cols, rest_m, rest_k, qstride, gs,
            ILP, APPLY_RHT, LAYOUT, FORMAT, USE_RCEIL, WARPS, LDWIDTH,
        ).launch(
            grid=(ncta_x, ncta_y, 1),
            block=(block_threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    _LAUNCH_CACHE["v"] = (_striped_launch, _wpr_launch)
    return _LAUNCH_CACHE["v"]


def _fp4_quantize_unified_impl(
    x: torch.Tensor,
    sign_vector: Optional[list] = None,
    fmt: str = "nvfp4",
    scaling_mode: str = "floor",
    scale_layout: str = "cublas_blocked",
    global_scale: Optional[float] = None,
    mapping: str = "wpr",
    threads: int = 128,
    ilp: int = 2,
    rows_per_cta: int = 1,
    warps: int = 4,
    xsplit: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.float32, torch.bfloat16)
    M, K = x.shape
    FORMAT = 0 if fmt == "nvfp4" else 1
    BLK = 16 if FORMAT == 0 else 32
    ATOM_M0 = 128 if FORMAT == 0 else 32
    USE_RCEIL = scaling_mode.lower() == "rceil"
    L = _LAYOUTS[scale_layout]
    if FORMAT == 0:
        assert K % 16 == 0, "NVFP4 requires K % 16 == 0"
    else:
        assert K % 32 == 0, "MXFP4 requires K % 32 == 0"
    k_blocks = K // BLK
    apply_rht = sign_vector is not None and len(sign_vector) > 0
    if apply_rht:
        assert len(sign_vector) == BLK, f"sign_vector must have length {BLK}"

    gs_val = global_scale
    if FORMAT == 0 and gs_val is None:
        gs_val = _GLOBAL_SCALE_NUMERATOR / x.abs().max().item()
    if gs_val is None:
        gs_val = 1.0

    qstride = ceil_div(K // 2, 16) * 16
    q_data = torch.empty((M, qstride), device=x.device, dtype=torch.uint8)
    pad_rows = ceil_div(M, 128) * 128
    pad_cols = ceil_div(k_blocks, 4) * 4
    rest_m = ceil_div(M, ATOM_M0 * 4)
    rest_k = ceil_div(k_blocks, 4)
    neutral = _E4M3_NEUTRAL if FORMAT == 0 else _E8M0_NEUTRAL
    if L == 0:
        scales_u8 = torch.empty((M, k_blocks), device=x.device, dtype=torch.uint8)
    elif L == 1:
        scales_u8 = torch.full(
            (pad_rows * pad_cols,), neutral, device=x.device, dtype=torch.uint8
        )
    else:
        scales_u8 = torch.full(
            (ATOM_M0 * 16 * rest_m * rest_k,), neutral, device=x.device,
            dtype=torch.uint8,
        )

    sign_src = sign_vector if apply_rht else [0] * BLK
    sign_dev = torch.tensor(
        [int(s) for s in sign_src], device=x.device, dtype=torch.int32
    )

    striped_launch, wpr_launch = _get_launches()
    GPR = K // 32
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    ldwidth = 4 if x.dtype == torch.float32 else 8
    common = (
        from_dlpack(x.view(-1), assumed_align=16),
        from_dlpack(q_data.view(-1), assumed_align=16),
        from_dlpack(scales_u8.view(-1), assumed_align=16),
        from_dlpack(sign_dev, assumed_align=16),
        cutlass.Int32(M), cutlass.Int32(K), cutlass.Int32(GPR),
        cutlass.Int32(pad_cols), cutlass.Int32(rest_m), cutlass.Int32(rest_k),
        cutlass.Int32(qstride), cutlass.Float32(gs_val),
    )

    if mapping == "wpr":
        assert K % 32 == 0, "wpr mapping requires K % 32 == 0"
        block_threads = 32 * warps
        ncta_y = ceil_div(M, warps)
        wargs = common + (block_threads, xsplit, ncta_y, stream)
        key = (str(x.dtype), apply_rht, L, FORMAT, USE_RCEIL, "wpr", warps, ilp)
        compiled = _JIT_CACHE.get(key)
        if compiled is None:
            import cutlass.cute as cute

            compiled = cute.compile(
                wpr_launch, *wargs, ilp, apply_rht, L, FORMAT, USE_RCEIL, warps,
                ldwidth,
            )
            _JIT_CACHE[key] = compiled
        compiled(*wargs)
    else:
        nthreads_needed = (GPR + ilp - 1) // ilp
        ncta_x = max(1, (nthreads_needed + threads - 1) // threads)
        ncta_y = ceil_div(M, rows_per_cta)
        args = common + (threads, ncta_x, ncta_y, stream)
        key = (str(x.dtype), apply_rht, L, threads, ilp, FORMAT, USE_RCEIL)
        compiled = _JIT_CACHE.get(key)
        if compiled is None:
            import cutlass.cute as cute

            compiled = cute.compile(
                striped_launch, *args, ilp, apply_rht, L, FORMAT, USE_RCEIL, ldwidth
            )
            _JIT_CACHE[key] = compiled
        compiled(*args)

    return q_data[:, : K // 2], scales_u8


@torch.library.custom_op("torchao::fp4_quantize_unified", mutates_args=())
def fp4_quantize_unified(
    input: torch.Tensor,
    sign_vector: list[int],
    fmt: str = "nvfp4",
    scaling_mode: str = "floor",
    scale_layout: str = "cublas_blocked",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified FP4 (NVFP4/MXFP4 +/- RHT) quantize custom op.

    Empty ``sign_vector`` selects the plain cast. For NVFP4 the per-tensor
    global scale is computed from the input amax.
    """
    return _fp4_quantize_unified_impl(
        input,
        sign_vector=list(sign_vector) if len(sign_vector) > 0 else None,
        fmt=fmt,
        scaling_mode=scaling_mode,
        scale_layout=scale_layout,
    )


@fp4_quantize_unified.register_fake
def _(
    input: torch.Tensor,
    sign_vector: list[int],
    fmt: str = "nvfp4",
    scaling_mode: str = "floor",
    scale_layout: str = "cublas_blocked",
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = input.shape
    blk = 16 if fmt == "nvfp4" else 32
    k_blocks = K // blk
    qdata = torch.empty((M, K // 2), device=input.device, dtype=torch.uint8)
    if scale_layout == "linear":
        scales = torch.empty((M, k_blocks), device=input.device, dtype=torch.uint8)
    elif scale_layout == "cublas_blocked":
        pr = ceil_div(M, 128) * 128
        pc = ceil_div(k_blocks, 4) * 4
        scales = torch.empty((pr * pc,), device=input.device, dtype=torch.uint8)
    else:
        atom_m0 = 128 if fmt == "nvfp4" else 32
        rest_m = ceil_div(M, atom_m0 * 4)
        rest_k = ceil_div(k_blocks, 4)
        scales = torch.empty(
            (atom_m0 * 16 * rest_m * rest_k,), device=input.device, dtype=torch.uint8
        )
    return qdata, scales


def fp4_quantize_unified_2d(
    x: torch.Tensor,
    sign_vector=None,
    fmt: str = "nvfp4",
    scaling_mode: str = "floor",
    scale_layout: str = "cublas_blocked",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gated public wrapper for the unified FP4 (+/- RHT) CuTeDSL quantize cast.

    Raises ``NotImplementedError`` (with the missing-runtime detail) when the
    CuTeDSL runtime / SM 10.x / CUDA >= 12.8 requirements are not met. An empty
    / ``None`` ``sign_vector`` selects the plain cast.
    """
    from torchao.prototype.mx_formats.cutedsl import (
        _fp4_cutedsl_kernels_available,
    )

    if not _fp4_cutedsl_kernels_available:
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            _missing_cutedsl_runtime_packages,
        )

        raise NotImplementedError(
            "fp4_quantize_unified requires CUDA SM10.x, CUDA>=12.8, and: "
            f"{_missing_cutedsl_runtime_packages() or 'nvidia-cutlass-dsl'}"
        )
    return fp4_quantize_unified(
        x, list(sign_vector) if sign_vector is not None else [], fmt, scaling_mode,
        scale_layout,
    )
