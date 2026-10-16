# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Dim-1 (colwise, 32x1 block) MXFP8 quantization, RCEIL or FLOOR scales,
tcgen05 128x4 swizzled E8M0 scale layout. Blackwell / sm_100.

The public entry point, :func:`mxfp8_quantize_cutedsl_2d_32x1`, is functional
(allocates and returns outputs, not destination-passing) and accepts bf16 or
fp32 input. ``offs`` validation reuses :func:`cute_utils.validate_group_sizes`.

Both dtype paths share the same scale/reciprocal algorithm, applied to a
per-dtype amax value:

  * RCEIL: divide amax by 448 and round the power of two UP with the
    hardware ``cvt.rp.ue8m0x2.f32`` instruction (deliberately no
    ``.satfinite``, so +-Inf/NaN rounds to the E8M0 NaN byte 255 instead of
    clamping to the largest finite scale). The reciprocal ``2**(127-e)`` is
    then built directly as an IEEE-754 bit pattern, ``(254-e) << 23``, with
    three fixups the plain formula gets wrong at its edges:

      - ``e == 254`` -> ``2**-127`` is subnormal; ``(254-e)<<23`` evaluates
        to 0 instead of the correct mantissa bit ``0x00400000``.
      - ``e == 0`` with ``amax > 0`` -> ``2**127``; ``e == 0`` with
        ``amax == 0`` exactly -> ``1.0``, so a truly zero block's data
        trivially stays zero after the multiply.
      - ``e == 255`` (Inf/NaN) -> NaN, so it propagates through the
        multiply and the following saturating fp8 convert.

  * FLOOR: not the performance-critical path (production always uses
    RCEIL), so both dtypes reuse :func:`cute_utils.compute_scale_from_amax`
    directly rather than a hand-tuned instruction sequence -- correct by
    construction for every degenerate case, at some cost in raw throughput
    for that mode only.

  * quantize: multiply by the reciprocal (exact, since it is always a power
    of two) then a saturating convert to e4m3, packed 2- or 4-wide per
    instruction.

``RCEIL``/``FLOOR``, ``offs``, and ``blocked_scale_output`` are all
``cutlass.Constexpr`` template parameters, so the unused branch has no
representation in the compiled kernel: passing ``offs=None`` compiles the
validation out entirely, and selecting one of ``RCEIL``/``FLOOR`` never
traces the other's instruction sequence.

The two dtypes differ in packing, amax reduction, and how the transpose to
(K,M) output is bridged -- see the section comments below for each.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from .cute_utils import compute_scale_from_amax, validate_group_sizes

# ============================================================================
# bf16 path. Two bf16 values packed per 32-bit register throughout, so there
# is no bf16->fp32 expansion on the load or compute side -- the amax value
# fed into the shared RCEIL/FLOOR reciprocal math above is a lossless fp32
# widening of the winning bf16 bit pattern (`<<16`), not a promoted sum.
# Block amax is a `max.NaN.xorsign.abs.bf16x2` reduction tree (NaN-
# propagating, so a lone NaN sharing a block with finite values is not
# silently dropped). The transpose from (M,K) input to (K,M) output is
# bridged through an XOR-swizzled SMEM tile holding the *fp8* output
# (1 byte/elem, half the traffic of staging the wider bf16 input), which
# turns the (K,M) stores into contiguous 16-byte writes per output row.
#
# CTA layout: THREADS = NWARPS*32 = 128 threads (4 warps), covering one
# 128-K-column block x 256 M rows (MPASS=2 passes of 128 rows each) per CTA
# on the M % 256 == 0 fast path; MPASS falls back to 1 (128 M rows/CTA)
# otherwise.
#
# Zoom in on ONE m-pass (128 M rows): the NWARPS=4 warps each own one
# CONSECUTIVE 32-row M block -- one whole 32x1 quantization block per warp,
# spanning all 128 K columns of the block:
#
#          <-------------------- 128 K columns (kb) -------------------->
#   warp 0 |     32 M rows, block 0     |  32 lanes x 4 K-cols/lane each  |
#   warp 1 |     32 M rows, block 1     |  (lane l owns k = 4l .. 4l+3)   |
#   warp 2 |     32 M rows, block 2     |                                 |
#   warp 3 |     32 M rows, block 3     |                                 |
#
# So every lane reduces amax/scale/quantize for its own 4 K-columns' worth
# of a 32-row block entirely in registers -- no cross-lane reduction, and no
# shared memory needed on the load side. MPASS such passes stack to cover
# the CTA's full M extent.
# ============================================================================

_BF16_NWARPS = 4  # warps per CTA (each covers 32 M rows)
_BF16_MPASS = 2  # m-passes per CTA on the fast path (needs M % 256 == 0)
_BF16_GROUP = 4096  # k-blocks per grid-swizzle group (>= gk -> plain k-major)
_BF16_NACC = 4  # independent amax accumulator chains


def _asm(ret_t, args, txt, cons):
    return llvm.inline_asm(
        ret_t, args, txt, cons,
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _bf16_absmax_bf16x2(acc: cutlass.Uint32, v: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """per-lane magnitude = max(|acc|,|v|); sign is garbage, masked at the end."""
    return cutlass.Uint32(_asm(
        T.i32(),
        [cutlass.Uint32(acc).ir_value(loc=loc, ip=ip), cutlass.Uint32(v).ir_value(loc=loc, ip=ip)],
        "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r"))


@dsl_user_op
def _bf16_rceil_e8m0x2(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    """hardware RCEIL -> (e(a) << 8) | e(b)).

    Deliberately no `.satfinite`: with it, +-Inf/NaN amax maps to the
    largest finite E8M0 byte (254) instead of the NaN sentinel (255), which
    would disagree with the non-saturating conversion
    `cute_utils._cvt_f32_to_ue8m0` uses for the FLOOR path in this file.
    """
    return cutlass.Uint32(_asm(
        T.i32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.rp.ue8m0x2.f32 h, $1, $2;\n\t"
        "cvt.u32.u16 $0, h;\n\t}\n", "=r,f,f"))


@dsl_user_op
def _bf16_quant2(w0: cutlass.Uint32, w1: cutlass.Uint32, ip2: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Two rows x two bf16 columns -> 4 packed e4m3 bytes.

    inv is always a power of two, so the bf16 multiply is exact and matches
    the reference's fp32 multiply bit for bit.
    result bytes (low->high): r0c0, r0c1, r1c0, r1c1
    """
    return cutlass.Uint32(_asm(
        T.i32(),
        [cutlass.Uint32(w0).ir_value(loc=loc, ip=ip), cutlass.Uint32(w1).ir_value(loc=loc, ip=ip),
         cutlass.Uint32(ip2).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t0,t1;\n\t.reg .b16 h0,h1;\n\t"
        "mul.rn.bf16x2 t0, $1, $3;\n\tmul.rn.bf16x2 t1, $2, $3;\n\t"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h0, t0;\n\t"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h1, t1;\n\t"
        "mov.b32 $0, {h0, h1};\n\t}\n", "=r,r,r,r"))


@dsl_user_op
def _bf16_prmt_lo(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(_asm(T.i32(),
        [cutlass.Uint32(a).ir_value(loc=loc, ip=ip), cutlass.Uint32(b).ir_value(loc=loc, ip=ip)],
        "prmt.b32 $0, $1, $2, 25632;", "=r,r,r"))  # 0x6420


@dsl_user_op
def _bf16_prmt_hi(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(_asm(T.i32(),
        [cutlass.Uint32(a).ir_value(loc=loc, ip=ip), cutlass.Uint32(b).ir_value(loc=loc, ip=ip)],
        "prmt.b32 $0, $1, $2, 30001;", "=r,r,r"))  # 0x7531


def _bf16_amax_tree(vals, nacc=4):
    """|max| over `vals` using `nacc` independent chains.

    Plain-Python helper: unrolled at trace time, so the 32-deep serial
    HMNMX2 dependency becomes `nacc` parallel chains of depth 32/nacc.
    """
    acc = list(vals[:nacc])
    for i in range(nacc, len(vals)):
        acc[i % nacc] = _bf16_absmax_bf16x2(acc[i % nacc], vals[i])
    n = nacc
    while n > 1:
        n //= 2
        for t in range(n):
            acc[t] = _bf16_absmax_bf16x2(acc[t], acc[t + n])
    return acc[0]


def _bf16_load(mXu, frg, mrow, kcol):
    cute.autovec_copy(cute.local_tile(mXu, (32, 2), (mrow, kcol)), frg)


def _bf16_inv_from_e_rceil(e, amaxb):
    """2**(127-e) as an IEEE-754 bit pattern, with the degenerate-block
    fixups described in the module docstring: amax==0 -> 1.0, e==0
    (amax>0) -> 2**127, e==254 -> 2**-127 (subnormal), e==255 -> NaN."""
    ib = (cutlass.Int32(254) - e) << 23
    ib = ib | ((((e ^ cutlass.Int32(254)) - cutlass.Int32(1)) >> 31) & cutlass.Int32(0x00400000))
    is255 = (cutlass.Int32(254) - e) >> 31  # -1 iff e == 255
    ib = (ib & (is255 ^ cutlass.Int32(-1))) | (is255 & cutlass.Int32(0x7FC00000))  # e==255 -> NaN
    eq0 = (e - cutlass.Int32(1)) >> 31  # -1 iff e == 0
    eqz = (amaxb - cutlass.Int32(1)) >> 31  # -1 iff amax == 0
    v0 = cutlass.Int32(0x7F000000) + (eqz & cutlass.Int32(0x3F800000 - 0x7F000000))
    return (ib & (eq0 ^ cutlass.Int32(-1))) | (eq0 & v0)


@cute.kernel
def _bf16_kernel(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor,
                  ncb: cutlass.Constexpr, gm: cutlass.Constexpr,
                  grp: cutlass.Constexpr, MP: cutlass.Constexpr,
                  USE_RCEIL: cutlass.Constexpr[bool],
                  BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
                  offs: Optional[cute.Tensor]):
    NW = _BF16_NWARPS
    mXu = cute.recast_tensor(mX, cutlass.Uint32)  # (M, K/2)
    mQu = cute.recast_tensor(mQ, cutlass.Uint32)  # (K, M/4)
    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        mSu = cute.recast_tensor(mS, cutlass.Uint32)  # (nrb*32, ncb*4)

    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    # Validation-only: compiles away entirely when offs is None.
    if cutlass.const_expr(offs is not None):
        if tidx == 0:
            validate_group_sizes(offs)
    g = bid // (grp * gm)
    rem = bid - g * (grp * gm)
    kb = g * grp + rem % grp
    mbg = rem // grp

    LPR = 2 * NW * MP
    QUADS = 128 * LPR
    NBLK = (NW * MP) // 4
    RPI = 32 // LPR
    KPW = 128 // NW

    w = tidx // 32
    l = tidx % 32
    l7 = l % 8
    jr = l % LPR
    rr = l // LPR

    smem = utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Uint32,
                              cute.make_layout((QUADS, 4), stride=(4, 1)),
                              byte_alignment=16)
    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        sS = smem.allocate_tensor(cutlass.Uint8,
                                  cute.make_layout((32 * NBLK, 16), stride=(16, 1)),
                                  byte_alignment=16)

    frgs = [cute.make_rmem_tensor(cute.make_layout((32, 2), stride=(2, 1)),
                                  cutlass.Uint32) for _ in range(MP)]
    tmp = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Uint32)
    out = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Uint32)

    for mp in cutlass.range_constexpr(MP):
        _bf16_load(mXu, frgs[mp], mbg * (NW * MP) + mp * NW + w, kb * 32 + l)

    for mp in cutlass.range_constexpr(MP):
        frg = frgs[mp]
        a0 = _bf16_amax_tree([frg[r, 0] for r in range(32)], _BF16_NACC)
        a1 = _bf16_amax_tree([frg[r, 1] for r in range(32)], _BF16_NACC)
        a0 = a0 & cutlass.Uint32(0x7FFF7FFF)
        a1 = a1 & cutlass.Uint32(0x7FFF7FFF)
        m0 = cutlass.Int32(a0) << 16
        m1 = cutlass.Int32(a0) & cutlass.Int32(0xFFFF0000)
        m2 = cutlass.Int32(a1) << 16
        m3 = cutlass.Int32(a1) & cutlass.Int32(0xFFFF0000)

        if cutlass.const_expr(USE_RCEIL):
            RMAX = cutlass.Float32(448.0)
            e01 = cutlass.Int32(_bf16_rceil_e8m0x2(m1.bitcast(cutlass.Float32) / RMAX,
                                                   m0.bitcast(cutlass.Float32) / RMAX))
            e23 = cutlass.Int32(_bf16_rceil_e8m0x2(m3.bitcast(cutlass.Float32) / RMAX,
                                                   m2.bitcast(cutlass.Float32) / RMAX))
            e0 = e01 & cutlass.Int32(255)
            e1 = (e01 >> 8) & cutlass.Int32(255)
            e2 = e23 & cutlass.Int32(255)
            e3 = (e23 >> 8) & cutlass.Int32(255)

            ib0 = _bf16_inv_from_e_rceil(e0, m0)
            ib1 = _bf16_inv_from_e_rceil(e1, m1)
            ib2 = _bf16_inv_from_e_rceil(e2, m2)
            ib3 = _bf16_inv_from_e_rceil(e3, m3)
        else:
            # FLOOR: not perf-critical (production uses RCEIL), so this
            # reuses cute_utils' own scale computation directly rather than
            # hand-deriving new PTX -- correct by construction for every
            # degenerate (zero/Inf/NaN/subnormal) case, at the cost of 4
            # scalar calls instead of 2 dual-packed hardware ones.
            sb0, iv0 = compute_scale_from_amax(m0.bitcast(cutlass.Float32), USE_RCEIL=False)
            sb1, iv1 = compute_scale_from_amax(m1.bitcast(cutlass.Float32), USE_RCEIL=False)
            sb2, iv2 = compute_scale_from_amax(m2.bitcast(cutlass.Float32), USE_RCEIL=False)
            sb3, iv3 = compute_scale_from_amax(m3.bitcast(cutlass.Float32), USE_RCEIL=False)
            e0, e1, e2, e3 = (cutlass.Int32(sb0), cutlass.Int32(sb1),
                              cutlass.Int32(sb2), cutlass.Int32(sb3))
            ib0, ib1, ib2, ib3 = (iv0.bitcast(cutlass.Int32), iv1.bitcast(cutlass.Int32),
                                  iv2.bitcast(cutlass.Int32), iv3.bitcast(cutlass.Int32))

        ip01 = cutlass.Uint32(ib1 | ((ib0 >> 16) & cutlass.Int32(0xFFFF)))
        ip23 = cutlass.Uint32(ib3 | ((ib2 >> 16) & cutlass.Int32(0xFFFF)))

        jloc = mp * NW + w
        if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
            srow = (jloc // 4) * 32 + 4 * l7
            scol = (l // 8) * 4 + (jloc % 4)
            sS[srow + 0, scol] = cutlass.Uint8(e0)
            sS[srow + 1, scol] = cutlass.Uint8(e1)
            sS[srow + 2, scol] = cutlass.Uint8(e2)
            sS[srow + 3, scol] = cutlass.Uint8(e3)
        else:
            # Plain (K, m_blocks) row-major layout: no SMEM staging needed,
            # scalar stores straight to global memory. m_block is the
            # 32-row block index of this mp/warp's M-tile; k is the raw
            # K column each of e0..e3 was reduced from (lane l owns
            # k-columns 4l..4l+3 within k-block kb, see the module header).
            m_block = mbg * (NW * MP) + jloc
            k_base = kb * 128 + 4 * l
            mS[k_base + 0, m_block] = cutlass.Uint8(e0)
            mS[k_base + 1, m_block] = cutlass.Uint8(e1)
            mS[k_base + 2, m_block] = cutlass.Uint8(e2)
            mS[k_base + 3, m_block] = cutlass.Uint8(e3)

        for half in cutlass.range_constexpr(2):
            cw = [[None] * 4 for _ in range(4)]
            for gg in cutlass.range_constexpr(4):
                r0 = half * 16 + gg * 4
                X01 = _bf16_quant2(frg[r0, 0], frg[r0 + 1, 0], ip01)
                Y01 = _bf16_quant2(frg[r0 + 2, 0], frg[r0 + 3, 0], ip01)
                X23 = _bf16_quant2(frg[r0, 1], frg[r0 + 1, 1], ip23)
                Y23 = _bf16_quant2(frg[r0 + 2, 1], frg[r0 + 3, 1], ip23)
                cw[0][gg] = _bf16_prmt_lo(X01, Y01)
                cw[1][gg] = _bf16_prmt_hi(X01, Y01)
                cw[2][gg] = _bf16_prmt_lo(X23, Y23)
                cw[3][gg] = _bf16_prmt_hi(X23, Y23)
            j = 2 * jloc + half
            for c in cutlass.range_constexpr(4):
                for gg in cutlass.range_constexpr(4):
                    tmp[gg] = cw[c][gg]
                k = 4 * l + c
                cute.autovec_copy(tmp, sQ[(k * LPR + (j ^ l7), None)])

    cute.arch.barrier()

    for i in cutlass.range_constexpr(KPW // RPI):
        k = KPW * w + i * RPI + rr
        cute.autovec_copy(sQ[(k * LPR + (jr ^ ((k // 4) % 8)), None)], out)
        gQ = cute.local_tile(mQu, (1, 4), (kb * 128 + k, mbg * LPR + jr))
        cute.autovec_copy(out, gQ[(0, None)])

    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        if tidx < 32 * NBLK:
            sSu = cute.recast_tensor(sS, cutlass.Uint32)
            cute.autovec_copy(sSu[(tidx, None)], out)
            sidx = 32 * (kb * ncb + mbg * NBLK + tidx // 32) + tidx % 32
            gS = cute.local_tile(mSu, (1, 4), (sidx // ncb, sidx % ncb))
            cute.autovec_copy(out, gS[(0, None)])


@cute.jit
def _bf16_launch(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor,
                  gk: cutlass.Constexpr, gm: cutlass.Constexpr,
                  ncb: cutlass.Constexpr, grp: cutlass.Constexpr,
                  MP: cutlass.Constexpr, USE_RCEIL: cutlass.Constexpr[bool],
                  BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
                  offs: Optional[cute.Tensor]):
    NW = _BF16_NWARPS
    sS_bytes = 32 * ((NW * MP) // 4) * 16 if cutlass.const_expr(BLOCKED_SCALE_OUTPUT) else 0
    _bf16_kernel(mX, mQ, mS, ncb, gm, grp, MP, USE_RCEIL, BLOCKED_SCALE_OUTPUT, offs).launch(
        grid=[gk * gm, 1, 1], block=[32 * NW, 1, 1],
        smem=128 * 2 * NW * MP * 16 + sS_bytes,
    )


def _bf16_group(gk):
    for g in range(min(_BF16_GROUP, gk), 0, -1):
        if gk % g == 0:
            return g
    return 1


_bf16_cache = {}


@torch.no_grad()
def _bf16_run(x, q, scale, use_rceil, blocked_scale_output, offs):
    """x: (M,K) bf16. q: (K,M) fp8_e4m3fn row-major.
    scale: flat blocked uint8, or (K, m_blocks) uint8 when not blocked."""
    has_offs = offs is not None
    key = (x.shape, use_rceil, blocked_scale_output, has_offs)
    fn = _bf16_cache.get(key)
    if fn is None:
        M, K = x.shape
        mp = _BF16_MPASS if M % (32 * _BF16_NWARPS * _BF16_MPASS) == 0 else 1
        offs_ct = (
            from_dlpack(offs, assumed_align=4, enable_tvm_ffi=True) if has_offs else None
        )
        options = (
            "--enable-tvm-ffi --enable-assertions" if has_offs else "--enable-tvm-ffi"
        )
        fn = cute.compile(
            _bf16_launch,
            from_dlpack(x, enable_tvm_ffi=True, assumed_align=16),
            from_dlpack(q, enable_tvm_ffi=True, assumed_align=16),
            from_dlpack(scale, enable_tvm_ffi=True, assumed_align=(16 if blocked_scale_output else 1)),
            K // 128, M // (32 * _BF16_NWARPS * mp), M // 128, _bf16_group(K // 128), mp,
            use_rceil, blocked_scale_output, offs_ct,
            options=options,
        )
        _bf16_cache[key] = fn
    fn(x, q, scale, offs)


# ============================================================================
# fp32 path. An fp32 value already fills a 32-bit register (no packing
# needed). Block amax uses `cute.arch.fmax`/`fmin` (NaN-propagating) plus one
# butterfly shuffle instead of a packed reduction tree. The quantized output
# is staged through a cuBLAS/TMA 128B-swizzled SMEM tile and pushed to
# global memory with a single `cp.async.bulk.tensor` box, rather than the
# bf16 path's per-thread vectorized stores.
#
# CTA layout: THREADS = 256 (8 warps), tile = TM=128 rows x TK=64 columns.
# Warp index w splits into mq = w % 4 (which 32-row sub-block of the 128-row
# tile) and kq = w // 4 (which 32-column half of the 64-column tile):
#
#          <---- kq=0 (32 K cols) ---->  <---- kq=1 (32 K cols) ---->
#   mq=0        warp 0                        warp 4
#   mq=1        warp 1                        warp 5
#   mq=2        warp 2                        warp 6
#   mq=3        warp 3                        warp 7
#
# Within one warp, lane l splits into b = l % 2 (which 16-row half of the
# warp's 32-row block) and a = l // 2 (0..15, indexing the warp's 32 K
# columns 2 at a time: k = kt*2 + j for j = 0, 1). The lane pair sharing an
# `a` (b=0 and b=1) jointly cover one 32x1 quantization block -- b=0 reduces
# rows [mq*32, mq*32+16), b=1 reduces [mq*32+16, mq*32+32), both over the
# same 2 K columns -- and a single `shuffle_sync_bfly(offset=1)` (exactly
# the b bit) merges the two 16-row partial amaxes into the full 32-row block
# amax; only b==0 then writes the scale byte.
# ============================================================================

_FP32_TM = 128
_FP32_TK = 64
_FP32_THREADS = 256


@dsl_user_op
def _fp32_ue8m0x2_rp(a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Int32:
    """(amax_a, amax_b) -> packed E8M0: bits[15:8] = e(a), bits[7:0] = e(b).

    Deliberately no `.satfinite`: with it, +-Inf/NaN amax maps to the
    largest finite E8M0 byte (254) instead of the NaN sentinel (255), which
    would disagree with the non-saturating conversion
    `cute_utils._cvt_f32_to_ue8m0` uses for the FLOOR path in this file.
    """
    r = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .f32 da, db;\n\t"
        ".reg .b16 sp;\n\t"
        "div.rn.f32 da, $1, 0f43E00000;\n\t"
        "div.rn.f32 db, $2, 0f43E00000;\n\t"
        "cvt.rp.ue8m0x2.f32 sp, da, db;\n\t"
        "cvt.u32.u16 $0, sp;\n\t"
        "}\n",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


@dsl_user_op
def _fp32_inv_from_e_rceil(e: cutlass.Int32, amax: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """2**(127-e) as an IEEE-754 bit pattern, with the degenerate-block
    fixups described in the module docstring: amax==0 -> 1.0, e==0
    (amax>0) -> 2**127, e==254 -> 2**-127 (subnormal), e==255 -> NaN.
    `amax` (not just `e`) is needed to distinguish a truly zero block from
    one whose amax merely rounds down to the smallest representable scale,
    both of which produce e==0.
    """
    r = llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(e).ir_value(loc=loc, ip=ip), cutlass.Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 t;\n\t"
        ".reg .pred p;\n\t"
        "sub.s32 t, 254, $1;\n\t"
        "shl.b32 t, t, 23;\n\t"
        "setp.eq.s32 p, $1, 254;\n\t"
        "selp.b32 t, 4194304, t, p;\n\t"
        "setp.eq.s32 p, $1, 0;\n\t"
        "selp.b32 t, 2130706432, t, p;\n\t"
        "setp.eq.f32 p, $2, 0f00000000;\n\t"
        "selp.b32 t, 1065353216, t, p;\n\t"
        "setp.eq.s32 p, $1, 255;\n\t"
        "selp.b32 t, 2143289344, t, p;\n\t"
        "mov.b32 $0, t;\n\t"
        "}\n",
        "=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Float32(r)


@cute.kernel
def _fp32_kernel(
    mX: cute.Tensor,
    store_atom: cute.CopyAtom,
    tma_q: cute.Tensor,
    mS: cute.Tensor,
    nMB: cutlass.Constexpr,
    USE_RCEIL: cutlass.Constexpr[bool],
    BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
    offs: Optional[cute.Tensor],
):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # Validation-only: compiles away entirely when offs is None.
    if cutlass.const_expr(offs is not None):
        if tid == 0:
            validate_group_sizes(offs)

    smem = utils.SmemAllocator()
    sQg = smem.allocate_tensor(
        cutlass.Float8E4M3FN,
        cute.make_layout((_FP32_TK * _FP32_TM // 16, 16), stride=(16, 1)),
        byte_alignment=1024,
    )
    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        sS = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(256), byte_alignment=16)
    sQ = cute.make_tensor(
        sQg.iterator,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0,
            cute.make_layout((_FP32_TK, _FP32_TM), stride=(_FP32_TM, 1)),
        ),
    )

    w = tid // 32
    ln = tid % 32
    mq = w % 4
    kq = w // 4
    b = ln % 2
    a = ln // 2

    kt = bx * (_FP32_TK // 2) + kq * 16 + a
    kl = kq * 32 + a * 2
    m0 = by * _FP32_TM + mq * 32 + b * 16
    mg = mq * 2 + b

    frg = cute.make_rmem_tensor(
        cute.make_layout((16, 2), stride=(2, 1)), cutlass.Float32
    )
    for i in cutlass.range_constexpr(16):
        cute.autovec_copy(
            cute.local_tile(mX, (1, 2), (m0 + i, kt))[(0, None)], frg[(i, None)]
        )

    # nan=True (NaN-propagating) throughout: the default IEEE
    # maximumNumber/minimumNumber semantics silently drop a lone NaN sharing
    # a block with any finite value instead of propagating it.
    mxv = [frg[(0, 0)], frg[(0, 1)]]
    mnv = [frg[(0, 0)], frg[(0, 1)]]
    for i in cutlass.range_constexpr(1, 16):
        for j in cutlass.range_constexpr(2):
            v = frg[(i, j)]
            mxv[j] = cute.arch.fmax(mxv[j], v, nan=True)
            mnv[j] = cute.arch.fmin(mnv[j], v, nan=True)

    am = []
    for j in cutlass.range_constexpr(2):
        v = cute.arch.fmax(mxv[j], cutlass.Float32(0.0) - mnv[j], nan=True)
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=1), nan=True)
        am.append(v)

    if cutlass.const_expr(USE_RCEIL):
        p0 = _fp32_ue8m0x2_rp(am[0], am[1])
        ev = [(p0 >> 8) & 0xFF, p0 & 0xFF]
        inv = [_fp32_inv_from_e_rceil(ev[0], am[0]), _fp32_inv_from_e_rceil(ev[1], am[1])]
    else:
        # FLOOR: not perf-critical, so reuse cute_utils' own scale
        # computation directly -- correct by construction for every
        # degenerate case.
        sb0, iv0 = compute_scale_from_amax(am[0], USE_RCEIL=False)
        sb1, iv1 = compute_scale_from_amax(am[1], USE_RCEIL=False)
        ev = [cutlass.Int32(sb0), cutlass.Int32(sb1)]
        inv = [iv0, iv1]

    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        if b == 0:
            for j in cutlass.range_constexpr(2):
                si = (a * 2 + j) * 8 + kq * 4 + mq
                sS[si ^ (((si >> 7) & 1) << 2)] = cutlass.Uint8(ev[j])
    else:
        # Plain (K, m_blocks) row-major layout: direct scalar stores, no
        # SMEM staging. Both lanes in a (b=0,b=1) pair hold the identical
        # post-shuffle `am`/`ev`, so only one needs to write.
        if b == 0:
            m_block = by * 4 + mq
            k_base = kt * 2
            mS[k_base + 0, m_block] = cutlass.Uint8(ev[0])
            mS[k_base + 1, m_block] = cutlass.Uint8(ev[1])

    tmp = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float32)
    fq = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float8E4M3FN)
    for j in cutlass.range_constexpr(2):
        for i in cutlass.range_constexpr(16):
            tmp[i] = frg[(i, j)] * inv[j]
        fq.store(tmp.load().to(cutlass.Float8E4M3FN))
        kk = kl + j
        cute.autovec_copy(fq, sQg[(kk * 8 + (mg ^ (kk % 8)), None)])

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()

    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        sbase = ((bx // 2) * nMB + by) * 512 + (bx % 2) * 8
        mS[sbase + (tid // 8) * 16 + (tid % 8)] = sS[tid ^ (((tid >> 7) & 1) << 2)]

    if w == 0:
        gQ = cute.local_tile(tma_q, (_FP32_TK, _FP32_TM), (None, None))
        tQsQ, tQgQ = cpasync.tma_partition(
            store_atom, 0, cute.make_layout(1),
            cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2),
        )
        cute.copy(store_atom, tQsQ, tQgQ[(None, bx, by)])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def _fp32_launch(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor,
                  USE_RCEIL: cutlass.Constexpr[bool],
                  BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
                  offs: Optional[cute.Tensor]):
    M, K = mX.shape
    nMB = M // _FP32_TM
    nKB = K // _FP32_TK
    smem_layout = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0,
        cute.make_layout((_FP32_TK, _FP32_TM), stride=(_FP32_TM, 1)),
    )
    store_atom, tma_q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mQ, smem_layout, (_FP32_TK, _FP32_TM)
    )
    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        mS_arg = cute.make_tensor(mS.iterator, cute.make_layout((K // 128) * nMB * 512))
        sS_bytes = 256
    else:
        mS_arg = mS  # genuine (K, m_blocks) uint8 tensor, indexed directly
        sS_bytes = 0
    _fp32_kernel(mX, store_atom, tma_q, mS_arg, nMB, USE_RCEIL, BLOCKED_SCALE_OUTPUT, offs).launch(
        grid=(nKB, nMB, 1), block=(_FP32_THREADS, 1, 1), smem=8448 + sS_bytes
    )


_fp32_cache = {}


def _fp32_run(x, q, scale, use_rceil, blocked_scale_output, offs):
    has_offs = offs is not None
    key = (x.shape[0], x.shape[1], use_rceil, blocked_scale_output, has_offs)
    fn = _fp32_cache.get(key)
    if fn is None:
        mx = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
        mq = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True)
        ms = from_dlpack(scale, assumed_align=(16 if blocked_scale_output else 1), enable_tvm_ffi=True)
        offs_ct = (
            from_dlpack(offs, assumed_align=4, enable_tvm_ffi=True) if has_offs else None
        )
        options = (
            "--enable-tvm-ffi --enable-assertions" if has_offs else "--enable-tvm-ffi"
        )
        fn = cute.compile(
            _fp32_launch, mx, mq, ms, use_rceil, blocked_scale_output, offs_ct,
            options=options,
        )
        _fp32_cache[key] = fn
    fn(x, q, scale, offs)


# ============================================================================
# Public entry point
# ============================================================================


def mxfp8_quantize_cutedsl_2d_32x1(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to MXFP8 format using CuTe DSL kernel with 32x1 scaling.

    Quantizes along the M dimension - each column has M//32 scales, one per block of 32 M elements.

    Args:
        x: Input tensor of shape (M, K)
        block_size: Block size for quantization along M (only 32 supported)
        scaling_mode: Scaling mode ("floor" or "rceil")
        stage_count: Accepted but ignored -- this kernel has no TMA pipeline
            to stage. Kept for API parity with the other cutedsl/flydsl
            quantize ops.
        blocked_scale_output: Whether to output scales in blocked layout
        offs: Optional tensor of group end offsets for validation (must have group sizes as multiples of 128)

    Returns:
        q_data: Quantized data in row-major layout with shape (M, K) (no padding on data)
        scales: Scales tensor with shape (K, M//32) or blocked layout compatible with torch._scaled_mm
                Same format as other dim1 MXFP8 kernels for consistency
    """
    assert x.dtype in (torch.float32, torch.bfloat16), (
        "Input tensor must be float32 or bfloat16"
    )
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    assert scaling_mode in ("floor", "rceil"), (
        f"scaling_mode must be 'floor' or 'rceil', got {scaling_mode!r}"
    )
    M, K = x.shape
    assert M % 128 == 0, "M must be divisible by 128"
    assert K % 128 == 0, "K must be divisible by 128"
    if offs is not None:
        assert offs.is_cuda, "offs tensor must be CUDA"
        assert offs.dtype == torch.int32, "offs must be int32 tensor"
        assert offs.dim() == 1, "offs must be 1D tensor"
    m_blocks = M // block_size
    q_km = torch.empty((K, M), device=x.device, dtype=torch.float8_e4m3fn)
    if blocked_scale_output:
        # This kernel indexes the blocked scale buffer as a genuinely 2D
        # tensor (cute.local_tile needs its rank to match the (1, 4)
        # tiler); the flat 1D convention `quant.py` uses for the public
        # contract is the same bytes reshaped -- K % 128 == 0 and
        # M % 128 == 0 are already asserted, so `K // 128 * 32` rows times
        # `M // 128 * 16` columns is exactly the padded blocked-scale
        # element count either way.
        scales_u8 = torch.empty(
            (K // 128 * 32, M // 128 * 16), device=x.device, dtype=torch.uint8
        )
    else:
        scales_u8 = torch.empty((K, m_blocks), device=x.device, dtype=torch.uint8)

    use_rceil = scaling_mode == "rceil"
    if x.dtype == torch.bfloat16:
        _bf16_run(x, q_km, scales_u8, use_rceil, blocked_scale_output, offs)
    else:
        _fp32_run(x, q_km, scales_u8, use_rceil, blocked_scale_output, offs)

    if blocked_scale_output:
        return q_km.t(), scales_u8.reshape(-1).view(torch.float8_e8m0fnu)
    return q_km.t(), scales_u8.view(torch.float8_e8m0fnu)
