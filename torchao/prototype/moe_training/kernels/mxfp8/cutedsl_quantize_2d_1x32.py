# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Dim-0 (rowwise, 1x32 block) MXFP8 quantization, RCEIL or FLOOR scales,
cuBLAS 128x4 swizzled E8M0 scale layout.  Blackwell / sm_100.

The public entry point, :func:`mxfp8_quantize_cutedsl_2d_1x32`, is functional
(allocates and returns outputs, not destination-passing) and accepts bf16 or
fp32 input. ``offs`` validation reuses :func:`cute_utils.validate_group_sizes`.

Everything stays in the bf16 domain -- no bf16->f32 expansion:
  * block amax  : max.xorsign.abs.bf16x2 tree + 2 shfl.bfly + integer max
                  (non-negative bf16 patterns order identically as uint32)
  * RCEIL       : with v = amax bf16 bits,  u = v - 0x3E1
                    scale byte e   = u >> 7
                    inv (bf16 bits) = 0x7F00 - (u & 0x7F80)
                  exactly reproduces ceil(log2(amax/448)) + 127 because
                  448 = 1.75 * 2^8 and the carry out of the mantissa
                  compare (mant > 0x60) is folded into the +0x1F.
  * FLOOR       : with v = amax bf16 bits, E = v >> 7 (biased exponent field)
                    scale byte e = clamp(E - 8, 0, 254)
                    inv (bf16 bits) = 0x7F00 - (e << 7)
                  reproduces floor(log2(amax/448)) + 127 as a pure exponent
                  shift with no rounding: dividing by 256 = 2^8 subtracts 8
                  from the biased exponent exactly, and floor(log2(448)) == 8
                  since 448 == 1.75 * 2^8.
  * quantize    : mul.bf16x2 (exact: multiplying by a power of two) then
                  cvt.rn.satfinite.e4m3x2.bf16x2  -> bit-identical to the
                  fp32 reference path.

RCEIL and FLOOR are selected by a ``cutlass.Constexpr[bool]`` template
parameter (``USE_RCEIL``): the unused branch has no representation in the
compiled kernel at all, so there is no runtime branch cost for either mode.
Similarly, ``offs`` validation is gated by
``cutlass.const_expr(offs is not None)``, so passing ``offs=None`` compiles
out the validation entirely.
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from torchao.utils import ceil_div

from .cute_utils import validate_group_sizes

# CTA layout for the bf16 kernel: THREADS=512 threads (NW=16 warps x 32
# lanes) covering TM=128 rows x one K-tile per CTA.
#
# Zoom in on ONE warp -- its 32 lanes split into RPW=2 half-warps of
# LPR=16 lanes, each half-warp owning one whole row's 128-element K span
# (8 bf16 elements per lane):
#
#              <----------- row r (128 K-elements) ----------->
#             +-----+-----+-----+-----+-----+-----+~ ~+-----+
#   lane      |  0  |  1  |  2  |  3  |  4  |  5  | .. | 15  |   half-warp A
#             +-----+-----+-----+-----+-----+-----+~ ~+-----+   (LPR=16 lanes,
#                8     8     8     8     8     8    ..   8        RPW slot 0)
#                              bf16 elements / lane
#             +-----+-----+-----+-----+-----+-----+~ ~+-----+
#   lane      | 16  | 17  | 18  | 19  | 20  | 21  | .. | 31  |   half-warp B
#             +-----+-----+-----+-----+-----+-----+~ ~+-----+   (RPW slot 1)
#              <---------- row r+1 (128 K-elements) ---------->
#
# So one warp advances RPW=2 rows per step. Zoom back out: all NW=16 warps
# do this simultaneously, each on its own row-pair, covering
# RPP = NW*RPW = 32 rows in one "M pass" (pm):
#
#       row:    0   1   2   3   4   5  ...  28  29  30  31
#     warp:  [  0   0   1   1   2   2  ...  14  14  15  15  ]   <- pass pm
#
# Four such passes (NPM = TM/RPP = 128/32 = 4, pm = 0..3) stack to cover
# the CTA's full TM=128 rows -- one complete 128x4 cuBLAS scale-swizzle
# tile's worth of rows:
#
#     +----------------------+
#     | pass 0: rows   0..31 |  <- 16 warps x 2 rows, as diagrammed above
#     +----------------------+
#     | pass 1: rows  32..63 |
#     +----------------------+
#     | pass 2: rows  64..95 |
#     +----------------------+
#     | pass 3: rows 96..127 |
#     +----------------------+ = TM = 128 rows = one 128x4 scale tile
#
# Each pass's 32 rows map 1:1 onto the swizzle tile's "row % 32" axis (see
# the `(r % 32) * 16 + (r // 32) * 4 + c` indexing further down) -- pm *is*
# r // 32.
#
# MBPM=2: ask the launch to keep at least 2 CTAs resident per SM.
TM = 128
THREADS = 512
MBPM = 2

LPR = 16
RPW = 32 // LPR
NW = THREADS // 32
RPP = NW * RPW
NPM = TM // RPP

# KSUB = how many 128-wide K-segments one CTA processes (TK = KSUB * 128).
# Nothing in the per-32-element amax/scale/quantize math depends on this --
# it is purely a work-per-CTA tuning choice, so it is a Constexpr[int]
# kernel parameter (like USE_RCEIL) rather than a fixed module constant.
# KSUB=2 (requires K % 256 == 0) is the tuned default; KSUB=1 is a fallback
# for K that is only a multiple of 128.

# Shared amax-reduction prefix: reduces 4 packed-bf16x2 registers (8 values)
# down to a single non-negative bf16 magnitude bit pattern in the low 15 bits
# of `a` (upper bits zero). RCEIL and FLOOR both consume this unchanged and
# only differ in how they turn `a` into a scale byte + reciprocal.
#
# Must be the `.NaN`-propagating form of the max instruction: the plain
# `max.xorsign.abs.bf16x2` follows IEEE min/max-suppression semantics (picks
# the non-NaN operand when exactly one input is NaN), so a lone NaN sharing a
# block with otherwise-small-magnitude finite values would get silently
# dropped by the very first reduction step and never reach the E==255 check
# in the RCEIL/FLOOR tails below. `max.u32` afterwards is plain unsigned
# integer comparison, so it has no NaN-semantics issue of its own.
_ASM_AMAX_PREFIX = (
    ".reg .b32 a, b;\n"
    "max.NaN.xorsign.abs.bf16x2 a, $2, $3;\n"
    "max.NaN.xorsign.abs.bf16x2 b, $4, $5;\n"
    "max.NaN.xorsign.abs.bf16x2 a, a, b;\n"
    "shr.u32 b, a, 16;\n"
    "max.NaN.xorsign.abs.bf16x2 a, a, b;\n"
    "and.b32 a, a, 32767;\n"
    "shfl.sync.bfly.b32 b, a, 1, 31, 0xffffffff;\n"
    "max.u32 a, a, b;\n"
    "shfl.sync.bfly.b32 b, a, 2, 31, 0xffffffff;\n"
    "max.u32 a, a, b;\n"
)

# E == 255 means the block's amax is +-Inf or NaN (bf16 exponent field all
# ones). That case must force scale byte 255 and a NaN reciprocal -- NaN *
# anything = NaN, and cvt.satfinite of NaN saturates every element of the
# block to the e4m3fn NaN encoding (0x7F), invalidating the whole block even
# for its otherwise-finite elements. Without this override, the plain
# formula below would map Inf/NaN amax into the "normal" range (e.g. E=255
# -> scale byte 247 under RCEIL's -993 offset), silently wrong.
_ASM_AMAX_RCEIL = (
    "{\n" + _ASM_AMAX_PREFIX + ".reg .b32 E, u, s;\n"
    ".reg .pred p;\n"
    "shr.u32 E, a, 7;\n"
    "add.s32 u, a, -993;\n"
    "max.s32 u, u, 0;\n"
    "shr.u32 s, u, 7;\n"
    "setp.eq.s32 p, E, 255;\n"
    "selp.b32 $0, 255, s, p;\n"
    "and.b32 b, u, 32640;\n"
    "sub.s32 b, 32512, b;\n"
    "prmt.b32 b, b, 0, 0x1010;\n"
    "selp.b32 $1, 0x7FC07FC0, b, p;\n"
    "}"
)

# e = clamp(E - 8, 0, 254), where E = a >> 7 is amax's biased bf16 exponent
# field (a is already the 15-bit-clean magnitude from the shared prefix, so
# no masking is needed before the shift, unlike RCEIL's tail). Same E==255
# (Inf/NaN) override as RCEIL above.
_ASM_AMAX_FLOOR = (
    "{\n" + _ASM_AMAX_PREFIX + ".reg .b32 E, e;\n"
    ".reg .pred p;\n"
    "shr.u32 E, a, 7;\n"
    "add.s32 e, E, -8;\n"
    "max.s32 e, e, 0;\n"
    "min.s32 e, e, 254;\n"
    "setp.eq.s32 p, E, 255;\n"
    "selp.b32 $0, 255, e, p;\n"
    "shl.b32 b, e, 7;\n"
    "sub.s32 b, 32512, b;\n"
    "prmt.b32 b, b, 0, 0x1010;\n"
    "selp.b32 $1, 0x7FC07FC0, b, p;\n"
    "}"
)

_ASM_QUANT = (
    "{\n"
    ".reg .b32 m0, m1, m2, m3;\n"
    ".reg .b16 h0, h1, h2, h3;\n"
    "mul.bf16x2 m0, $2, $6;\n"
    "mul.bf16x2 m1, $3, $6;\n"
    "mul.bf16x2 m2, $4, $6;\n"
    "mul.bf16x2 m3, $5, $6;\n"
    "cvt.rn.satfinite.e4m3x2.bf16x2 h0, m0;\n"
    "cvt.rn.satfinite.e4m3x2.bf16x2 h1, m1;\n"
    "cvt.rn.satfinite.e4m3x2.bf16x2 h2, m2;\n"
    "cvt.rn.satfinite.e4m3x2.bf16x2 h3, m3;\n"
    "mov.b32 $0, {h0, h1};\n"
    "mov.b32 $1, {h2, h3};\n"
    "}"
)

_I32X2 = "!llvm.struct<(i32,i32)>"


@dsl_user_op
def _amax_rceil(r0, r1, r2, r3, *, loc=None, ip=None):
    ops = [cutlass.Int32(v).ir_value(loc=loc, ip=ip) for v in (r0, r1, r2, r3)]
    res = llvm.inline_asm(
        ir.Type.parse(_I32X2), ops, _ASM_AMAX_RCEIL, "=r,=r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Int32(llvm.extractvalue(T.i32(), res, [0])),
            cutlass.Int32(llvm.extractvalue(T.i32(), res, [1])))


@dsl_user_op
def _amax_floor(r0, r1, r2, r3, *, loc=None, ip=None):
    ops = [cutlass.Int32(v).ir_value(loc=loc, ip=ip) for v in (r0, r1, r2, r3)]
    res = llvm.inline_asm(
        ir.Type.parse(_I32X2), ops, _ASM_AMAX_FLOOR, "=r,=r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Int32(llvm.extractvalue(T.i32(), res, [0])),
            cutlass.Int32(llvm.extractvalue(T.i32(), res, [1])))


@dsl_user_op
def _quant8(r0, r1, r2, r3, inv, *, loc=None, ip=None):
    ops = [cutlass.Int32(v).ir_value(loc=loc, ip=ip)
           for v in (r0, r1, r2, r3, inv)]
    res = llvm.inline_asm(
        ir.Type.parse(_I32X2), ops, _ASM_QUANT, "=r,=r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Int32(llvm.extractvalue(T.i32(), res, [0])),
            cutlass.Int32(llvm.extractvalue(T.i32(), res, [1])))


@cute.kernel
def _mxfp8_kernel(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor,
                  kb_tiles: cutlass.Int32,
                  offs: Optional[cute.Tensor],
                  USE_RCEIL: cutlass.Constexpr[bool],
                  KSUB: cutlass.Constexpr[int],
                  BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool]):
    NL = NPM * KSUB
    SWORDS = KSUB * 128

    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # Validation-only: compiles away entirely when offs is None (has_offs is
    # part of the compile cache key, so this specialization never traces the
    # branch body at all).
    if cutlass.const_expr(offs is not None):
        if tidx == 0:
            validate_group_sizes(offs)

    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    rw = lane // LPR
    cl = lane % LPR
    c = cl // 4

    # The blocked/swizzled layout stages scale bytes through shared memory
    # so the whole SWORDS-byte strip can be flushed to global memory as one
    # coalesced access (see below). The plain row-major layout has no such
    # locality to exploit, so it skips the smem staging and writes
    # mS[m, k_block] directly with scalar per-element stores. All 4 lanes
    # that share a scale block (same `c`) compute the identical eb via the
    # warp-shuffle reduction above and therefore redundantly write the same
    # value to the same address; harmless, and scale bytes are already
    # 1/64th the bandwidth of the main data write.
    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        smem = utils.SmemAllocator()
        sS8 = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SWORDS * 4),
                                   byte_alignment=16)
        sS = cute.recast_tensor(sS8, cutlass.Int32)

    rr = warp * RPW + rw
    m0 = by * TM + rr
    kv = bx * (KSUB * LPR) + cl

    frg = cute.make_rmem_tensor(
        cute.make_layout((NL, 4), stride=(4, 1)), cutlass.Int32)
    for i in range(NL):
        pm, ks = i // KSUB, i % KSUB
        g = cute.local_tile(mX, (1, 4), (m0 + pm * RPP, kv + ks * LPR))
        cute.autovec_copy(g[(0, None)], frg[(i, None)])

    qfrg = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Int32)

    for i in range(NL):
        pm, ks = i // KSUB, i % KSUB
        r0, r1, r2, r3 = frg[(i, 0)], frg[(i, 1)], frg[(i, 2)], frg[(i, 3)]
        if cutlass.const_expr(USE_RCEIL):
            eb, inv = _amax_rceil(r0, r1, r2, r3)
        else:
            eb, inv = _amax_floor(r0, r1, r2, r3)
        p0, p1 = _quant8(r0, r1, r2, r3, inv)
        qfrg[0] = p0
        qfrg[1] = p1
        gq = cute.local_tile(mQ, (1, 2), (m0 + pm * RPP, kv + ks * LPR))
        cute.autovec_copy(
            qfrg, gq[(0, None)],
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
        )

        if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
            r = rr + pm * RPP
            sS8[ks * 512 + (r % 32) * 16 + (r // 32) * 4 + c] = eb.to(cutlass.Uint8)
        else:
            m = m0 + pm * RPP
            k_block = bx * (KSUB * 4) + ks * 4 + c
            mS[m, k_block] = eb.to(cutlass.Uint8)

    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        cute.arch.barrier()
        if tidx < SWORDS:
            mS[(by * kb_tiles + bx * KSUB) * 128 + tidx] = sS[tidx]


@cute.jit
def _mxfp8_launch(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor,
                  offs: Optional[cute.Tensor],
                  USE_RCEIL: cutlass.Constexpr[bool],
                  KSUB: cutlass.Constexpr[int],
                  BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool]):
    TK = KSUB * 128
    SWORDS = KSUB * 128
    m_tiles = cute.size(mX, mode=[0]) // TM
    k_tiles = (cute.size(mX, mode=[1]) * 2) // TK
    _mxfp8_kernel(
        mX, mQ, mS, k_tiles * KSUB, offs, USE_RCEIL=USE_RCEIL, KSUB=KSUB,
        BLOCKED_SCALE_OUTPUT=BLOCKED_SCALE_OUTPUT,
    ).launch(
        grid=[k_tiles, m_tiles, 1],
        block=[THREADS, 1, 1],
        smem=(SWORDS * 4 if cutlass.const_expr(BLOCKED_SCALE_OUTPUT) else 0),
        min_blocks_per_mp=MBPM,
    )


_compiled_cache = {}


def _get_compiled(
    scaling_mode: str, ksub: int, blocked_scale_output: bool,
    x_i32, q_i32, s_tensor, offs_i32,
):
    has_offs = offs_i32 is not None
    key = (scaling_mode, has_offs, ksub, blocked_scale_output)
    compiled = _compiled_cache.get(key)
    if compiled is None:
        x_ct = (from_dlpack(x_i32, assumed_align=16, enable_tvm_ffi=True)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, divisibility=64))
        q_ct = (from_dlpack(q_i32, assumed_align=16, enable_tvm_ffi=True)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, divisibility=32))
        if blocked_scale_output:
            # s_tensor is the flat (padded_scale_rows * padded_scale_cols,)
            # uint8 buffer, viewed as int32 for the vectorized bulk copy.
            s_ct = from_dlpack(
                s_tensor, assumed_align=16, enable_tvm_ffi=True
            ).mark_layout_dynamic()
        else:
            # s_tensor is the (M, k_blocks) uint8 buffer, written with
            # scalar per-element stores -- both dims genuinely vary at
            # runtime, so mark both dynamic like x_ct/q_ct.
            s_ct = (
                from_dlpack(s_tensor, assumed_align=1, enable_tvm_ffi=True)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, divisibility=1)
            )
        if has_offs:
            offs_ct = from_dlpack(
                offs_i32, assumed_align=4, enable_tvm_ffi=True
            ).mark_layout_dynamic()
            options = "--enable-tvm-ffi --enable-assertions"
        else:
            offs_ct = None
            options = "--enable-tvm-ffi"
        compiled = cute.compile(
            _mxfp8_launch, x_ct, q_ct, s_ct, offs_ct,
            USE_RCEIL=(scaling_mode == "rceil"),
            KSUB=ksub,
            BLOCKED_SCALE_OUTPUT=blocked_scale_output,
            options=options,
        )
        _compiled_cache[key] = compiled
    return compiled


# ============================================================================
# FP32 path. Structurally different from the bf16 kernel above: an fp32
# value already fills a 32-bit register (no 2x packing needed), and the
# exponent/reciprocal math uses the hardware cvt.{rp,rz}.ue8m0x2.f32
# conversion instruction (no `.satfinite` -- see _cvt_ue8m0_rceil) rather
# than a hand-rolled integer bit-trick. K only needs to be a multiple of 128
# here (TILE_COLS=128) -- no KSUB-style 256-wide fallback needed.
# ============================================================================

_F32_NTHREADS = 128
_F32_VEC = 8  # fp32 per lane; 4 | VEC | 32
_F32_TILE_COLS = 128
_F32_ALIGN = 4 * _F32_VEC
_F32_NWARPS = _F32_NTHREADS // 32
_F32_LPR = _F32_TILE_COLS // _F32_VEC  # lanes covering one row
_F32_RPS = 32 // _F32_LPR  # rows covered by one warp step
_F32_NT = _F32_NWARPS * _F32_RPS  # rows handled per step (must be 8)
_F32_NSTEPS = 32 // _F32_NT  # steps == the 4 sub-rows
_F32_LPB = 32 // _F32_VEC  # lanes spanned by one 32-element MX block
_F32_SHFL = tuple(1 << i for i in range(_F32_LPB.bit_length() - 1))
_F32_NKB = _F32_TILE_COLS // 128  # 128x4 scale blocks per tile
_F32_SMEM_BYTES = 128 * _F32_NKB


@dsl_user_op
def _cvt_e4m3x4(a0, a1, a2, a3, *, loc=None, ip=None) -> cutlass.Uint32:
    """4 x f32 -> 4 packed e4m3 bytes (a0 in the LOW byte)."""
    v0 = cutlass.Float32(a0).ir_value(loc=loc, ip=ip)
    v1 = cutlass.Float32(a1).ir_value(loc=loc, ip=ip)
    v2 = cutlass.Float32(a2).ir_value(loc=loc, ip=ip)
    v3 = cutlass.Float32(a3).ir_value(loc=loc, ip=ip)
    res = llvm.inline_asm(
        T.i32(),
        [v0, v1, v2, v3],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t"
        "}\n",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Uint32(res)


@dsl_user_op
def _cvt_ue8m0_rceil(a, *, loc=None, ip=None) -> cutlass.Int32:
    """Biased E8M0 exponent of `a`, rounding the power of two UP (RCEIL).

    Deliberately no `.satfinite`: with it, +-Inf amax clamps to the largest
    finite E8M0 byte (254, 2^127) instead of mapping to the NaN byte (255),
    which is required so an Inf/NaN block invalidates its scale rather than
    quantizing as if it were merely a very large finite value. Non-Inf/NaN
    inputs are unaffected either way (nothing to saturate).
    """
    v = cutlass.Float32(a).ir_value(loc=loc, ip=ip)
    res = llvm.inline_asm(
        T.i32(),
        [v],
        "{\n\t"
        ".reg .b16 t;\n\t"
        "cvt.rp.ue8m0x2.f32 t, $1, $1;\n\t"
        "cvt.u32.u16 $0, t;\n\t"
        "}\n",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(res) & cutlass.Int32(0xFF)


@dsl_user_op
def _cvt_ue8m0_floor(a, *, loc=None, ip=None) -> cutlass.Int32:
    """Biased E8M0 exponent of `a`, rounding the power of two DOWN (FLOOR).

    No `.satfinite`, same reasoning as `_cvt_ue8m0_rceil`.
    """
    v = cutlass.Float32(a).ir_value(loc=loc, ip=ip)
    res = llvm.inline_asm(
        T.i32(),
        [v],
        "{\n\t"
        ".reg .b16 t;\n\t"
        "cvt.rz.ue8m0x2.f32 t, $1, $1;\n\t"
        "cvt.u32.u16 $0, t;\n\t"
        "}\n",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(res) & cutlass.Int32(0xFF)


@cute.kernel
def _mxfp8_kernel_fp32(gX: cute.Tensor, gQ: cute.Tensor, gS: cute.Tensor,
                       offs: Optional[cute.Tensor],
                       USE_RCEIL: cutlass.Constexpr[bool],
                       BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool]):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()

    if cutlass.const_expr(offs is not None):
        if tidx == 0:
            validate_group_sizes(offs)

    sub = by % 4
    mb = by // 4

    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        smem = utils.SmemAllocator()
        sRaw = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout(_F32_SMEM_BYTES), byte_alignment=16
        )
        sByte = cute.make_tensor(sRaw.iterator, cute.make_layout(_F32_SMEM_BYTES))
        sOut = cute.make_tensor(
            sRaw.iterator, cute.make_layout((4, 32, _F32_NKB), stride=(1, 4, 128))
        )

    col_lane = lane % _F32_LPR
    row_off = lane // _F32_LPR
    blk = col_lane // _F32_LPB  # 32-element MX block inside the tile row
    c_in_kb = blk % 4
    kb_local = blk // 4

    # Issue every load up front: NSTEPS independent LDG.256 in flight per
    # thread is what keeps the DRAM pipe full at this occupancy.
    rXs = [cute.make_rmem_tensor((_F32_VEC,), cutlass.Float32) for _ in range(_F32_NSTEPS)]
    for s in cutlass.range_constexpr(_F32_NSTEPS):
        cute.autovec_copy(
            gX[(None, col_lane, row_off, warp, s, bx, sub, mb)], rXs[s]
        )

    for s in cutlass.range_constexpr(_F32_NSTEPS):
        rX = rXs[s]
        v = rX.load()
        amax = cute.abs(v).reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        # nan=True: the shuffle-reduce (unlike the .reduce() above) defaults
        # to NaN-quiet "maximumNumber" semantics, which would silently drop
        # a NaN sharing a block with a finite value from a different lane,
        # producing a normal-looking (wrong) scale instead of invalidating
        # the block.
        for oi in cutlass.range_constexpr(len(_F32_SHFL)):
            amax = cute.arch.fmax(
                amax, cute.arch.shuffle_sync_bfly(amax, _F32_SHFL[oi]), nan=True
            )

        if cutlass.const_expr(USE_RCEIL):
            descale = amax / cutlass.Float32(448.0)
            e = _cvt_ue8m0_rceil(descale)
        else:
            descale = amax * cutlass.Float32(1.0 / 256.0)
            e = _cvt_ue8m0_floor(descale)

        # Reciprocal from the biased E8M0 byte -- mode-agnostic, same for
        # RCEIL and FLOOR. The general formula (254-e)<<23 (a normal fp32
        # with exponent field 254-e, i.e. 2^(127-e)) is already correct at
        # e==0 (2^127), so no special case is needed there.
        #
        # e==254 needs an explicit case: 2^(127-254) = 2^-127 needs fp32
        # *subnormal* representation (exponent field 0, mantissa MSB set),
        # which the general formula can't produce (it would give exponent
        # field 0 with mantissa 0, i.e. 0.0).
        #
        # e==255 (block amax is +-Inf or NaN) must produce a NaN reciprocal,
        # not 0.0: finite * 0.0 == 0.0 (wrong -- the whole block must
        # invalidate to the e4m3fn NaN encoding, 0x7F), whereas
        # finite * NaN == NaN, which satfinite-converts to 0x7F correctly
        # for every element in the block.
        inv_bits = (cutlass.Int32(254) - e) << 23
        inv_bits = cutlass.Int32(1 << 22) if e == 254 else inv_bits
        inv_bits = cutlass.Int32(0x7FC00000) if e == 255 else inv_bits
        inv = inv_bits.bitcast(cutlass.Float32)

        rQ = cute.make_rmem_tensor((_F32_VEC // 4,), cutlass.Uint32)
        for j in cutlass.range_constexpr(_F32_VEC // 4):
            rQ[j] = _cvt_e4m3x4(
                rX[4 * j + 0] * inv,
                rX[4 * j + 1] * inv,
                rX[4 * j + 2] * inv,
                rX[4 * j + 3] * inv,
            )
        cute.autovec_copy(
            rQ, gQ[(None, col_lane, row_off, warp, s, bx, sub, mb)]
        )

        if col_lane % _F32_LPB == 0:
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
                sByte[
                    kb_local * 128
                    + (s * _F32_NT + warp * _F32_RPS + row_off) * 4
                    + c_in_kb
                ] = e.to(cutlass.Uint8)
            else:
                abs_row = mb * 128 + sub * 32 + (s * _F32_NT + warp * _F32_RPS + row_off)
                abs_kblock = bx * _F32_NKB * 4 + kb_local * 4 + c_in_kb
                gS[abs_row, abs_kblock] = e.to(cutlass.Uint8)

    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        cute.arch.barrier()
        if tidx < 32 * _F32_NKB:
            rS = cute.make_rmem_tensor((4,), cutlass.Uint8)
            cute.autovec_copy(sOut[(None, tidx % 32, tidx // 32)], rS)
            cute.autovec_copy(rS, gS[(None, tidx % 32, tidx // 32, bx, sub, mb)])


@cute.jit
def _mxfp8_launch_fp32(mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor,
                       offs: Optional[cute.Tensor],
                       USE_RCEIL: cutlass.Constexpr[bool],
                       BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool]):
    M, K = mX.shape

    # (VEC, col_lane, row_off, warp, i, col_tile, jgrp, scale_block_row)
    gX = cute.make_tensor(
        mX.iterator,
        cute.make_layout(
            (_F32_VEC, _F32_LPR, _F32_RPS, _F32_NWARPS, _F32_NSTEPS,
             K // _F32_TILE_COLS, 4, M // 128),
            stride=(1, _F32_VEC, K, _F32_RPS * K, _F32_NT * K, _F32_TILE_COLS,
                    32 * K, 128 * K),
        ),
    )
    qp = cute.recast_ptr(mQ.iterator, dtype=cutlass.Uint32)
    gQ = cute.make_tensor(
        qp,
        cute.make_layout(
            (_F32_VEC // 4, _F32_LPR, _F32_RPS, _F32_NWARPS, _F32_NSTEPS,
             K // _F32_TILE_COLS, 4, M // 128),
            stride=(
                1,
                _F32_VEC // 4,
                K // 4,
                _F32_RPS * K // 4,
                _F32_NT * K // 4,
                _F32_TILE_COLS // 4,
                32 * K // 4,
                128 * K // 4,
            ),
        ),
    )
    if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
        # (i*4+c, j_in_group, col_tile, jgrp, scale_block_row)
        gS = cute.make_tensor(
            mS.iterator,
            cute.make_layout(
                (4, 32, _F32_NKB, K // _F32_TILE_COLS, 4, M // 128),
                stride=(1, 16, 512, _F32_NKB * 512, 4, 4 * K),
            ),
        )
    else:
        # mS is already the plain (M, k_blocks) uint8 buffer -- the kernel
        # indexes it directly with absolute (row, k_block) coordinates.
        gS = mS

    _mxfp8_kernel_fp32(
        gX, gQ, gS, offs, USE_RCEIL=USE_RCEIL,
        BLOCKED_SCALE_OUTPUT=BLOCKED_SCALE_OUTPUT,
    ).launch(
        grid=(K // _F32_TILE_COLS, M // 32, 1),
        block=(_F32_NTHREADS, 1, 1),
        smem=(_F32_SMEM_BYTES + 16 if cutlass.const_expr(BLOCKED_SCALE_OUTPUT) else 0),
    )


_compiled_cache_fp32 = {}


def _get_compiled_fp32(
    scaling_mode: str, blocked_scale_output: bool, x, q, s_tensor, offs, M, K,
):
    # Unlike the bf16 path, this compiles per-(M, K) shape rather than using
    # dynamic-shape markers.
    has_offs = offs is not None
    key = (scaling_mode, has_offs, blocked_scale_output, M, K)
    compiled = _compiled_cache_fp32.get(key)
    if compiled is None:
        x_ct = from_dlpack(x, assumed_align=_F32_ALIGN, enable_tvm_ffi=True)
        q_ct = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True)
        s_ct = from_dlpack(s_tensor, assumed_align=16, enable_tvm_ffi=True)
        if has_offs:
            offs_ct = from_dlpack(
                offs, assumed_align=4, enable_tvm_ffi=True
            ).mark_layout_dynamic()
            options = "--enable-tvm-ffi --enable-assertions"
        else:
            offs_ct = None
            options = "--enable-tvm-ffi"
        compiled = cute.compile(
            _mxfp8_launch_fp32, x_ct, q_ct, s_ct, offs_ct,
            USE_RCEIL=(scaling_mode == "rceil"),
            BLOCKED_SCALE_OUTPUT=blocked_scale_output,
            options=options,
        )
        _compiled_cache_fp32[key] = compiled
    return compiled


def _mxfp8_quantize_bf16(
    x: torch.Tensor,
    block_size: int,
    scaling_mode: str,
    blocked_scale_output: bool,
    offs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    assert M % 128 == 0, "M must be divisible by 128"
    assert K % 128 == 0, "K must be divisible by 128"
    # KSUB=2 (256-wide K-tiles) is the tuned default; KSUB=1 (128-wide
    # K-tiles) is the fallback for K that is only a multiple of 128, not
    # 256 -- see the KSUB comment near the top of this file.
    ksub = 2 if K % 256 == 0 else 1

    q_data = torch.empty_strided(
        (M, K), (K, 1), device=x.device, dtype=torch.float8_e4m3fn,
    )
    k_blocks = K // block_size
    if blocked_scale_output:
        padded_scale_rows = ceil_div(M, 128) * 128
        padded_scale_cols = ceil_div(k_blocks, 4) * 4
        scales_u8 = torch.empty(
            padded_scale_rows * padded_scale_cols, device=x.device, dtype=torch.uint8,
        )
        s_tensor = scales_u8.view(torch.int32)
    else:
        scales_u8 = torch.empty((M, k_blocks), device=x.device, dtype=torch.uint8)
        s_tensor = scales_u8

    x_i32 = x.view(torch.int32)
    q_i32 = q_data.view(torch.uint8).view(torch.int32)
    offs_i32 = offs if offs is None else offs.contiguous()

    compiled = _get_compiled(
        scaling_mode, ksub, blocked_scale_output, x_i32, q_i32, s_tensor, offs_i32
    )
    compiled(x_i32, q_i32, s_tensor, offs_i32)

    if blocked_scale_output:
        scales = (
            scales_u8.view(torch.float8_e8m0fnu)
            .view(padded_scale_rows, padded_scale_cols)
        )
    else:
        # Note the dtype asymmetry with the blocked branch above: this
        # returns scales as raw uint8, not float8_e8m0fnu.
        scales = scales_u8
    return q_data, scales


def _mxfp8_quantize_fp32(
    x: torch.Tensor,
    block_size: int,
    scaling_mode: str,
    blocked_scale_output: bool,
    offs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    assert M % 128 == 0, "M must be divisible by 128"
    assert K % 128 == 0, "K must be divisible by 128"

    q_data = torch.empty_strided(
        (M, K), (K, 1), device=x.device, dtype=torch.float8_e4m3fn,
    )
    k_blocks = K // block_size
    if blocked_scale_output:
        padded_scale_rows = ceil_div(M, 128) * 128
        padded_scale_cols = ceil_div(k_blocks, 4) * 4
        scales_u8 = torch.empty(
            padded_scale_rows * padded_scale_cols, device=x.device, dtype=torch.uint8,
        )
        s_tensor = scales_u8
    else:
        scales_u8 = torch.empty((M, k_blocks), device=x.device, dtype=torch.uint8)
        s_tensor = scales_u8

    offs_c = offs if offs is None else offs.contiguous()

    compiled = _get_compiled_fp32(
        scaling_mode, blocked_scale_output, x, q_data, s_tensor, offs_c, M, K
    )
    compiled(x, q_data, s_tensor, offs_c)

    if blocked_scale_output:
        scales = (
            scales_u8.view(torch.float8_e8m0fnu)
            .view(padded_scale_rows, padded_scale_cols)
        )
    else:
        # Same dtype asymmetry as the bf16 path above: uint8, not
        # float8_e8m0fnu, for the non-blocked case.
        scales = scales_u8
    return q_data, scales


@torch.no_grad()
def mxfp8_quantize_cutedsl_2d_1x32(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D bf16 or fp32 tensor to MXFP8 (dim-0 / rowwise, 1x32
    blocks). Dispatches on ``x.dtype`` to one of two structurally different
    kernels: bf16 uses hand-rolled packed-bf16x2 inline PTX; fp32 uses the
    ``cvt.{rp,rz}.ue8m0x2.f32`` hardware conversion instruction directly.
    There is no separate dtype argument -- ``x`` already carries its own
    dtype. Functional call convention: allocates and returns outputs, does
    not write into caller-provided buffers. There is no ``stage_count``
    parameter -- there is no TMA pipeline here to configure the stage count
    of.

    For bf16, K that is a multiple of 256 uses the tuned KSUB=2 (256-wide
    K-tile) kernel; K that is only a multiple of 128 falls back to an
    untuned KSUB=1 (128-wide K-tile) specialization of the same kernel.
    The fp32 kernel only ever needs K % 128 == 0, with no such fallback.

    Args:
        x: Input tensor of shape (M, K), bfloat16 or float32.
        block_size: Block size for quantization along K (only 32 supported).
        scaling_mode: Scaling mode ("floor" or "rceil").
        blocked_scale_output: Whether to output scales in blocked
            (cuBLAS 128x4 swizzled) layout. If False (default), scales are
            plain row-major, shape (M, K // block_size), written with
            scalar per-element stores rather than the blocked path's
            smem-staged coalesced writes.
        offs: Optional tensor of group end offsets for validation (must have
            group sizes as multiples of 128).

    Returns:
        q_data: Quantized data in row-major layout with shape (M, K).
        scales: If blocked_scale_output, shape
            (ceil_div(M, 128) * 32, ceil_div(K // block_size, 4) * 16).
            Otherwise shape (M, K // block_size).
    """
    assert x.dtype in (torch.bfloat16, torch.float32), (
        "Only bfloat16 or float32 input is supported"
    )
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    assert scaling_mode in ("floor", "rceil"), (
        f"scaling_mode must be 'floor' or 'rceil', got {scaling_mode!r}"
    )
    if offs is not None:
        assert offs.is_cuda, "offs tensor must be CUDA"
        assert offs.dtype == torch.int32, "offs must be int32 tensor"
        assert offs.dim() == 1, "offs must be 1D tensor"

    if x.dtype == torch.bfloat16:
        return _mxfp8_quantize_bf16(x, block_size, scaling_mode, blocked_scale_output, offs)
    return _mxfp8_quantize_fp32(x, block_size, scaling_mode, blocked_scale_output, offs)
