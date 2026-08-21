# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Fused gated activation (SwiGLU) + MXFP8 CuTe DSL kernel for Blackwell,
tuned and validated on GB200/SM100 (the inline PTX deliberately avoids the
sm_100a-only cvt so sm_103a is expected to work; see ``_mul_cvt_2x``). One
pass computes the activation and its RCEIL MXFP8 cast, so the activation
never round-trips through global memory:

    forward:   h     = silu(gate) * up
    backward:  dGate = grad_h * up * d_silu(gate),  dUp = grad_h * silu(gate)

``gated_input`` is bf16 [M, 2K] holding ``gate`` then ``up``; forward outputs
are K wide, backward outputs 2K wide (``[dGate | dUp]``). Rowwise (1x32)
scales, colwise (32x1) scales, or both come from that single read, in the
blocked tcgen05 layouts. Mode flags, chunk geometry, and the ``ACT_PAIR``
activation policy are ``Constexpr`` — each (mode, geometry, device)
combination compiles once and is cached — while M and K are runtime
arguments, so a single specialization serves every shape. Requires M and K
multiples of 128 and
``2*K*M - K - 1 <= INT32_MAX`` (index arithmetic assumes 32-bit offsets).
"""

import functools
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cute import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op

from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _missing_cutedsl_runtime_packages,
)
from torchao.utils import ceil_div

# Kernel geometry constants. SCALE_DIM_X/Y: MXFP8 block sizes (rowwise 1x32,
# colwise 32x1). BUFFS_NUM: smem double-buffer depth. BUFF_DIM_Y: rows per
# pipeline stage buffer. PACK_SIZE: E4M3 bytes packed per b32 store word.
# WAVES: 8B vector groups covering one 1x32 block in the staged rowwise
# swizzled traversal. Chunk shapes (CX, CY) are per-mode Constexpr launch
# parameters; defaults live in _DEFAULT_GEOMETRY below.
SCALE_DIM_Y = 32
SCALE_DIM_X = 32
BUFFS_NUM = 2
BUFF_DIM_Y = 32
PACK_SIZE = 4
WAVES = SCALE_DIM_X // PACK_SIZE

# Chunk geometry (CX, CY, direct) per (is_bwd, rowwise, colwise), tuned on
# GB200; ``direct`` selects the single-pass path over the staged pipeline.
_DEFAULT_GEOMETRY = {
    (False, True, False): (128, 64, True),
    (False, False, True): (64, 64, False),
    (False, True, True): (64, 64, True),
    (True, True, False): (64, 64, True),
    (True, False, True): (64, 64, False),
    (True, True, True): (64, 64, True),
}

_INT32_MAX = 2**31 - 1


# -- Scale indexing, PTX numeric helpers, activation policy (kernel-private) --

EVICT_FIRST = cute.nvgpu.common.CacheEvictionPriority.EVICT_FIRST

# All TMA-accessed shared-memory buffers (G2S destinations and S2G sources)
# must be 128-byte aligned.
TMA_SHMEM_ALIGNMENT = 128


def _gemm_swizzled_scale_idx(row, scale_col, num_scale_col_blocks):
    """Index into the blocked (tcgen05) scale layout expected by MXFP8 GEMMs:
    the logical ``[rows, cols/32]`` scale matrix stored as 512-byte blocks of
    128 rows x 4 scale columns (cuBLAS "128x4 block scaling factors layout");
    ``num_scale_col_blocks`` = ceil(num_scale_cols / 4). For colwise scales
    pass transposed coordinates.
    """
    return (
        ((row >> 7) * num_scale_col_blocks + (scale_col >> 2)) * 512
        + (row & 31) * 16
        + ((row >> 5) & 3) * 4
        + (scale_col & 3)
    )


@cute.jit
def _scale_idx(row, scale_col, ncb, stride, SWIZ: cutlass.Constexpr):
    """Scale-tensor index for one 32-block: the blocked (GEMM-swizzled)
    layout, or the compact row-major ``[rows, cols/32]`` layout."""
    if cutlass.const_expr(SWIZ):
        return _gemm_swizzled_scale_idx(row, scale_col, ncb)
    else:
        return row * stride + scale_col


@dsl_user_op
def _bitcast_i32_to_f32(val: Int32, *, loc=None, ip=None) -> Float32:
    """Bitcast an int32 value to float32 without changing the bit pattern."""
    return Float32(
        mlir_arith.bitcast(T.f32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


# bf16 == top 16 bits of f32, so widening is a free bit-shift.
@dsl_user_op
def _bf16x2_lo_to_f32(bits, *, loc=None, ip=None) -> Float32:
    return _bitcast_i32_to_f32(
        (Int32(bits) & Int32(0xFFFF)) << Int32(16), loc=loc, ip=ip
    )


@dsl_user_op
def _bf16x2_hi_to_f32(bits, *, loc=None, ip=None) -> Float32:
    # `(x >> 16) << 16` == `x & 0xFFFF0000` without a signed literal; the
    # left shift zeroes the arithmetic shift's smeared sign bits.
    return _bitcast_i32_to_f32((Int32(bits) >> Int32(16)) << Int32(16), loc=loc, ip=ip)


# The ``.NaN`` max variants match the standalone quantizers' amax reduction,
# which propagates NaN; plain ``max`` would return the non-NaN operand.
@dsl_user_op
def _max_nan_bf16x2(a, b, *, loc=None, ip=None):
    """NaN-propagating packed bf16x2 max."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.NaN.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _abs_max_nan_bf16x2(a, b, *, loc=None, ip=None):
    """NaN-propagating packed bf16x2 |max|; per-lane sign bits are junk."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _prmt_even(a, b, *, loc=None, ip=None):
    """Select bytes [0,2,4,6] from a pair of b32 words."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, 0x6420;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _prmt_odd(a, b, *, loc=None, ip=None):
    """Select bytes [1,3,5,7] from a pair of b32 words."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, 0x7531;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul_cvt_2x(w0, w1, s, *, loc=None, ip=None):
    """Scale two bf16x2 words by bf16x2 ``s`` and pack four E4M3 bytes into
    one b32 store word. ``cvt.rn.satfinite.e4m3x2.bf16x2`` is missing on some
    Blackwells (GB300's sm_103a), so keep the bf16 multiply for identical
    rounding, widen exactly to f32, and use the portable f32-source cvt.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
                cutlass.Int32(w1).ir_value(loc=loc, ip=ip),
                cutlass.Int32(s).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b16 a, b, t0_lo, t0_hi, t1_lo, t1_hi;\n"
            ".reg .b32 t0, t1;\n"
            ".reg .f32 f0_lo, f0_hi, f1_lo, f1_hi;\n"
            "mul.rn.bf16x2 t0, $1, $3;\n"
            "mul.rn.bf16x2 t1, $2, $3;\n"
            "mov.b32 {t0_lo, t0_hi}, t0;\n"
            "mov.b32 {t1_lo, t1_hi}, t1;\n"
            "cvt.f32.bf16 f0_lo, t0_lo;\n"
            "cvt.f32.bf16 f0_hi, t0_hi;\n"
            "cvt.f32.bf16 f1_lo, t1_lo;\n"
            "cvt.f32.bf16 f1_hi, t1_hi;\n"
            "cvt.rn.satfinite.e4m3x2.f32 a, f0_hi, f0_lo;\n"
            "cvt.rn.satfinite.e4m3x2.f32 b, f1_hi, f1_lo;\n"
            "mov.b32 $0, {a, b}; }",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(hi, lo) f32 -> packed bf16x2 word, RNE. lo occupies bits [15:0]."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _float_to_e8m0(u: Int32) -> Int32:
    """Biased E8M0 byte for a non-negative bf16 amax given as f32 bits.

    Finite: the RCEIL mantissa-carry path, matching what the standalone
    quantizers' ``cvt.rp.ue8m0x2.f32`` (no ``.satfinite``) emits for
    ``amax / 448``. Non-finite: NaN or Inf amax invalidates the block with
    scale byte 255; without the branch, Inf would land on 247 and NaN could
    carry into the sign bit.
    """
    e = cutlass.max(((u + Int32(0x1F0000)) >> 23) - Int32(8), Int32(0))
    if (u & Int32(0x7F800000)) == Int32(0x7F800000):
        e = Int32(255)
    return e


@cute.jit
def _exp2f_rcp_bf16(e: Int32) -> Int32:
    """Inverse scale as bf16 bits (the quantization multiply in
    :func:`_mul_cvt_2x` is bf16x2, not f32), matching the standalone
    quantizers' ``ue8m0(254 - scale_byte)`` reciprocal: 2^(127 - e) for the
    normal range (a clamped byte 0 from a zero or tiny amax descales by
    2^127), and NaN for an invalidated block (byte 255), so every element of
    a NaN/Inf-amax block quantizes to the E4M3 NaN code.
    """
    b = (Int32(254) - e) << 7
    if e == Int32(255):
        b = Int32(0x7FC0)
    return b


@dsl_user_op
def _sigmoidf(x, *, loc=None, ip=None):
    """Sigmoid as ``__frcp_rn(1.0f + __expf(-x))``, emitted as raw PTX,
    instruction for instruction::

        mul.f32        t, x, 0fBFB8AA3B    // -x * log2(e)
        ex2.approx.f32 t, t
        add.f32        t, t, 0f3F800000
        rcp.rn.f32     s, t                // correctly rounded

    No higher-level formulation reproduces ``ex2.approx``, and the ``rcp.rn``
    vs ``div.full`` choice shows at a few output codes per million.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 t;\n"
            "mul.f32 t, $1, 0fBFB8AA3B;\n"
            "ex2.approx.f32 t, t;\n"
            "add.f32 t, t, 0f3F800000;\n"
            "rcp.rn.f32 $0, t; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _silu_pair(x0, x1, lin0, lin1, g0, g1, IS_BWD: cutlass.Constexpr):
    """SwiGLU activation policy for a pair of elements: ``x`` is the
    activation input, ``lin`` the linear multiplier, ``g`` the incoming
    gradient (ignored unless IS_BWD)::

        s = sigmoid(x); act = x * s
        forward:  out_act = act * lin
        backward: dact = x*s*(1-s) + s      (contracted into one FMA)
                  out_act = (dact * g) * lin,  out_gate = act * g

    Returns f32 ``(out_act0, out_act1, out_gate0, out_gate1)`` (gate pair
    zero for forward); callers must round to BF16 immediately, before any
    amax, caching, or quantization.

    In the module docstring's terms: ``x`` = gate, ``lin`` = up, ``g`` =
    grad_h; backward's ``out_act`` half holds dGate and ``out_gate`` holds
    dUp — the *_gate-suffixed symbols throughout the kernel carry dUp (stored
    at offset K), not a gate gradient.
    """
    one = cutlass.Float32(1.0)
    s0 = _sigmoidf(x0)
    s1 = _sigmoidf(x1)
    act0, act1 = cute.arch.mul_packed_f32x2((x0, x1), (s0, s1))
    if cutlass.const_expr(IS_BWD):
        om0, om1 = cute.arch.sub_packed_f32x2((one, one), (s0, s1))
        dact0, dact1 = cute.arch.fma_packed_f32x2((act0, act1), (om0, om1), (s0, s1))
        t0, t1 = cute.arch.mul_packed_f32x2((dact0, dact1), (g0, g1))
        oa0, oa1 = cute.arch.mul_packed_f32x2((t0, t1), (lin0, lin1))
        og0, og1 = cute.arch.mul_packed_f32x2((act0, act1), (g0, g1))
        return oa0, oa1, og0, og1
    else:
        oa0, oa1 = cute.arch.mul_packed_f32x2((act0, act1), (lin0, lin1))
        return oa0, oa1, cutlass.Float32(0.0), cutlass.Float32(0.0)


def _load_direct_inputs(gXv, gLinv, gGradv, half, blk, bx, grow, IS_BWD):
    """Issue one stage's evict-first vector input loads into fresh rmem
    tensors; trace-time helper for the direct path. ``rg`` is None in
    forward mode."""
    rx = cute.make_rmem_tensor(8, cutlass.Int32)
    rl = cute.make_rmem_tensor(8, cutlass.Int32)
    cute.autovec_copy(
        gXv[(None, half, blk, bx, grow)],
        rx,
        l1c_evict_priority=EVICT_FIRST,
    )
    cute.autovec_copy(
        gLinv[(None, half, blk, bx, grow)],
        rl,
        l1c_evict_priority=EVICT_FIRST,
    )
    rg = None
    if IS_BWD:
        rg = cute.make_rmem_tensor(8, cutlass.Int32)
        cute.autovec_copy(
            gGradv[(None, half, blk, bx, grow)],
            rg,
            l1c_evict_priority=EVICT_FIRST,
        )
    return rx, rl, rg


@cute.jit
def _fold_amax(am: cutlass.Int32) -> cutlass.Int32:
    """Reduce a packed bf16x2 amax word to bf16 amax bits in [15:0].

    The input's per-lane sign bits are junk (see ``_abs_max_nan_bf16x2``); mask
    them, then fold the two lanes with the NaN-propagating max.
    """
    am = am & cutlass.Int32(0x7FFF7FFF)
    am = _max_nan_bf16x2(am, am >> 16)
    return am & cutlass.Int32(0xFFFF)


@cute.kernel
def gated_act_mxfp8_kernel(
    atom_x: cute.CopyAtom,
    gX: cute.Tensor,
    atom_lin: cute.CopyAtom,
    gLin: cute.Tensor,
    atom_grad: cute.CopyAtom,
    gGrad: cute.Tensor,
    gXv: cute.Tensor,
    gLinv: cute.Tensor,
    gGradv: cute.Tensor,
    atom_row_act: cute.CopyAtom,
    gRowAct: cute.Tensor,
    atom_row_gate: cute.CopyAtom,
    gRowGate: cute.Tensor,
    atom_col_act: cute.CopyAtom,
    gColAct: cute.Tensor,
    atom_col_gate: cute.CopyAtom,
    gColGate: cute.Tensor,
    mRS: cute.Tensor,
    mCS: cute.Tensor,
    rs_ncb: cutlass.Int32,
    rs_stride: cutlass.Int32,
    rgate_scol_off: cutlass.Int32,
    cs_ncb: cutlass.Int32,
    cs_stride: cutlass.Int32,
    cgate_col_off: cutlass.Int32,
    IS_BWD: cutlass.Constexpr,
    ROWWISE: cutlass.Constexpr,
    COLWISE: cutlass.Constexpr,
    SWIZ: cutlass.Constexpr,
    ACT_PAIR: cutlass.Constexpr,
    DIRECT: cutlass.Constexpr,
    CX: cutlass.Constexpr,
    CY: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
):
    """The gated-activation MXFP8 kernel, specialized entirely at compile
    time. Unused parameters are never referenced: disabled directions receive
    dummies from the launcher, and the non-selected DIRECT/staged path's
    views are simply ignored."""
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    IS_CACHED_ACT_OP = ROWWISE and COLWISE
    ONLY_COLWISE = COLWISE and not ROWWISE
    OUT_HALVES_C = 2 if IS_BWD else 1
    # DIRECT reads inputs straight into registers and computes the activation
    # exactly once (the staged two-pass structure was latency-bound on
    # GB200); the colwise side goes through a padded XOR-swizzled transposed
    # shared tile instead of recomputation.
    COLWISE_WF = THREADS // CX  # thread rows stacked along each column
    N_STAGES = CY // BUFF_DIM_Y
    LOG2_CX = CX.bit_length() - 1
    # 1x32 scale blocks per chunk row (staged rowwise: also threads per row;
    # direct: two threads split each block).
    TXR = CX // SCALE_DIM_X
    LOG2_TXR = TXR.bit_length() - 1
    BUFF_BYTES = BUFF_DIM_Y * CX * 2
    # Direct colwise path: bf16x2 words hold two adjacent columns, so the
    # padded transposed tile is indexed by column pair.
    PAIRS = CX // 2
    LOG2_PAIRS = PAIRS.bit_length() - 1
    SP_PAD = BUFF_DIM_Y + 4
    if cutlass.const_expr(DIRECT and COLWISE):
        # Reader: TPP threads per (column pair, output half); each thread
        # owns NCH of the pair's eight 16B row-chunks.
        assert THREADS % (PAIRS * OUT_HALVES_C) == 0
        TPP = THREADS // (PAIRS * OUT_HALVES_C)
        assert TPP & (TPP - 1) == 0 and 1 <= TPP <= 8
        LOG2_TPP = TPP.bit_length() - 1
        NCH = 8 // TPP
    # Prefetch only where the reader phase hides the load latency; measured
    # on GB200, prefetch costs ~16 registers and regressed rowwise-only 13%.
    DO_PREFETCH = COLWISE
    # Direct loads have no mbarrier chain ordering output-buffer reuse against
    # outstanding TMA-store reads; only the colwise path's pre-reader barrier
    # can cover that, so rowwise-only direct needs one buffer per stage.
    assert not DIRECT or COLWISE or N_STAGES <= BUFFS_NUM
    # The staged path has no in-loop TMA-store drain: output buffers are safe
    # only because the post-loop drain runs before any reuse, which requires
    # the stage count to fit the double buffer. The launcher rejects
    # violating geometries; this is the trace-time backstop.
    assert DIRECT or N_STAGES <= BUFFS_NUM
    # Staged rowwise consumes the colwise pass's cached activation; there is
    # no standalone staged rowwise compute.
    assert DIRECT or COLWISE or not ROWWISE

    # Shared buffers get a value-typed view (TMA partitioning, scalar access)
    # plus a b32-word view (vectorized 8B loads, packed 4B stores).
    smem = cutlass.utils.SmemAllocator()

    in_elems = BUFF_DIM_Y * CX * BUFFS_NUM
    if cutlass.const_expr(not DIRECT):
        in_layout = cute.make_layout(
            (BUFF_DIM_Y, CX, BUFFS_NUM),
            stride=(CX, 1, BUFF_DIM_Y * CX),
        )
        # 8B-vector view: (lane pair, 8B group, row, buff).
        in_word_layout = cute.make_layout(
            (2, CX // 4, BUFF_DIM_Y, BUFFS_NUM),
            stride=(1, 2, CX // 2, BUFF_DIM_Y * CX // 2),
        )

        px = smem.allocate(in_elems * 2, byte_alignment=TMA_SHMEM_ALIGNMENT)
        sX = cute.make_tensor(cute.recast_ptr(px, dtype=cutlass.BFloat16), in_layout)
        sXw = cute.make_tensor(cute.recast_ptr(px, dtype=cutlass.Int32), in_word_layout)
        pl = smem.allocate(in_elems * 2, byte_alignment=TMA_SHMEM_ALIGNMENT)
        sLin = cute.make_tensor(cute.recast_ptr(pl, dtype=cutlass.BFloat16), in_layout)
        sLinw = cute.make_tensor(
            cute.recast_ptr(pl, dtype=cutlass.Int32), in_word_layout
        )
        if cutlass.const_expr(IS_BWD):
            pg = smem.allocate(in_elems * 2, byte_alignment=TMA_SHMEM_ALIGNMENT)
            sGrad = cute.make_tensor(
                cute.recast_ptr(pg, dtype=cutlass.BFloat16), in_layout
            )
        # Bidirectional mode: the columnwise pass caches the post-activation
        # BF16 in the input buffers so the rowwise pass never recomputes it
        # (which is why staged ROWWISE without COLWISE is invalid — asserted
        # above).
        cached_act, cached_actw = sX, sXw
        cached_gate, cached_gatew = sLin, sLinw

    if cutlass.const_expr(ROWWISE):
        row_out_layout = cute.make_layout(
            (BUFF_DIM_Y, CX, BUFFS_NUM),
            stride=(CX, 1, BUFF_DIM_Y * CX),
        )
        row_out_word_layout = cute.make_layout(
            (BUFF_DIM_Y, CX // 4, BUFFS_NUM),
            stride=(CX // 4, 1, BUFF_DIM_Y * CX // 4),
        )
        # Quad view for the direct path's 16B vector stores.
        row_out_quad_layout = cute.make_layout(
            (4, CX // 16, BUFF_DIM_Y, BUFFS_NUM),
            stride=(1, 4, CX // 4, BUFF_DIM_Y * CX // 4),
        )
        pra = smem.allocate(in_elems, byte_alignment=TMA_SHMEM_ALIGNMENT)
        sRowAct = cute.make_tensor(
            cute.recast_ptr(pra, dtype=cutlass.Float8E4M3FN), row_out_layout
        )
        sRowActw = cute.make_tensor(
            cute.recast_ptr(pra, dtype=cutlass.Int32), row_out_word_layout
        )
        sRowQuad = cute.make_tensor(
            cute.recast_ptr(pra, dtype=cutlass.Int32), row_out_quad_layout
        )
        if cutlass.const_expr(IS_BWD):
            prg = smem.allocate(in_elems, byte_alignment=TMA_SHMEM_ALIGNMENT)
            sRowGate = cute.make_tensor(
                cute.recast_ptr(prg, dtype=cutlass.Float8E4M3FN), row_out_layout
            )
            sRowGatew = cute.make_tensor(
                cute.recast_ptr(prg, dtype=cutlass.Int32), row_out_word_layout
            )
            sRowGateQuad = cute.make_tensor(
                cute.recast_ptr(prg, dtype=cutlass.Int32), row_out_quad_layout
            )
    if cutlass.const_expr(COLWISE):
        # Transposed (output column, row) tiles; staging + TMA store beats
        # direct scattered global stores (measured: direct stores pushed
        # L1TEX to 97% of peak and cost +12% on bwd_rc).
        col_out_layout = cute.make_layout(
            (CX, BUFF_DIM_Y, BUFFS_NUM),
            stride=(BUFF_DIM_Y, 1, CX * BUFF_DIM_Y),
        )
        col_out_word_layout = cute.make_layout(
            (CX, BUFF_DIM_Y // 4, BUFFS_NUM),
            stride=(BUFF_DIM_Y // 4, 1, CX * BUFF_DIM_Y // 4),
        )
        pca = smem.allocate(in_elems, byte_alignment=TMA_SHMEM_ALIGNMENT)
        sColAct = cute.make_tensor(
            cute.recast_ptr(pca, dtype=cutlass.Float8E4M3FN), col_out_layout
        )
        sColActw = cute.make_tensor(
            cute.recast_ptr(pca, dtype=cutlass.Int32), col_out_word_layout
        )
        if cutlass.const_expr(IS_BWD):
            pcg = smem.allocate(in_elems, byte_alignment=TMA_SHMEM_ALIGNMENT)
            sColGate = cute.make_tensor(
                cute.recast_ptr(pcg, dtype=cutlass.Float8E4M3FN), col_out_layout
            )
            sColGatew = cute.make_tensor(
                cute.recast_ptr(pcg, dtype=cutlass.Int32), col_out_word_layout
            )
        if cutlass.const_expr(DIRECT):
            # Reader slice view: each of a column's TPP threads owns NCH
            # contiguous words.
            col_slice_layout = cute.make_layout(
                (NCH, TPP, CX, BUFFS_NUM),
                stride=(1, NCH, BUFF_DIM_Y // 4, CX * BUFF_DIM_Y // 4),
            )
            sColSliceA = cute.make_tensor(
                cute.recast_ptr(pca, dtype=cutlass.Int32), col_slice_layout
            )
            if cutlass.const_expr(IS_BWD):
                sColSliceG = cute.make_tensor(
                    cute.recast_ptr(pcg, dtype=cutlass.Int32), col_slice_layout
                )

    if cutlass.const_expr(DIRECT and COLWISE):
        # Padded transposed staging between the compute pass and the colwise
        # reader; a pair's 32 rows are eight 16B chunks at addr(pair, row) =
        # pair*SP_PAD + 4*((row>>2) ^ ((pair>>3 & 3) << 1)) + (row&3). The
        # padding plus XOR swizzle make both the writer's word stores and the
        # reader's 16B loads shared-memory bank-conflict-free.
        ppad = smem.allocate(PAIRS * SP_PAD * OUT_HALVES_C * 4, byte_alignment=16)
        sPadW = cute.make_tensor(
            cute.recast_ptr(ppad, dtype=cutlass.Int32),
            cute.make_layout(
                (PAIRS, 8, 4, OUT_HALVES_C),
                stride=(SP_PAD, 4, 1, PAIRS * SP_PAD),
            ),
        )
        sPadR = cute.make_tensor(
            cute.recast_ptr(ppad, dtype=cutlass.Int32),
            cute.make_layout(
                (4, 8, PAIRS, OUT_HALVES_C),
                stride=(1, 4, SP_PAD, PAIRS * SP_PAD),
            ),
        )
    if cutlass.const_expr(not DIRECT):
        mbar = smem.allocate_array(cutlass.Int64, N_STAGES, byte_alignment=8)
    if cutlass.const_expr(ONLY_COLWISE and not DIRECT):
        # Partial-amax exchange between the COLWISE_WF thread rows; the
        # single sSubAmax slot per column assumes exactly one non-zero row.
        assert COLWISE_WF == 2
        psub = smem.allocate(CX * 4, byte_alignment=4)
        sSubAmax = cute.make_tensor(
            cute.recast_ptr(psub, dtype=cutlass.Int32), cute.make_layout(CX)
        )

    if cutlass.const_expr(not DIRECT):
        # tma_partition takes the no-multicast CTA coord (0) and layout; the
        # smem view groups its buffer modes first, the gmem view is the tile
        # shape ``cute.zipped_divide`` produced in the launcher.
        tXs, tXg = cpasync.tma_partition(
            atom_x, 0, cute.make_layout(1), cute.group_modes(sX, 0, 2), gX
        )
        tLs, tLg = cpasync.tma_partition(
            atom_lin, 0, cute.make_layout(1), cute.group_modes(sLin, 0, 2), gLin
        )
        if cutlass.const_expr(IS_BWD):
            tGs, tGg = cpasync.tma_partition(
                atom_grad, 0, cute.make_layout(1), cute.group_modes(sGrad, 0, 2), gGrad
            )
    if cutlass.const_expr(ROWWISE):
        tRAs, tRAg = cpasync.tma_partition(
            atom_row_act,
            0,
            cute.make_layout(1),
            cute.group_modes(sRowAct, 0, 2),
            gRowAct,
        )
        if cutlass.const_expr(IS_BWD):
            tRGs, tRGg = cpasync.tma_partition(
                atom_row_gate,
                0,
                cute.make_layout(1),
                cute.group_modes(sRowGate, 0, 2),
                gRowGate,
            )
    if cutlass.const_expr(COLWISE):
        tCAs, tCAg = cpasync.tma_partition(
            atom_col_act,
            0,
            cute.make_layout(1),
            cute.group_modes(sColAct, 0, 2),
            gColAct,
        )
        if cutlass.const_expr(IS_BWD):
            tCGs, tCGg = cpasync.tma_partition(
                atom_col_gate,
                0,
                cute.make_layout(1),
                cute.group_modes(sColGate, 0, 2),
                gColGate,
            )

    if cutlass.const_expr(not DIRECT):
        # Barrier arrive count = CTA thread count (every thread arrives once
        # per stage; TMA bytes are tracked on top via ``expect_tx``); the
        # init fence makes the initialization visible to the TMA async proxy.
        if tidx == 0:
            for i in cutlass.range_constexpr(N_STAGES):
                cute.arch.mbarrier_init(mbar + i, THREADS)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

    if cutlass.const_expr(DIRECT):
        # Stage-invariant mapping: two threads split each 1x32 block.
        half = tidx & 1
        blk = (tidx >> 1) & cutlass.Int32(TXR - 1)
        row = tidx >> (1 + LOG2_TXR)
        if cutlass.const_expr(COLWISE):
            # sPad swizzle key; the thread's pairs are 16*blk + 8*half + j,
            # so pair bits 3-4 (hence the key) are stage-invariant.
            wchk = (row >> 2) ^ (((blk * 2 + half) & cutlass.Int32(3)) * 2)
            wrow = row & cutlass.Int32(3)
        rxs = [None] * N_STAGES
        rls = [None] * N_STAGES
        rgs = [None] * N_STAGES
        if cutlass.const_expr(DO_PREFETCH):
            grow0 = by * cutlass.Int32(CY) + row
            rxs[0], rls[0], rgs[0] = _load_direct_inputs(
                gXv, gLinv, gGradv, half, blk, bx, grow0, IS_BWD
            )

    row_tile0 = by * N_STAGES
    if cutlass.const_expr(not DIRECT):
        copies = (
            (atom_x, tXg[(None, (row_tile0, bx))], tXs[(None, 0)]),
            (atom_lin, tLg[(None, (row_tile0, bx))], tLs[(None, 0)]),
        )
        if cutlass.const_expr(IS_BWD):
            copies += ((atom_grad, tGg[(None, (row_tile0, bx))], tGs[(None, 0)]),)
        # The TMA copies must issue under a warp-uniform predicate (a
        # single-thread predicate deadlocks the DSL's issuing-lane election);
        # thread 0 expects the combined byte count, every thread arrives once.
        if warp_idx == 0:
            for atom, g, s in copies:
                cute.copy(atom, g, s, tma_bar_ptr=mbar)
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, len(copies) * BUFF_BYTES)
        else:
            cute.arch.mbarrier_arrive(mbar)

    for stage in cutlass.range_constexpr(N_STAGES):
        buff = stage % BUFFS_NUM
        row_tile = by * N_STAGES + stage

        if cutlass.const_expr((not DIRECT) and stage + 1 < N_STAGES):
            # Prefetch the next stage's inputs. Output smem is never reused
            # inside this loop (N_STAGES <= BUFFS_NUM, asserted above), so
            # TMA-store groups need no drain before the post-loop
            # wait_group(0); the input refill is ordered by the
            # end-of-stage sync.
            nbuff = (stage + 1) % BUFFS_NUM
            nmbar = mbar + (stage + 1)
            row_tile_n = by * N_STAGES + (stage + 1)
            copies = (
                (atom_x, tXg[(None, (row_tile_n, bx))], tXs[(None, nbuff)]),
                (atom_lin, tLg[(None, (row_tile_n, bx))], tLs[(None, nbuff)]),
            )
            if cutlass.const_expr(IS_BWD):
                copies += (
                    (atom_grad, tGg[(None, (row_tile_n, bx))], tGs[(None, nbuff)]),
                )
            if warp_idx == 0:
                for atom, g, s in copies:
                    cute.copy(atom, g, s, tma_bar_ptr=nmbar)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(nmbar, len(copies) * BUFF_BYTES)
            else:
                cute.arch.mbarrier_arrive(nmbar)

        if cutlass.const_expr(not DIRECT):
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_wait(mbar + stage, 0)

        # -- Columnwise producer pass (staged path) -------------------------
        if cutlass.const_expr(COLWISE and not DIRECT):
            col = tidx & cutlass.Int32(CX - 1)
            ROWS_PER_THREAD = SCALE_DIM_Y // COLWISE_WF
            NWORDS = ROWS_PER_THREAD // 2

            # 1. Compute post-activation values, round to BF16, find amax.
            w_act = [None] * NWORDS
            w_gate = [None] * NWORDS
            am_act = cutlass.Int32(0)
            am_gate = cutlass.Int32(0)
            for j in cutlass.range_constexpr(NWORDS):
                # Contiguous per-thread row split: the amax reduction is
                # order-independent, and contiguous rows pack into 4B words.
                if cutlass.const_expr(COLWISE_WF == 1):
                    rlo = 2 * j
                    rhi = 2 * j + 1
                else:
                    ty = tidx >> LOG2_CX
                    rlo = ty * ROWS_PER_THREAD + 2 * j
                    rhi = rlo + 1
                x0 = sX[(rlo, col, buff)].to(cutlass.Float32)
                x1 = sX[(rhi, col, buff)].to(cutlass.Float32)
                l0 = sLin[(rlo, col, buff)].to(cutlass.Float32)
                l1 = sLin[(rhi, col, buff)].to(cutlass.Float32)
                if cutlass.const_expr(IS_BWD):
                    g0 = sGrad[(rlo, col, buff)].to(cutlass.Float32)
                    g1 = sGrad[(rhi, col, buff)].to(cutlass.Float32)
                else:
                    g0 = cutlass.Float32(0.0)
                    g1 = cutlass.Float32(0.0)
                oa0, oa1, og0, og1 = ACT_PAIR(x0, x1, l0, l1, g0, g1, IS_BWD)
                # Numerical truncation to the input type before anything else.
                wa = _pack_bf16x2(oa1, oa0)
                w_act[j] = wa
                am_act = _abs_max_nan_bf16x2(am_act, wa)
                if cutlass.const_expr(IS_BWD):
                    wg = _pack_bf16x2(og1, og0)
                    w_gate[j] = wg
                    am_gate = _abs_max_nan_bf16x2(am_gate, wg)
                if cutlass.const_expr(IS_CACHED_ACT_OP):
                    cached_act[(rlo, col, buff)] = oa0.to(cutlass.BFloat16)
                    cached_act[(rhi, col, buff)] = oa1.to(cutlass.BFloat16)
                    if cutlass.const_expr(IS_BWD):
                        cached_gate[(rlo, col, buff)] = og0.to(cutlass.BFloat16)
                        cached_gate[(rhi, col, buff)] = og1.to(cutlass.BFloat16)

            am_act = _fold_amax(am_act)
            if cutlass.const_expr(IS_BWD):
                am_gate = _fold_amax(am_gate)

            # Reduce partial amaxes across the two thread rows (colwise-only
            # staged always launches THREADS == 2*CX, so ty is 0 or 1; the
            # exchange does not generalize to more rows).
            if cutlass.const_expr(ONLY_COLWISE):
                ty = tidx >> LOG2_CX
                if ty > 0:
                    sSubAmax[col] = am_act
                cute.arch.sync_threads()
                if ty == 0:
                    am_act = _max_nan_bf16x2(am_act, sSubAmax[col])
                    sSubAmax[col] = am_act
                cute.arch.sync_threads()
                am_act = sSubAmax[col]
                if cutlass.const_expr(IS_BWD):
                    # The previous reads must complete before the rewrite.
                    cute.arch.sync_threads()
                    if ty > 0:
                        sSubAmax[col] = am_gate
                    cute.arch.sync_threads()
                    if ty == 0:
                        am_gate = _max_nan_bf16x2(am_gate, sSubAmax[col])
                        sSubAmax[col] = am_gate
                    cute.arch.sync_threads()
                    am_gate = sSubAmax[col]

            # 2. Compute and store the E8M0 scales (one per 32x1 block).
            out_col = bx * cutlass.Int32(CX) + col
            mcol = row_tile
            u_act = am_act << 16
            e_act = _float_to_e8m0(u_act)
            sidx = _scale_idx(out_col, mcol, cs_ncb, cs_stride, SWIZ)
            if cutlass.const_expr(ONLY_COLWISE):
                if tidx < cutlass.Int32(CX):
                    mCS[sidx] = e_act.to(cutlass.Uint8)
            else:
                mCS[sidx] = e_act.to(cutlass.Uint8)
            r_act = _exp2f_rcp_bf16(e_act) * cutlass.Int32(0x10001)

            if cutlass.const_expr(IS_BWD):
                u_gate = am_gate << 16
                e_gate = _float_to_e8m0(u_gate)
                gidx = _scale_idx(
                    out_col + cgate_col_off, mcol, cs_ncb, cs_stride, SWIZ
                )
                if cutlass.const_expr(ONLY_COLWISE):
                    if tidx < cutlass.Int32(CX):
                        mCS[gidx] = e_gate.to(cutlass.Uint8)
                else:
                    mCS[gidx] = e_gate.to(cutlass.Uint8)
                r_gate = _exp2f_rcp_bf16(e_gate) * cutlass.Int32(0x10001)

            # 3. Scale and pack into the transposed shared output tile.
            if cutlass.const_expr(COLWISE_WF == 1):
                wbase = 0
            else:
                wbase = (tidx >> LOG2_CX) * (NWORDS // 2)
            for w in cutlass.range_constexpr(NWORDS // 2):
                sColActw[(col, wbase + w, buff)] = _mul_cvt_2x(
                    w_act[2 * w], w_act[2 * w + 1], r_act
                )
                if cutlass.const_expr(IS_BWD):
                    sColGatew[(col, wbase + w, buff)] = _mul_cvt_2x(
                        w_gate[2 * w], w_gate[2 * w + 1], r_gate
                    )

        # -- Direct single-pass compute -------------------------------------
        if cutlass.const_expr(DIRECT):
            # One contiguous 32B evict-first load per stream (each input is
            # touched exactly once); the activation feeds both orientations.
            grow = by * cutlass.Int32(CY) + cutlass.Int32(stage * BUFF_DIM_Y) + row

            # With prefetch, issue the NEXT stage's loads so their latency
            # hides behind this stage's compute and reader phases.
            ld = stage + 1 if DO_PREFETCH else stage
            if cutlass.const_expr(ld < N_STAGES):
                grow_ld = by * cutlass.Int32(CY) + cutlass.Int32(ld * BUFF_DIM_Y) + row
                rxs[ld], rls[ld], rgs[ld] = _load_direct_inputs(
                    gXv, gLinv, gGradv, half, blk, bx, grow_ld, IS_BWD
                )
            rx = rxs[stage]
            rl = rls[stage]
            rg = rgs[stage]

            am_act = cutlass.Int32(0)
            am_gate = cutlass.Int32(0)
            w_act = [None] * 8
            w_gate = [None] * 8
            for j in cutlass.range_constexpr(8):
                x0 = _bf16x2_lo_to_f32(rx[j])
                x1 = _bf16x2_hi_to_f32(rx[j])
                l0 = _bf16x2_lo_to_f32(rl[j])
                l1 = _bf16x2_hi_to_f32(rl[j])
                if cutlass.const_expr(IS_BWD):
                    g0 = _bf16x2_lo_to_f32(rg[j])
                    g1 = _bf16x2_hi_to_f32(rg[j])
                else:
                    g0 = cutlass.Float32(0.0)
                    g1 = cutlass.Float32(0.0)
                oa0, oa1, og0, og1 = ACT_PAIR(x0, x1, l0, l1, g0, g1, IS_BWD)
                w_act[j] = _pack_bf16x2(oa1, oa0)
                am_act = _abs_max_nan_bf16x2(am_act, w_act[j])
                if cutlass.const_expr(IS_BWD):
                    w_gate[j] = _pack_bf16x2(og1, og0)
                    am_gate = _abs_max_nan_bf16x2(am_gate, w_gate[j])
                # Park each column-pair word in the swizzled transposed tile
                # for the columnwise reader.
                if cutlass.const_expr(COLWISE):
                    pair = (
                        blk * cutlass.Int32(16)
                        + half * cutlass.Int32(8)
                        + cutlass.Int32(j)
                    )
                    sPadW[(pair, wchk, wrow, 0)] = w_act[j]
                    if cutlass.const_expr(IS_BWD):
                        sPadW[(pair, wchk, wrow, 1)] = w_gate[j]

        # -- Direct columnwise reader, part 1 (issue) -------------------------
        if cutlass.const_expr(DIRECT and COLWISE):
            if cutlass.const_expr(stage >= BUFFS_NUM):
                # TMA stores committed at stage - BUFFS_NUM may still read
                # the tiles this stage overwrites: warp 0 drains them, the
                # sync below releases everyone's stores.
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(BUFFS_NUM - 1, read=True)
            cute.arch.sync_threads()
            # Loads issued here so their latency hides behind the rowwise
            # block; thread tq of a (pair, half) owns NCH 16B row-chunks.
            tq = tidx & cutlass.Int32(TPP - 1)
            cpr = (tidx >> LOG2_TPP) & cutlass.Int32(PAIRS - 1)
            arr = tidx >> (LOG2_TPP + LOG2_PAIRS)
            rsw = ((cpr >> 3) & cutlass.Int32(3)) * 2

            # Undo the writer's chunk swizzle with the same XOR key.
            vs = [cute.make_rmem_tensor(4, cutlass.Int32) for _ in range(NCH)]
            for c in cutlass.range_constexpr(NCH):
                li = tq * cutlass.Int32(NCH) + cutlass.Int32(c)
                cute.autovec_copy(sPadR[(None, li ^ rsw, cpr, arr)], vs[c])

        if cutlass.const_expr(DIRECT and ROWWISE):
            # Butterfly-combine the half-block amaxes; the even lane of each
            # pair owns the scale-byte store.
            scol = bx * cutlass.Int32(TXR) + blk
            am_act = am_act & cutlass.Int32(0x7FFF7FFF)
            am_act = _max_nan_bf16x2(am_act, cute.arch.shuffle_sync_bfly(am_act, 1))
            am_act = _max_nan_bf16x2(am_act, am_act >> 16)
            u_act = (am_act & cutlass.Int32(0xFFFF)) << 16
            e_act = _float_to_e8m0(u_act)
            sidx = _scale_idx(grow, scol, rs_ncb, rs_stride, SWIZ)
            if half == 0:
                mRS[sidx] = e_act.to(cutlass.Uint8)
            r_act = _exp2f_rcp_bf16(e_act) * cutlass.Int32(0x10001)
            qa = cute.make_rmem_tensor(4, cutlass.Int32)
            for q in cutlass.range_constexpr(4):
                qa[q] = _mul_cvt_2x(w_act[2 * q], w_act[2 * q + 1], r_act)
            wq = blk * 2 + half
            cute.autovec_copy(qa, sRowQuad[(None, wq, row, buff)])

            if cutlass.const_expr(IS_BWD):
                am_gate = am_gate & cutlass.Int32(0x7FFF7FFF)
                am_gate = _max_nan_bf16x2(
                    am_gate, cute.arch.shuffle_sync_bfly(am_gate, 1)
                )
                am_gate = _max_nan_bf16x2(am_gate, am_gate >> 16)
                u_gate = (am_gate & cutlass.Int32(0xFFFF)) << 16
                e_gate = _float_to_e8m0(u_gate)
                gidx = _scale_idx(grow, scol + rgate_scol_off, rs_ncb, rs_stride, SWIZ)
                if half == 0:
                    mRS[gidx] = e_gate.to(cutlass.Uint8)
                r_gate = _exp2f_rcp_bf16(e_gate) * cutlass.Int32(0x10001)
                qg = cute.make_rmem_tensor(4, cutlass.Int32)
                for q in cutlass.range_constexpr(4):
                    qg[q] = _mul_cvt_2x(w_gate[2 * q], w_gate[2 * q + 1], r_gate)
                cute.autovec_copy(qg, sRowGateQuad[(None, wq, row, buff)])

        # -- Direct columnwise reader, part 2 (consume) -----------------------
        if cutlass.const_expr(DIRECT and COLWISE):
            ac = cutlass.Int32(0)
            for c in cutlass.range_constexpr(NCH):
                for t in cutlass.range_constexpr(4):
                    ac = _abs_max_nan_bf16x2(ac, vs[c][t])
            ac = ac & cutlass.Int32(0x7FFF7FFF)
            # Butterfly-combine the TPP partial amaxes (bit-identical to a
            # single-thread fold).
            for d in cutlass.range_constexpr(LOG2_TPP):
                ac = _max_nan_bf16x2(ac, cute.arch.shuffle_sync_bfly(ac, 1 << d))

            # Independent per-column scales for the two packed lanes.
            uc0 = (ac & cutlass.Int32(0xFFFF)) << 16
            uc1 = ac & cutlass.Int32(-65536)
            ec0 = _float_to_e8m0(uc0)
            ec1 = _float_to_e8m0(uc1)
            s01 = _exp2f_rcp_bf16(ec0) | (_exp2f_rcp_bf16(ec1) << 16)

            # arr is 0 for every active thread in forward mode, so the
            # gate-half offset term vanishes there.
            c_out_col = bx * cutlass.Int32(CX) + cpr * 2 + arr * cgate_col_off
            if tq == 0:
                ci0 = _scale_idx(c_out_col, row_tile, cs_ncb, cs_stride, SWIZ)
                ci1 = _scale_idx(c_out_col + 1, row_tile, cs_ncb, cs_stride, SWIZ)
                mCS[ci0] = ec0.to(cutlass.Uint8)
                mCS[ci1] = ec1.to(cutlass.Uint8)

            # Quantize and de-interleave the two columns' bytes into the TMA
            # tile; the ~2-way bank conflict is inherent (the tile's 32B
            # column blocks must stay contiguous for TMA).
            fq = cute.make_rmem_tensor(NCH, cutlass.Int32)
            fqb = cute.make_rmem_tensor(NCH, cutlass.Int32)
            for c in cutlass.range_constexpr(NCH):
                a01 = _mul_cvt_2x(vs[c][0], vs[c][1], s01)
                a23 = _mul_cvt_2x(vs[c][2], vs[c][3], s01)
                fq[c] = _prmt_even(a01, a23)
                fqb[c] = _prmt_odd(a01, a23)
            col_local = cpr * 2
            if cutlass.const_expr(IS_BWD):
                if arr == 0:
                    cute.autovec_copy(fq, sColSliceA[(None, tq, col_local, buff)])
                    cute.autovec_copy(fqb, sColSliceA[(None, tq, col_local + 1, buff)])
                else:
                    cute.autovec_copy(fq, sColSliceG[(None, tq, col_local, buff)])
                    cute.autovec_copy(fqb, sColSliceG[(None, tq, col_local + 1, buff)])
            else:
                cute.autovec_copy(fq, sColSliceA[(None, tq, col_local, buff)])
                cute.autovec_copy(fqb, sColSliceA[(None, tq, col_local + 1, buff)])

        if cutlass.const_expr(ROWWISE and not DIRECT):
            row = tidx >> LOG2_TXR
            tx = tidx & cutlass.Int32(TXR - 1)
            bank_group = (tidx & 31) >> 2
            grow = by * cutlass.Int32(CY) + cutlass.Int32(stage * BUFF_DIM_Y) + row

            # Make the columnwise pass's cache writes visible.
            cute.arch.sync_threads()

            # 1. Load the cached post-activation values with the bank-group
            # swizzle; each thread owns a whole 1x32 block per output half.
            am_act = cutlass.Int32(0)
            am_gate = cutlass.Int32(0)
            iv_act = [cute.make_rmem_tensor(2, cutlass.Int32) for _ in range(WAVES)]
            if cutlass.const_expr(IS_BWD):
                iv_gate = [
                    cute.make_rmem_tensor(2, cutlass.Int32) for _ in range(WAVES)
                ]
            for w in cutlass.range_constexpr(WAVES):
                grp = tx * cutlass.Int32(WAVES) + ((cutlass.Int32(w) + bank_group) & 7)
                cute.autovec_copy(cached_actw[(None, grp, row, buff)], iv_act[w])
                if cutlass.const_expr(IS_BWD):
                    cute.autovec_copy(cached_gatew[(None, grp, row, buff)], iv_gate[w])
                am_act = _abs_max_nan_bf16x2(am_act, iv_act[w][0])
                am_act = _abs_max_nan_bf16x2(am_act, iv_act[w][1])
                if cutlass.const_expr(IS_BWD):
                    am_gate = _abs_max_nan_bf16x2(am_gate, iv_gate[w][0])
                    am_gate = _abs_max_nan_bf16x2(am_gate, iv_gate[w][1])

            # 2. One independent E8M0 scale per 1x32 block per output half.
            scol = bx * cutlass.Int32(TXR) + tx
            u_act = _fold_amax(am_act) << 16
            e_act = _float_to_e8m0(u_act)
            sidx = _scale_idx(grow, scol, rs_ncb, rs_stride, SWIZ)
            mRS[sidx] = e_act.to(cutlass.Uint8)
            r_act = _exp2f_rcp_bf16(e_act) * cutlass.Int32(0x10001)
            if cutlass.const_expr(IS_BWD):
                u_gate = _fold_amax(am_gate) << 16
                e_gate = _float_to_e8m0(u_gate)
                gidx = _scale_idx(grow, scol + rgate_scol_off, rs_ncb, rs_stride, SWIZ)
                mRS[gidx] = e_gate.to(cutlass.Uint8)
                r_gate = _exp2f_rcp_bf16(e_gate) * cutlass.Int32(0x10001)

            # 3. Scale and pack, storing with the same swizzled traversal.
            for w in cutlass.range_constexpr(WAVES):
                grp = tx * cutlass.Int32(WAVES) + ((cutlass.Int32(w) + bank_group) & 7)
                sRowActw[(row, grp, buff)] = _mul_cvt_2x(
                    iv_act[w][0], iv_act[w][1], r_act
                )
                if cutlass.const_expr(IS_BWD):
                    sRowGatew[(row, grp, buff)] = _mul_cvt_2x(
                        iv_gate[w][0], iv_gate[w][1], r_gate
                    )

        # Make shared-memory writes visible to the TMA engine, then issue the
        # TMA stores under warp 0's warp-uniform predicate (the DSL elects the
        # issuing lane) and commit them as one bulk group from the same warp.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        if warp_idx == 0:
            if cutlass.const_expr(ROWWISE):
                cute.copy(
                    atom_row_act, tRAs[(None, buff)], tRAg[(None, (row_tile, bx))]
                )
                if cutlass.const_expr(IS_BWD):
                    cute.copy(
                        atom_row_gate, tRGs[(None, buff)], tRGg[(None, (row_tile, bx))]
                    )
            if cutlass.const_expr(COLWISE):
                cute.copy(
                    atom_col_act, tCAs[(None, buff)], tCAg[(None, (bx, row_tile))]
                )
                if cutlass.const_expr(IS_BWD):
                    cute.copy(
                        atom_col_gate, tCGs[(None, buff)], tCGg[(None, (bx, row_tile))]
                    )
            cute.arch.cp_async_bulk_commit_group()

    # Drain every outstanding TMA-store group before the CTA retires: bulk
    # async groups are not implicitly awaited at exit, and the smem tiles the
    # in-flight stores read are deallocated with the CTA (a successor CTA may
    # reuse them). Warp 0 committed every group, so it alone waits.
    if warp_idx == 0:
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    if cutlass.const_expr(not DIRECT):
        cute.arch.sync_threads()
        if tidx == 0:
            # ``mbarrier.inval`` is not exposed by the DSL; emit it as raw PTX.
            for i in cutlass.range_constexpr(N_STAGES):
                llvm.inline_asm(
                    None,
                    [Int32((mbar + i).toint()).ir_value()],
                    "mbarrier.inval.shared::cta.b64 [$0];",
                    "r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )


@cute.jit
def launcher(
    ag: cutlass.Int64,
    agi: cutlass.Int64,
    arq: cutlass.Int64,
    ars: cutlass.Int64,
    acq: cutlass.Int64,
    acs: cutlass.Int64,
    m: cutlass.Int32,
    k: cutlass.Int32,
    stream,
    IS_BWD: cutlass.Constexpr,
    ROWWISE: cutlass.Constexpr,
    COLWISE: cutlass.Constexpr,
    SWIZ: cutlass.Constexpr,
    ACT_PAIR: cutlass.Constexpr,
    DIRECT: cutlass.Constexpr,
    CX: cutlass.Constexpr,
    CY: cutlass.Constexpr,
):
    """Build the TMA views and launch one (CX, CY)-chunk grid.

    ``x``/``lin`` are the two halves of the packed [M, 2K] input; ``grad``
    is the [M, K] incoming gradient. Rowwise outputs land in the row-major
    [M, out_k] tensor at column offsets 0 and K; colwise outputs in the
    transposed [out_k, M] storage at row offsets 0 and K.

    Pointer arguments: ``ag`` = grad_h (backward only, else 0); ``agi`` =
    gated_input; ``arq``/``ars`` = rowwise quantized-output/scale;
    ``acq``/``acs`` = the colwise pair. Disabled directions pass 0.
    """
    OUT_HALVES = 2 if cutlass.const_expr(IS_BWD) else 1
    # Direct: two threads per 1x32 block; staged: one thread per column,
    # except colwise-only which stacks two thread rows per column.
    if cutlass.const_expr(DIRECT):
        THREADS = 2 * CX
    else:
        THREADS = (2 * CX) if cutlass.const_expr(COLWISE and not ROWWISE) else CX
    out_k = OUT_HALVES * k

    # The TMA smem layouts describe one buffer, not the full multi-buffer
    # allocation.
    in_smem = cute.make_layout((BUFF_DIM_Y, CX), stride=(CX, 1))
    in_tiler = (BUFF_DIM_Y, CX)
    col_smem = cute.make_layout((CX, BUFF_DIM_Y), stride=(BUFF_DIM_Y, 1))
    col_tiler = (CX, BUFF_DIM_Y)

    px = cute.make_ptr(cutlass.BFloat16, agi, AddressSpace.gmem, assumed_align=16)
    plin = px + k
    mX = cute.make_tensor(px, cute.make_layout((m, k), stride=(2 * k, 1)))
    mLin = cute.make_tensor(plin, cute.make_layout((m, k), stride=(2 * k, 1)))
    atom_x, tma_x = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mX, in_smem, in_tiler
    )
    atom_lin, tma_lin = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mLin, in_smem, in_tiler
    )
    gX = cute.zipped_divide(tma_x, in_tiler)
    gLin = cute.zipped_divide(tma_lin, in_tiler)

    if cutlass.const_expr(IS_BWD):
        pgrad = cute.make_ptr(cutlass.BFloat16, ag, AddressSpace.gmem, assumed_align=16)
        mGrad = cute.make_tensor(pgrad, cute.make_layout((m, k), stride=(k, 1)))
        atom_grad, tma_grad = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mGrad, in_smem, in_tiler
        )
        gGrad = cute.zipped_divide(tma_grad, in_tiler)
    else:
        atom_grad, gGrad = atom_x, gX

    # b32 word views for the direct path: (word, block half, 1x32 scale
    # block, x-tile, row). The runtime strides k and k//2 would collapse the
    # sliced pointer's provable alignment to one word and narrow autovec_copy
    # to 32-bit accesses; K % 128 == 0 makes both multiples of 8 words, and
    # cute.assume encodes that so the 8-word copies stay 256-bit loads.
    kw = cute.assume(k, divby=8)  # gated row stride: 2k bf16 = k words
    khw = cute.assume(k // 2, divby=8)  # Lin base offset / grad row stride
    pxw = cute.make_ptr(cutlass.Int32, agi, AddressSpace.gmem, assumed_align=32)
    word_layout = cute.make_layout(
        (8, 2, CX // SCALE_DIM_X, k // CX, m), stride=(1, 8, 16, CX // 2, kw)
    )
    gXv = cute.make_tensor(pxw, word_layout)
    gLinv = cute.make_tensor(pxw + khw, word_layout)
    if cutlass.const_expr(IS_BWD):
        pgw = cute.make_ptr(cutlass.Int32, ag, AddressSpace.gmem, assumed_align=32)
        gGradv = cute.make_tensor(
            pgw,
            cute.make_layout(
                (8, 2, CX // SCALE_DIM_X, k // CX, m),
                stride=(1, 8, 16, CX // 2, khw),
            ),
        )
    else:
        gGradv = gXv

    if cutlass.const_expr(ROWWISE):
        pra = cute.make_ptr(
            cutlass.Float8E4M3FN, arq, AddressSpace.gmem, assumed_align=16
        )
        mRowAct = cute.make_tensor(pra, cute.make_layout((m, k), stride=(out_k, 1)))
        atom_row_act, tma_ra = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mRowAct, in_smem, in_tiler
        )
        gRowAct = cute.zipped_divide(tma_ra, in_tiler)
        if cutlass.const_expr(IS_BWD):
            prg = pra + k
            mRowGate = cute.make_tensor(
                prg, cute.make_layout((m, k), stride=(out_k, 1))
            )
            atom_row_gate, tma_rg = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mRowGate, in_smem, in_tiler
            )
            gRowGate = cute.zipped_divide(tma_rg, in_tiler)
        else:
            atom_row_gate, gRowGate = atom_row_act, gRowAct
    else:
        atom_row_act, gRowAct = atom_x, gX
        atom_row_gate, gRowGate = atom_x, gX

    if cutlass.const_expr(COLWISE):
        pca = cute.make_ptr(
            cutlass.Float8E4M3FN, acq, AddressSpace.gmem, assumed_align=16
        )
        mColAct = cute.make_tensor(pca, cute.make_layout((k, m), stride=(m, 1)))
        atom_col_act, tma_ca = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mColAct, col_smem, col_tiler
        )
        gColAct = cute.zipped_divide(tma_ca, col_tiler)
        if cutlass.const_expr(IS_BWD):
            pcg = pca + k * m
            mColGate = cute.make_tensor(pcg, cute.make_layout((k, m), stride=(m, 1)))
            atom_col_gate, tma_cg = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mColGate, col_smem, col_tiler
            )
            gColGate = cute.zipped_divide(tma_cg, col_tiler)
        else:
            atom_col_gate, gColGate = atom_col_act, gColAct
    else:
        atom_col_act, gColAct = atom_x, gX
        atom_col_gate, gColGate = atom_x, gX

    prs = cute.make_ptr(cutlass.Uint8, ars, AddressSpace.gmem, assumed_align=16)
    mRS = cute.make_tensor(prs, cute.make_layout(m * (out_k // 32)))
    pcs = cute.make_ptr(cutlass.Uint8, acs, AddressSpace.gmem, assumed_align=16)
    mCS = cute.make_tensor(pcs, cute.make_layout(out_k * (m // 32)))

    gated_act_mxfp8_kernel(
        atom_x,
        gX,
        atom_lin,
        gLin,
        atom_grad,
        gGrad,
        gXv,
        gLinv,
        gGradv,
        atom_row_act,
        gRowAct,
        atom_row_gate,
        gRowGate,
        atom_col_act,
        gColAct,
        atom_col_gate,
        gColGate,
        mRS,
        mCS,
        out_k // 128,  # rs_ncb: rowwise 128x4 scale-column blocks
        out_k // 32,  # rs_stride: rowwise compact-scale row stride
        k // 32,  # rgate_scol_off: dUp-half rowwise scale-column offset
        m // 128,  # cs_ncb: colwise 128x4 scale-column blocks
        m // 32,  # cs_stride: colwise compact-scale row stride
        k,  # cgate_col_off: dUp-half colwise output-row offset
        IS_BWD,
        ROWWISE,
        COLWISE,
        SWIZ,
        ACT_PAIR,
        DIRECT,
        CX,
        CY,
        THREADS,
    ).launch(
        grid=(k // CX, m // CY, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


@functools.cache
def _compile_kernel(
    is_bwd, rowwise, colwise, swizzled_scales, act_pair, direct, cx, cy, device_index
):
    """Compile and cache one kernel specialization. ``act_pair`` must be a
    module-level function so the cache key stays stable; ``device_index`` is
    part of the cache key only (compilation targets the active device)."""
    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        raise NotImplementedError(
            f"gated_act_mxfp8 requires CUDA SM 10.x (Blackwell); "
            f"found sm_{cap[0]}{cap[1]}"
        )
    del device_index
    from cutlass.cute.runtime import make_fake_stream

    null = cutlass.Int64(0)
    dim = cutlass.Int32(128)
    return cute.compile(
        launcher,
        null,
        null,
        null,
        null,
        null,
        null,
        dim,
        dim,
        make_fake_stream(),
        is_bwd,
        rowwise,
        colwise,
        swizzled_scales,
        act_pair,
        direct,
        cx,
        cy,
    )


def _validate_inputs(gated_input, grad_h=None):
    if not gated_input.is_cuda:
        raise ValueError("gated_input must be a CUDA tensor")
    if gated_input.dtype != torch.bfloat16:
        raise TypeError("gated_input must have dtype torch.bfloat16")
    if gated_input.ndim != 2 or not gated_input.is_contiguous():
        raise ValueError("gated_input must be contiguous with shape [M, 2K]")
    M, two_k = gated_input.shape
    if two_k % 2:
        raise ValueError("gated_input.shape[1] must be even")
    K = two_k // 2
    # Keeps CTA chunks whole and the blocked scale layout padding-free, so
    # every element of the scale tensors is written by the kernel. Zero-size
    # inputs satisfy every modulus but cannot form a launch grid or a TMA
    # descriptor, so they are rejected here instead of failing opaquely.
    if M == 0 or K == 0 or M % 128 or K % 128:
        raise ValueError("M and K must be nonzero multiples of 128")
    # Index arithmetic and scale layouts assume 32-bit offsets; the largest
    # offset any layout reaches is 2*K*M - K - 1 elements.
    if 2 * K * M - K - 1 > _INT32_MAX:
        raise ValueError(
            f"M={M}, K={K} exceeds the kernel's 32-bit indexing limit "
            f"(needs 2*K*M - K - 1 <= {_INT32_MAX})"
        )
    # The launcher passes raw device pointers promised as assumed_align=32
    # (b32 word views backing 256-bit loads) and assumed_align=16 (TMA
    # descriptors); contiguity does not imply base alignment for
    # storage-offset views. Checked after the shape/int32 gates so
    # FakeTensor probes (no data_ptr) exercise those first.
    if gated_input.data_ptr() % 32:
        raise ValueError(
            "gated_input must be 32-byte aligned (data_ptr() % 32 == 0); "
            "storage-offset views are not -- pass a fresh copy, e.g. .clone()"
        )
    if grad_h is not None:
        if (
            not grad_h.is_cuda
            or grad_h.dtype != torch.bfloat16
            or not grad_h.is_contiguous()
            or tuple(grad_h.shape) != (M, K)
        ):
            raise ValueError("grad_h must be contiguous BF16 CUDA [M, K]")
        if grad_h.device != gated_input.device:
            raise ValueError(
                f"grad_h is on {grad_h.device} but gated_input is on "
                f"{gated_input.device}; both must be on the same CUDA device"
            )
        if grad_h.data_ptr() % 32:
            raise ValueError(
                "grad_h must be 32-byte aligned (data_ptr() % 32 == 0); "
                "storage-offset views are not -- pass a fresh copy, e.g. "
                ".clone()"
            )
    return M, K


def _ptr(tensor):
    return 0 if tensor is None else tensor.data_ptr()


@torch.no_grad()
def _launch_gated_act_mxfp8(
    gated_input, grad_h, outputs, rowwise, colwise, geometry=None
):
    """Validate, compile the matching specialization, and launch into
    ``outputs`` = ``(output_rowwise, output_colwise, scales_rowwise,
    scales_colwise)``, caller-allocated. Disabled directions are zero-sized
    and not written; scales are always in the blocked (GEMM-swizzled
    tcgen05) layout. ``geometry`` overrides the per-mode default
    ``(CX, CY, direct)`` 3-tuple (tuning/testing only; staged rowwise
    requires the colwise producer — see the kernel's trace-time asserts).
    """
    if not (rowwise or colwise):
        raise ValueError("at least one of rowwise/colwise must be enabled")
    M, K = _validate_inputs(gated_input, grad_h)
    output_rowwise, output_colwise, scales_rowwise, scales_colwise = outputs
    for out, enabled, name in (
        (output_rowwise, rowwise, "output_rowwise"),
        (output_colwise, colwise, "output_colwise"),
    ):
        if enabled and out.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{name} must have dtype torch.float8_e4m3fn")

    # Compile and launch under the input's device: a caller holding cuda:0
    # current while passing a cuda:1 tensor must not launch foreign pointers.
    with torch.cuda.device(gated_input.device):
        # Wrap per call; caching CUstream handles could alias recycled streams.
        stream = CUstream(torch.cuda.current_stream(gated_input.device).cuda_stream)
        geom = geometry or _DEFAULT_GEOMETRY[(grad_h is not None, rowwise, colwise)]
        cx, cy, direct = geom
        # CX feeds bit-mask/shift thread mapping (tidx & (CX-1), >> LOG2_CX)
        # and exact-divide layouts; CY feeds the floor grid (k//CX, m//CY) and
        # N_STAGES = CY // BUFF_DIM_Y — a violating override silently skips
        # columns/rows instead of failing.
        if cx & (cx - 1) or cx % SCALE_DIM_X or K % cx:
            raise ValueError(
                f"geometry CX={cx} must be a power of two, a multiple of "
                f"{SCALE_DIM_X}, and divide K={K}"
            )
        if cy % BUFF_DIM_Y or M % cy:
            raise ValueError(
                f"geometry CY={cy} must be a multiple of {BUFF_DIM_Y} and divide M={M}"
            )
        # Row chunks ride CUDA grid dim y, which caps at 65535 on every
        # compute capability; the int32 element bound alone admits small-K
        # shapes past it.
        if M // cy > 65535:
            raise ValueError(
                f"M={M} with geometry CY={cy} needs {M // cy} row chunks, "
                "over CUDA's 65535 grid y-dimension limit"
            )
        # The staged path drains TMA-store groups only after its stage loop,
        # so output smem buffers must never be reused inside it: the stage
        # count (CY / 32) is capped at the double buffer's depth.
        if not direct and cy // BUFF_DIM_Y > BUFFS_NUM:
            raise ValueError(
                f"geometry CY={cy} needs {cy // BUFF_DIM_Y} pipeline stages; "
                f"the staged path supports at most {BUFFS_NUM} (output smem "
                "is double-buffered with no in-loop store drain)"
            )
        fn = _compile_kernel(
            grad_h is not None,
            rowwise,
            colwise,
            True,  # blocked/GEMM-swizzled scales; compact is compile-time only
            _silu_pair,
            direct,
            cx,
            cy,
            gated_input.device.index,
        )
        fn(
            _ptr(grad_h),
            gated_input.data_ptr(),
            _ptr(output_rowwise) if rowwise else 0,
            _ptr(scales_rowwise) if rowwise else 0,
            _ptr(output_colwise) if colwise else 0,
            _ptr(scales_colwise) if colwise else 0,
            M,
            K,
            stream,
        )


def _gated_act_mxfp8_outputs(
    gated_input: torch.Tensor,
    out_k: int,
    rowwise: bool,
    colwise: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the fixed four outputs.

    Torch-only, so it serves both the real op and its fake: on a meta input it
    returns meta tensors with the shapes and strides the kernel writes.
    """
    if not (rowwise or colwise):
        raise ValueError("at least one of rowwise/colwise must be enabled")
    m = gated_input.shape[0]
    empty_qdata = gated_input.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_scales = gated_input.new_empty(0, dtype=torch.float8_e8m0fnu)

    if rowwise:
        output_rowwise = torch.empty_strided(
            (m, out_k),
            (out_k, 1),
            device=gated_input.device,
            dtype=torch.float8_e4m3fn,
        )
        scales_rowwise = gated_input.new_empty(
            (ceil_div(m, 128) * 128, ceil_div(out_k // 32, 4) * 4),
            dtype=torch.float8_e8m0fnu,
        )
    else:
        output_rowwise, scales_rowwise = empty_qdata, empty_scales

    if colwise:
        output_colwise = torch.empty_strided(
            (m, out_k),
            (1, m),
            device=gated_input.device,
            dtype=torch.float8_e4m3fn,
        )
        # Flat 1D, matching mxfp8_quantize_2d_32x1_cutedsl.
        scales_colwise = gated_input.new_empty(
            ((ceil_div(out_k, 128) * 128) * (ceil_div(m // 32, 4) * 4),),
            dtype=torch.float8_e8m0fnu,
        )
    else:
        output_colwise, scales_colwise = empty_qdata, empty_scales

    return output_rowwise, output_colwise, scales_rowwise, scales_colwise


@torch.library.custom_op("torchao::gated_act_mxfp8_cutedsl_forward", mutates_args=())
def _gated_act_mxfp8_cutedsl_forward(
    gated_input: torch.Tensor,
    rowwise: bool = True,
    colwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = _gated_act_mxfp8_outputs(
        gated_input, gated_input.shape[1] // 2, rowwise, colwise
    )
    _launch_gated_act_mxfp8(gated_input, None, outputs, rowwise, colwise)
    return outputs


@torch.library.custom_op("torchao::gated_act_mxfp8_cutedsl_backward", mutates_args=())
def _gated_act_mxfp8_cutedsl_backward(
    grad_h: torch.Tensor,
    gated_input: torch.Tensor,
    rowwise: bool = True,
    colwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = _gated_act_mxfp8_outputs(
        gated_input, gated_input.shape[1], rowwise, colwise
    )
    _launch_gated_act_mxfp8(gated_input, grad_h, outputs, rowwise, colwise)
    return outputs


@_gated_act_mxfp8_cutedsl_forward.register_fake
def _fake_gated_act_mxfp8_cutedsl_forward(
    gated_input: torch.Tensor,
    rowwise: bool = True,
    colwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert gated_input.ndim == 2, "gated_input must be 2D"
    return _gated_act_mxfp8_outputs(
        gated_input, gated_input.shape[1] // 2, rowwise, colwise
    )


@_gated_act_mxfp8_cutedsl_backward.register_fake
def _fake_gated_act_mxfp8_cutedsl_backward(
    grad_h: torch.Tensor,
    gated_input: torch.Tensor,
    rowwise: bool = True,
    colwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert grad_h.ndim == 2, "grad_h must be 2D"
    assert gated_input.ndim == 2, "gated_input must be 2D"
    return _gated_act_mxfp8_outputs(gated_input, gated_input.shape[1], rowwise, colwise)


def gated_act_mxfp8_cutedsl_forward(
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fuse the gated activation (SwiGLU) forward and its RCEIL MXFP8 cast into
    one pass on SM100: ``h = silu(gate) * up`` is quantized without ever
    being written to global memory.

    Args:
        gated_input: BF16 tensor of shape (M, 2K), ``gate`` in the first K
            columns and ``up`` in the last K; M and K multiples of 128.
        rowwise: emit 1x32-scaled, row-major output.
        colwise: emit 32x1-scaled, column-major (stride ``(1, M)``) output.

    Returns:
        ``(output_rowwise, output_colwise, scales_rowwise, scales_colwise)``,
        width K, E8M0 scales in the same blocked tcgen05 layouts as the
        standalone quantizers. Disabled directions return zero-sized tensors,
        so the output arity never varies. Special values follow the standalone
        quantizers' contract: a NaN/Inf block amax yields scale byte 0xFF and
        all-NaN output codes; zero/tiny amaxes clamp to scale byte 0x00.
    """
    # Read the shared availability flag at call time (tests monkeypatch it).
    from torchao.prototype.moe_training.kernels.mxfp8 import quant as _quant

    if not _quant._mxfp8_cutedsl_kernels_available:
        missing_packages = _missing_cutedsl_runtime_packages()
        if missing_packages:
            missing = ", ".join(missing_packages)
            raise NotImplementedError(
                "gated_act_mxfp8_cutedsl_forward requires additional Python "
                f"runtime package(s): {missing}. Please install "
                "`nvidia-cutlass-dsl` and `apache-tvm-ffi`."
            )
        raise NotImplementedError(
            "gated_act_mxfp8_cutedsl_forward requires CUDA, SM 10.x, and CUDA 12.8+."
        )
    return _gated_act_mxfp8_cutedsl_forward(gated_input, rowwise, colwise)


def gated_act_mxfp8_cutedsl_backward(
    grad_h: torch.Tensor,
    gated_input: torch.Tensor,
    *,
    rowwise: bool = True,
    colwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fuse the gated activation (SwiGLU) backward and its RCEIL MXFP8 cast into
    one pass on SM100: quantizes the concatenated ``[dGate | dUp]`` tensor a
    fused w13 weight expects for the wgrad GEMM.

    Args:
        grad_h: BF16 gradient of shape (M, K).
        gated_input: the forward input, shape (M, 2K), as in
            :func:`gated_act_mxfp8_cutedsl_forward`.
        rowwise: emit 1x32-scaled, row-major output.
        colwise: emit 32x1-scaled, column-major (stride ``(1, M)``) output.

    Returns:
        Four tensors of width 2K, same order and layouts as
        :func:`gated_act_mxfp8_cutedsl_forward`.
    """
    from torchao.prototype.moe_training.kernels.mxfp8 import quant as _quant

    if not _quant._mxfp8_cutedsl_kernels_available:
        missing_packages = _missing_cutedsl_runtime_packages()
        if missing_packages:
            missing = ", ".join(missing_packages)
            raise NotImplementedError(
                "gated_act_mxfp8_cutedsl_backward requires additional Python "
                f"runtime package(s): {missing}. Please install "
                "`nvidia-cutlass-dsl` and `apache-tvm-ffi`."
            )
        raise NotImplementedError(
            "gated_act_mxfp8_cutedsl_backward requires CUDA, SM 10.x, and CUDA 12.8+."
        )
    return _gated_act_mxfp8_cutedsl_backward(grad_h, gated_input, rowwise, colwise)
