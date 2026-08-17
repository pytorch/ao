# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side epilogue primitives shared by the MXFP8 grouped-MLP kernels.

These are the pieces the three grouped kernels (FC1 GEMM + SwiGLU + dual quant,
FC2 dgrad + dSwiGLU + dual quant, and grouped wgrad) have in common: blocked
scale addressing, the RCEIL E8M0 conversion, NaN-propagating packed amax, and
the gated-activation policy. They are lifted from the activation-only gated
kernel, whose numerics are already validated bitwise against the standalone
quantizers.

NUMERICAL PRECONDITION (load-bearing, read before reusing anything here):

    :func:`float_to_e8m0` is integer bit math whose rounding constant assumes
    its input is a BF16 value widened to FP32. It is exact for a BF16 amax and
    is NOT exact for a general FP32 amax -- for example FP32 0x40600001 gives
    120 where the canonical conversion gives 121.

A fused GEMM epilogue holds FP32 accumulators, so it must round to BF16 *before*
taking the amax, which is what the kernel contract already requires at every
activation boundary. Do not "optimize" that rounding away while keeping this
conversion; if an FP32-amax path is ever wanted, use the canonical
``cute_utils.compute_scale_rceil`` (a real ``cvt.rp`` instruction) instead.

Scale semantics follow torchao #4725: RCEIL with saturation disabled, a
non-finite amax yielding scale byte 255 and an all-NaN block, a zero block
yielding scale byte 0 (which dequantizes to 2^-127, not 1.0), and
``inv_scale = e8m0(254 - byte)``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "SCALE_BLOCK",
    "SCALE_TILE_ROWS",
    "SCALE_TILE_COLS",
    "SCALE_TILE_BYTES",
    "blocked_scale_idx",
    "float_to_e8m0",
    "e8m0_reciprocal_bf16",
    "max_nan_bf16x2",
    "abs_max_nan_bf16x2",
    "fold_amax",
    "pack_bf16x2",
    "bf16x2_lo_to_f32",
    "bf16x2_hi_to_f32",
    "mul_cvt_2x",
    "prmt_even",
    "prmt_odd",
    "sigmoidf",
    "silu_pair",
    "validate_group_offsets_device",
]

# MXFP8 scaling block: 32 values share one E8M0 scale.
SCALE_BLOCK = 32
# tcgen05 blocked scale tile geometry: 128 scale rows x 4 scale columns = 512 B.
SCALE_TILE_ROWS = 128
SCALE_TILE_COLS = 4
SCALE_TILE_BYTES = SCALE_TILE_ROWS * SCALE_TILE_COLS
# Every TMA-accessed shared-memory buffer must be 128-byte aligned.
TMA_SHMEM_ALIGNMENT = 128


def blocked_scale_idx(row, scale_col, num_scale_col_blocks):
    """Flat index of one scale byte in the tcgen05 blocked (128x4) layout.

    The logical ``[rows, cols/32]`` scale matrix is stored as 512-byte tiles of
    128 rows x 4 scale columns (cuBLAS "128x4 block scaling factors layout"),
    tiles ordered ``row_block * num_scale_col_blocks + col_block``.
    ``num_scale_col_blocks`` is ``ceil_div(num_scale_cols, 4)``.

    Coordinates are ABSOLUTE, never per-group. That is legal for both
    orientations this family emits:

    * rowwise, where the ragged axis is the scale row -- because every expert
      row count is a multiple of 128, no 128-row tile straddles a group
      boundary, so per-group blocking and whole-matrix blocking are the same
      bytes;
    * columnwise, where the ragged axis is the scale column -- because this
      family defines those buffers as whole-matrix ``to_blocked`` (see the
      kernel contract 4.2.1). Note this deliberately differs from torchao's
      ``triton_mx_block_rearrange_2d_K_groups``, which pads per group.

    For columnwise scales pass transposed coordinates (feature index as ``row``,
    row-block index as ``scale_col``).
    """
    return (
        ((row >> 7) * num_scale_col_blocks + (scale_col >> 2)) * SCALE_TILE_BYTES
        + (row & 31) * 16
        + ((row >> 5) & 3) * 4
        + (scale_col & 3)
    )


@dsl_user_op
def _bitcast_i32_to_f32(val: Int32, *, loc=None, ip=None) -> Float32:
    """Bitcast int32 to float32 without changing the bit pattern."""
    return Float32(
        mlir_arith.bitcast(T.f32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


# bf16 == top 16 bits of f32, so widening is a free bit-shift.
@dsl_user_op
def bf16x2_lo_to_f32(bits, *, loc=None, ip=None) -> Float32:
    return _bitcast_i32_to_f32(
        (Int32(bits) & Int32(0xFFFF)) << Int32(16), loc=loc, ip=ip
    )


@dsl_user_op
def bf16x2_hi_to_f32(bits, *, loc=None, ip=None) -> Float32:
    # `(x >> 16) << 16` == `x & 0xFFFF0000` without a signed literal; the left
    # shift zeroes the arithmetic shift's smeared sign bits.
    return _bitcast_i32_to_f32((Int32(bits) >> Int32(16)) << Int32(16), loc=loc, ip=ip)


# Each packed-bf16x2 op below is written out explicitly rather than produced by a
# factory: the DSL keys its compile cache on function identity and name, so ops
# sharing a `__name__` are a cache hazard.
#
# The `.NaN` max variants match the standalone quantizers' amax reduction, which
# propagates NaN; a plain max would return the non-NaN operand and silently
# rescue a block that must be invalidated.
@dsl_user_op
def max_nan_bf16x2(a, b, *, loc=None, ip=None):
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
def abs_max_nan_bf16x2(a, b, *, loc=None, ip=None):
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


@cute.jit
def fold_amax(am: Int32) -> Int32:
    """Reduce a packed bf16x2 amax word to bf16 amax bits in [15:0].

    The input's per-lane sign bits are junk (see :func:`abs_max_nan_bf16x2`);
    mask them, then fold the two lanes with the NaN-propagating max.
    """
    am = am & Int32(0x7FFF7FFF)
    am = max_nan_bf16x2(am, am >> 16)
    return am & Int32(0xFFFF)


@dsl_user_op
def prmt_even(a, b, *, loc=None, ip=None):
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
def prmt_odd(a, b, *, loc=None, ip=None):
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
def mul_cvt_2x(w0, w1, s, *, loc=None, ip=None):
    """Scale two bf16x2 words by bf16x2 ``s`` and pack four E4M3 bytes into one
    b32 store word.

    ``cvt.rn.satfinite.e4m3x2.bf16x2`` is missing on some Blackwells (GB300's
    sm_103a), so keep the bf16 multiply for identical rounding, widen exactly to
    f32, and use the portable f32-source cvt. Saturation to +/-448 comes from
    this conversion; do not add an explicit clamp.
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
def pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(hi, lo) f32 -> packed bf16x2 word, round-to-nearest-even.

    This is the mandatory "truncate to BF16 before any amax" step for a GEMM
    epilogue holding FP32 accumulators; see the module docstring.
    """
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
def float_to_e8m0(u: Int32) -> Int32:
    """Biased E8M0 RCEIL scale byte for a non-negative BF16 amax, as f32 bits.

    Pass the RAW amax. The division by 448 is folded into the constants -- the
    ``- 8`` shifts the exponent by 256 and the ``+ 0x1F0000`` mantissa offset
    supplies the remaining 1.75 factor (448 = 256 * 1.75) together with the
    RCEIL round-up carry. Pre-dividing by 448 before calling this double-counts
    the division and yields a scale 8-9 codes too low.

    Finite: the RCEIL mantissa-carry path, matching what ``cvt.rp.ue8m0x2.f32``
    (no ``.satfinite``) emits for ``amax / 448``. Non-finite: a NaN or Inf amax
    invalidates the block with scale byte 255; without the branch, Inf would
    land on 247 and NaN could carry into the sign bit.

    Exact only for a BF16-valued amax -- see the module docstring.
    """
    e = cutlass.max(((u + Int32(0x1F0000)) >> 23) - Int32(8), Int32(0))
    if (u & Int32(0x7F800000)) == Int32(0x7F800000):
        e = Int32(255)
    return e


@cute.jit
def e8m0_reciprocal_bf16(e: Int32) -> Int32:
    """Inverse scale as bf16 bits, matching ``ue8m0(254 - scale_byte)``.

    The quantization multiply in :func:`mul_cvt_2x` is bf16x2, not f32, so the
    reciprocal is synthesized directly in bf16: 2^(127 - e) over the normal
    range (a byte-0 block from a zero or tiny amax descales by 2^127), and NaN
    for an invalidated block (byte 255) so every element quantizes to the E4M3
    NaN code.

    This is NOT a general ``ue8m0`` helper: byte 254 yields +0.0 rather than
    2^-127, and bytes above 254 go negative. That is safe here only because a
    scale byte produced by this family can never exceed 247 (amax is divided by
    448 first). Do not reuse it to dequantize externally produced scale bytes.
    """
    b = (Int32(254) - e) << 7
    if e == Int32(255):
        b = Int32(0x7FC0)
    return b


@dsl_user_op
def sigmoidf(x, *, loc=None, ip=None):
    """Sigmoid as ``__frcp_rn(1.0f + __expf(-x))``, emitted as raw PTX,
    instruction for instruction::

        mul.f32        t, x, 0fBFB8AA3B    // -x * log2(e)
        ex2.approx.f32 t, t
        add.f32        t, t, 0f3F800000
        rcp.rn.f32     s, t                // correctly rounded

    No higher-level formulation reproduces ``ex2.approx``, and the ``rcp.rn`` vs
    ``div.full`` choice shows at a few output codes per million.
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
def silu_pair(x0, x1, lin0, lin1, g0, g1, IS_BWD: cutlass.Constexpr):
    """SwiGLU / dSwiGLU policy for a pair of elements.

    ``x`` is the gate (activation input), ``lin`` the up (linear multiplier),
    ``g`` the incoming gradient (ignored unless IS_BWD)::

        s = sigmoid(x); act = x * s
        forward:  out_act = act * lin
        backward: dact = x*s*(1-s) + s      (contracted into one FMA)
                  out_act = (dact * g) * lin   -> dGate
                  out_gate = act * g           -> dUp

    Returns f32 ``(out_act0, out_act1, out_gate0, out_gate1)``, the gate pair
    zero in forward. Callers MUST round to BF16 immediately, before any amax,
    caching, or quantization -- both because the kernel contract defines
    correctness at that boundary and because :func:`float_to_e8m0` requires it.

    Pass this as a Constexpr kernel parameter. It must stay a module-level
    function: the DSL keys its compile cache on the function object, so a lambda
    or closure built per call misses the cache and recompiles every launch.
    """
    one = Float32(1.0)
    s0 = sigmoidf(x0)
    s1 = sigmoidf(x1)
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
        return oa0, oa1, Float32(0.0), Float32(0.0)


@cute.jit
def validate_group_offsets_device(offs: cute.Tensor, allocated_rows: Int32):
    """Device-side precondition check on the exclusive-end group offsets.

    Checks what the host cannot see without synchronizing: every per-expert row
    count is a nonnegative multiple of 128, the offsets are nondecreasing, and
    the active row count does not exceed the allocation.

    DO NOT RELY ON THIS AS A GUARDRAIL. ``cute.testing.assert_`` is compiled out
    unless ``CUTE_DSL_ENABLE_ASSERTIONS=1``, so in a default build this function
    is a no-op -- measured directly: an always-false assertion in four
    placements (plain, warp-0, elected, block-0-elected) lets the kernel run to
    completion and write its output. torchao's own ``validate_group_sizes`` has
    the same property. Even with assertions enabled the failure mode is the
    message followed by ``unspecified launch failure``, i.e. a dead CUDA
    context, not a catchable error.

    The real enforcement of the 128-multiple precondition is the host-side
    metadata validation in ``grouped_mlp_validation`` at the custom-op boundary,
    which raises ``ValueError`` before any launch. This function is a debugging
    aid for assertion-enabled builds. It matters that the distinction is
    explicit: with malformed offsets the ragged-K path (Kernel C) silently
    returns a WRONG weight gradient rather than crashing, so "it did not fault"
    is not evidence that the offsets were valid.
    """
    num_groups = offs.shape[0]
    prev = Int32(0)
    for i in range(num_groups):
        end = offs[i]
        size = end - prev
        cute.testing.assert_(size >= 0, "Group offsets must be nondecreasing")
        cute.testing.assert_(size % 128 == 0, "Group sizes must be multiples of 128")
        prev = end
    cute.testing.assert_(
        prev <= allocated_rows,
        "Active row count offsets[-1] must not exceed the allocated row count",
    )
