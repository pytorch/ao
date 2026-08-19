"""CuteDSL RHT + NVFP4 E2M1 columnwise/rowwise quantization kernels for SM100.

Private impl for the torchao:: ops. Imported lazily by the op wrappers so the top-level
``import cutlass`` only runs when a cutedsl op is actually called.

Two kernels, both built on the same single A load shared by two consumers (the MMA warp
does the columnwise RHT, a row warp group reads the same tile):
  - _Tcgen05RowColFused: quantizes col=RHT(A.t()) and row=A to NVFP4.
  - _Tcgen05RhtAmax: reduces col=max|RHT(A.t())|, row=max|A|.
"""

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import blackwell_helpers as sm100_utils
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

# DEFAULT_SIGN_VECTOR is re-exported: several modules import it from here, and
# hadamard_utils is the runtime-free home so the reference and tests can reach it too.
from .hadamard_utils import DEFAULT_SIGN_VECTOR, get_rht_matrix  # noqa: F401

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
FP32_MAX = torch.finfo(torch.float32).max

HADAMARD_DIM = 16


# ---------------------------------------------------------------------------
# CuteDSL inline PTX ops
# ---------------------------------------------------------------------------


@dsl_user_op
def _mul_cvt_rn_e2m1x8_f32(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    v4: cutlass.Float32,
    v5: cutlass.Float32,
    v6: cutlass.Float32,
    v7: cutlass.Float32,
    scale: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Scale eight BF16-origin values with packed FP32 multiplies and pack to FP4."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v0.ir_value(loc=loc, ip=ip),
                v1.ir_value(loc=loc, ip=ip),
                v2.ir_value(loc=loc, ip=ip),
                v3.ir_value(loc=loc, ip=ip),
                v4.ir_value(loc=loc, ip=ip),
                v5.ir_value(loc=loc, ip=ip),
                v6.ir_value(loc=loc, ip=ip),
                v7.ir_value(loc=loc, ip=ip),
                scale.ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n"
                ".reg .b64 s2, p01, p23, p45, p67;\n"
                ".reg .f32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                ".reg .b8 b0, b1, b2, b3;\n"
                "mov.b64 s2, {$9, $9};\n"
                "mov.b64 p01, {$1, $2};\n"
                "mov.b64 p23, {$3, $4};\n"
                "mov.b64 p45, {$5, $6};\n"
                "mov.b64 p67, {$7, $8};\n"
                "mul.f32x2 p01, p01, s2;\n"
                "mul.f32x2 p23, p23, s2;\n"
                "mul.f32x2 p45, p45, s2;\n"
                "mul.f32x2 p67, p67, s2;\n"
                "mov.b64 {a1, a0}, p01;\n"
                "mov.b64 {a3, a2}, p23;\n"
                "mov.b64 {a5, a4}, p45;\n"
                "mov.b64 {a7, a6}, p67;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b0, a0, a1;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b1, a2, a3;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b2, a4, a5;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b3, a6, a7;\n"
                "mov.b32 $0, {b0, b1, b2, b3};\n"
                "}"
            ),
            "=r,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_cvt_rn_e2m1x8_acc_f32(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    v4: cutlass.Float32,
    v5: cutlass.Float32,
    v6: cutlass.Float32,
    v7: cutlass.Float32,
    scale: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """``_mul_cvt_rn_e2m1x8_f32`` for raw tcgen05 RHT accumulators.

    Same packed multiply/convert, with the exact-mode bfloat16 round-through folded in: each
    pair is rounded with one ``cvt.rn.bf16x2.f32`` and re-widened by shift/mask, which is what
    the scalar path did before multiplying. The explicit clamp to +-FP4_E2M1_MAX is dropped
    because ``cvt.rn.satfinite`` already saturates there.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v0.ir_value(loc=loc, ip=ip),
                v1.ir_value(loc=loc, ip=ip),
                v2.ir_value(loc=loc, ip=ip),
                v3.ir_value(loc=loc, ip=ip),
                v4.ir_value(loc=loc, ip=ip),
                v5.ir_value(loc=loc, ip=ip),
                v6.ir_value(loc=loc, ip=ip),
                v7.ir_value(loc=loc, ip=ip),
                scale.ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n"
                ".reg .b64 s2, p01, p23, p45, p67;\n"
                ".reg .b32 t01, t23, t45, t67, e0, e1, e2, e3, e4, e5, e6, e7;\n"
                ".reg .f32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                ".reg .b8 b0, b1, b2, b3;\n"
                "mov.b64 s2, {$9, $9};\n"
                "cvt.rn.bf16x2.f32 t01, $2, $1;\n"
                "cvt.rn.bf16x2.f32 t23, $4, $3;\n"
                "cvt.rn.bf16x2.f32 t45, $6, $5;\n"
                "cvt.rn.bf16x2.f32 t67, $8, $7;\n"
                "shl.b32 e0, t01, 16;\n"
                "and.b32 e1, t01, 0xffff0000;\n"
                "shl.b32 e2, t23, 16;\n"
                "and.b32 e3, t23, 0xffff0000;\n"
                "shl.b32 e4, t45, 16;\n"
                "and.b32 e5, t45, 0xffff0000;\n"
                "shl.b32 e6, t67, 16;\n"
                "and.b32 e7, t67, 0xffff0000;\n"
                "mov.b64 p01, {e0, e1};\n"
                "mov.b64 p23, {e2, e3};\n"
                "mov.b64 p45, {e4, e5};\n"
                "mov.b64 p67, {e6, e7};\n"
                "mul.f32x2 p01, p01, s2;\n"
                "mul.f32x2 p23, p23, s2;\n"
                "mul.f32x2 p45, p45, s2;\n"
                "mul.f32x2 p67, p67, s2;\n"
                "mov.b64 {a1, a0}, p01;\n"
                "mov.b64 {a3, a2}, p23;\n"
                "mov.b64 {a5, a4}, p45;\n"
                "mov.b64 {a7, a6}, p67;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b0, a0, a1;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b1, a2, a3;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b2, a4, a5;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b3, a6, a7;\n"
                "mov.b32 $0, {b0, b1, b2, b3};\n"
                "}"
            ),
            "=r,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_cvt_rs_e2m1x8_f32(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    v4: cutlass.Float32,
    v5: cutlass.Float32,
    v6: cutlass.Float32,
    v7: cutlass.Float32,
    scale: cutlass.Float32,
    rb0: cutlass.Uint32,
    rb1: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Stochastic-rounding analog of ``_mul_cvt_rn_e2m1x8_f32``.

    Same packed FP32 multiplies, but the four ``cvt.rn.satfinite.e2m1x2.f32`` collapse into two
    ``cvt.rs.satfinite.e2m1x4.f32``, each consuming one 32-bit random word: ``rb0`` covers
    ``v0..v3``, ``rb1`` covers ``v4..v7``. ``cvt.rs`` takes its four sources most-significant
    nibble first, so ``{a3, a2, a1, a0}`` lays ``v0..v3`` down in ascending nibble order.

    The explicit clamp to +-FP4_E2M1_MAX the scalar path applied is dropped, as in
    ``_mul_cvt_rn_e2m1x8_acc_f32``: ``.satfinite`` already saturates there.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v0.ir_value(loc=loc, ip=ip),
                v1.ir_value(loc=loc, ip=ip),
                v2.ir_value(loc=loc, ip=ip),
                v3.ir_value(loc=loc, ip=ip),
                v4.ir_value(loc=loc, ip=ip),
                v5.ir_value(loc=loc, ip=ip),
                v6.ir_value(loc=loc, ip=ip),
                v7.ir_value(loc=loc, ip=ip),
                scale.ir_value(loc=loc, ip=ip),
                rb0.ir_value(loc=loc, ip=ip),
                rb1.ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n"
                ".reg .b64 s2, p01, p23, p45, p67;\n"
                ".reg .f32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                ".reg .b16 h0, h1;\n"
                "mov.b64 s2, {$9, $9};\n"
                "mov.b64 p01, {$1, $2};\n"
                "mov.b64 p23, {$3, $4};\n"
                "mov.b64 p45, {$5, $6};\n"
                "mov.b64 p67, {$7, $8};\n"
                "mul.f32x2 p01, p01, s2;\n"
                "mul.f32x2 p23, p23, s2;\n"
                "mul.f32x2 p45, p45, s2;\n"
                "mul.f32x2 p67, p67, s2;\n"
                "mov.b64 {a0, a1}, p01;\n"
                "mov.b64 {a2, a3}, p23;\n"
                "mov.b64 {a4, a5}, p45;\n"
                "mov.b64 {a6, a7}, p67;\n"
                "cvt.rs.satfinite.e2m1x4.f32 h0, {a3, a2, a1, a0}, $10;\n"
                "cvt.rs.satfinite.e2m1x4.f32 h1, {a7, a6, a5, a4}, $11;\n"
                "mov.b32 $0, {h0, h1};\n"
                "}"
            ),
            "=r,f,f,f,f,f,f,f,f,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_cvt_rs_e2m1x8_acc_f32(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    v4: cutlass.Float32,
    v5: cutlass.Float32,
    v6: cutlass.Float32,
    v7: cutlass.Float32,
    scale: cutlass.Float32,
    rb0: cutlass.Uint32,
    rb1: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """``_mul_cvt_rs_e2m1x8_f32`` for raw tcgen05 RHT accumulators.

    Carries the same exact-mode bfloat16 round-through as ``_mul_cvt_rn_e2m1x8_acc_f32``.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v0.ir_value(loc=loc, ip=ip),
                v1.ir_value(loc=loc, ip=ip),
                v2.ir_value(loc=loc, ip=ip),
                v3.ir_value(loc=loc, ip=ip),
                v4.ir_value(loc=loc, ip=ip),
                v5.ir_value(loc=loc, ip=ip),
                v6.ir_value(loc=loc, ip=ip),
                v7.ir_value(loc=loc, ip=ip),
                scale.ir_value(loc=loc, ip=ip),
                rb0.ir_value(loc=loc, ip=ip),
                rb1.ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n"
                ".reg .b64 s2, p01, p23, p45, p67;\n"
                ".reg .b32 t01, t23, t45, t67, e0, e1, e2, e3, e4, e5, e6, e7;\n"
                ".reg .f32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
                ".reg .b16 h0, h1;\n"
                "mov.b64 s2, {$9, $9};\n"
                "cvt.rn.bf16x2.f32 t01, $2, $1;\n"
                "cvt.rn.bf16x2.f32 t23, $4, $3;\n"
                "cvt.rn.bf16x2.f32 t45, $6, $5;\n"
                "cvt.rn.bf16x2.f32 t67, $8, $7;\n"
                "shl.b32 e0, t01, 16;\n"
                "and.b32 e1, t01, 0xffff0000;\n"
                "shl.b32 e2, t23, 16;\n"
                "and.b32 e3, t23, 0xffff0000;\n"
                "shl.b32 e4, t45, 16;\n"
                "and.b32 e5, t45, 0xffff0000;\n"
                "shl.b32 e6, t67, 16;\n"
                "and.b32 e7, t67, 0xffff0000;\n"
                "mov.b64 p01, {e0, e1};\n"
                "mov.b64 p23, {e2, e3};\n"
                "mov.b64 p45, {e4, e5};\n"
                "mov.b64 p67, {e6, e7};\n"
                "mul.f32x2 p01, p01, s2;\n"
                "mul.f32x2 p23, p23, s2;\n"
                "mul.f32x2 p45, p45, s2;\n"
                "mul.f32x2 p67, p67, s2;\n"
                "mov.b64 {a0, a1}, p01;\n"
                "mov.b64 {a2, a3}, p23;\n"
                "mov.b64 {a4, a5}, p45;\n"
                "mov.b64 {a6, a7}, p67;\n"
                "cvt.rs.satfinite.e2m1x4.f32 h0, {a3, a2, a1, a0}, $10;\n"
                "cvt.rs.satfinite.e2m1x4.f32 h1, {a7, a6, a5, a4}, $11;\n"
                "mov.b32 $0, {h0, h1};\n"
                "}"
            ),
            "=r,f,f,f,f,f,f,f,f,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _div_rn_f32(
    a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    """Correctly rounded FP32 division, matching TransformerEngine's default path."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bf16lo_to_f32(w: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Float32:
    """Widen the low bf16 of a packed pair to f32, in one instruction.

    bfloat16 is the truncation of float32 -- same 8-bit exponent field and bias, mantissa
    zero-filled -- so widening is exactly a shift: no rounding, and no special case for
    subnormals, infinities or NaN payloads. Reading the pair as one u32 and shifting is what
    ``cutlass::bfloat16_t::operator float()`` does, so it is what TransformerEngine's
    epilogues get for free. Going through the DSL's ``BFloat16`` element type instead costs
    two instructions per value: ptxas materializes the 16-bit extract as a ``PRMT`` and then
    widens. Callers reach the pairs with ``cute.recast_tensor(rBlk, cutlass.Uint32)``.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [w.ir_value(loc=loc, ip=ip)],
            ("{\n.reg .b32 t;\nshl.b32 t, $1, 16;\nmov.b32 $0, t;\n}"),
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bf16hi_to_f32(w: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Float32:
    """Widen the high bf16 of a packed pair to f32. See ``_bf16lo_to_f32``.

    The high half's shift-right-then-shift-left collapses into a single mask.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [w.ir_value(loc=loc, ip=ip)],
            ("{\n.reg .b32 t;\nand.b32 t, $1, 0xffff0000;\nmov.b32 $0, t;\n}"),
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mulhi_u32(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None):
    """High 32 bits of a 32x32 unsigned multiply (triton's math.umulhi)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# Philox-4x32-10 (PHILOX_KEY_A/B, PHILOX_ROUND_A/B, 10 rounds).
# The generator is the same one triton.language.random uses, but the counter is not:
# every kernel here draws through philox4_all, one counter per 16-element block with all
# four output words consumed, rather than triton's per-packed-byte stride. So the FP4
# codes agree with triton under RTNE and are a different, equally valid stream under
# stochastic rounding.
PHILOX_ROUNDS = 10
_PHILOX_KEY_A, _PHILOX_KEY_B = 0x9E3779B9, 0xBB67AE85
_PHILOX_ROUND_A, _PHILOX_ROUND_B = 0xD2511F53, 0xCD9E8D57

# 16-element quantization blocks in one 128x128 tile, i.e. one Philox draw each, and the
# stride between tile ids in the stochastic-rounding counter.
TILE_BLOCKS = (128 * 128) // 16


def philox_prep(seed_lo, seed_hi, offset_base):
    """Hoist every launch-uniform part of Philox out of the epilogues.

    ``tl.randint`` enters with ``c2 = c3 = 0``, so round 1 leaves ``c1 = 0`` and makes
    ``c2``/``c3`` functions of ``c0`` alone -- and ``c0`` is the low half of the offset,
    i.e. the launch-uniform ``offset_base``. Only ``c0' = c1 ^ k0`` carries the
    per-element counter, and round 2's ``c0``/``c1`` are still counter-independent. The
    key schedule depends only on the round index, so all ten steps precompute too.

    Returns the opaque state ``philox4_all`` consumes. Building it once per kernel keeps
    the per-element cost at eight full rounds plus a two-instruction round-2 tail.
    """
    sched = [
        (
            seed_lo + cutlass.Uint32((r * _PHILOX_KEY_A) & 0xFFFFFFFF),
            seed_hi + cutlass.Uint32((r * _PHILOX_KEY_B) & 0xFFFFFFFF),
        )
        for r in range(PHILOX_ROUNDS)
    ]
    A, B = cutlass.Uint32(_PHILOX_ROUND_A), cutlass.Uint32(_PHILOX_ROUND_B)
    c2_r1 = _mulhi_u32(A, offset_base) ^ sched[0][1]
    c3_r1 = A * offset_base
    c0_r2 = _mulhi_u32(B, c2_r1) ^ sched[1][0]
    c1_r2 = B * c2_r1
    return sched, c0_r2, c1_r2, c3_r1


def philox4_all(state, chunk_counter):
    """The four random words a 16-element chunk needs, from a single Philox draw.

    These kernels once reproduced triton's counter stride, drawing one word per packed
    byte at counters ``p0, p0+1, p0+4, p0+5`` because that is what triton's ``cvt.rs`` asm
    consumes. That cost four full round schedules per chunk and discarded three of every
    four output words -- 124 multiplies for 128 bits that one draw produces. Consuming a
    single draw whole costs 34, and yields the same 128 bits.

    The counter must be derived from tile coordinates rather than from a running
    per-thread value: these kernels are persistent, and the grouped one schedules through
    CLC, so visit order is not fixed and a running counter would make the output depend on
    scheduling rather than on position.
    """
    sched, c0_r2, c1_r2, c3_r1 = state
    A, B = cutlass.Uint32(_PHILOX_ROUND_A), cutlass.Uint32(_PHILOX_ROUND_B)
    c0_r1 = chunk_counter ^ sched[0][0]
    c0, c1 = c0_r2, c1_r2
    c2 = _mulhi_u32(A, c0_r1) ^ c3_r1 ^ sched[1][1]
    c3 = A * c0_r1
    for r in range(2, PHILOX_ROUNDS):
        _c0, _c2 = c0, c2
        c0 = _mulhi_u32(B, _c2) ^ c1 ^ sched[r][0]
        c2 = _mulhi_u32(A, _c0) ^ c3 ^ sched[r][1]
        c1 = B * _c2
        c3 = A * _c0
    return c0, c1, c2, c3


@dsl_user_op
def _min_f32(
    a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _max_f32(
    a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    # max.NaN.f32 returns NaN if either operand is NaN, so the amax reductions propagate NaN
    # (matching triton_rht_amax). The non-NaN max.f32 variant would silently drop it.
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "max.NaN.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _atom_max_f32_nonneg(
    addr: cutlass.Pointer, val: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    """Atomic max on global memory for non-negative float32.

    For non-negative floats, bit patterns are ordered the same as unsigned integers,
    so we can use atom.global.max.u32 on the reinterpreted bits.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [addr.llvm_ptr, val.ir_value(loc=loc, ip=ip)],
            (
                "{\n"
                ".reg .b32 v_bits, old_bits;\n"
                "mov.b32 v_bits, $2;\n"
                "atom.global.max.u32 old_bits, [$1], v_bits;\n"
                "mov.b32 $0, old_bits;\n"
                "}"
            ),
            "=f,l,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _abs_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Absolute value of float32."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ---------------------------------------------------------------------------
# Hadamard matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compilation and caching
# ---------------------------------------------------------------------------


# Device-scoped cache to avoid redundant per-call work in the hot path
@functools.lru_cache(maxsize=4)
def _get_num_sms(device_idx: int) -> int:
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


# ---------------------------------------------------------------------------
# Fused row+col tcgen05 RHT+NVFP4 kernel (single A load, dual-consumer sA)
# ---------------------------------------------------------------------------
M_TILE = 128
N_TILE = 16
K = 16
MMA_TILER = (M_TILE, N_TILE, K)  # instruction atom stays 128x16x16

NUM_AB_STAGE = 2
NUM_ACC_STAGE = 1

# --- warp specialization, supertile-independent part (RHT mode): col is a cheap TMEM epilogue ---
N_COL_WARPS = 4
COL_WARP_END = N_COL_WARPS  # warps 0..3 = col
ROW_WARP_BEGIN = N_COL_WARPS  # 4
COL_THREADS = 32 * N_COL_WARPS  # 128

# --- weight mode (apply_rht=False, no MMA): col now does the heavy transposed SMEM read itself,
# so col and row get EQUAL warps for the 256-row supertile (their per-tile outputs are
# equal-sized; at 128-row col keeps 8 warps while row drops to 4). Col's 8 warps cover the
# 128 N-rows with 2 threads/row (each owns half the col-group blocks). ---
COL_WARP_END_W = 8  # warps 0..7 = col (2 threads / N-row)
ROW_WARP_BEGIN_W = 8
COL_THREADS_W = 32 * COL_WARP_END_W  # 256

TMEM_ALLOC_BAR = 1
TMEM_DEALLOC_BAR = 2
EPI_STORE_BAR = 3
ROW_STORE_BAR = 4
ROW_FP4_STAGES = 2

# Swizzled scale-factor layout (cutlass NVFP4): SF[r,c] -> [r//128, c//4, r%32, (r%128//32)*4 + c%4].
# Per super-tile, the SF tile has a 32x16 (=16B-wide) inner -> TMA-storable. Block inner = 32*16.
SF_BLK = 32 * 16  # 512 fp8 per (128-row x 4-col) swizzle block
SF_RGRP = (M_TILE // 16) // 4  # 2  : N-groups (c//4) per row super-tile (8 SF cols)


def _set_supertile_geometry(kernel_obj, col_groups_per_supertile: int):
    """Derive the supertile/warp geometry onto a kernel instance.

    col_groups_per_supertile = the number of 16-column blocks each main-loop iteration
    processes; a supertile spans kw = 16 * col_groups_per_supertile M-positions. 16 (a
    256-row supertile) serves M % 256 and is the tuned config; 8 (128-row) serves M % 128
    and is the floor (sf_rblk = kw//128 must stay >= 1; the plain col-SF box is
    col_groups_per_supertile bytes wide, so swizzle_sf=False needs 16). Row warps own one
    M-row per thread (row_threads == kw), so they scale with the supertile height.
    """
    assert col_groups_per_supertile in (8, 16), (
        f"col_groups_per_supertile must be 8 or 16, got {col_groups_per_supertile}"
    )
    kernel_obj.col_groups_per_supertile = col_groups_per_supertile
    kernel_obj.kw = K * col_groups_per_supertile
    kernel_obj.sf_gcol = (
        col_groups_per_supertile // 4
    )  # M-groups (c//4) per col super-tile
    kernel_obj.sf_rblk = kernel_obj.kw // 128  # M-blocks (r//128) per row super-tile
    n_row_warps = kernel_obj.kw // 32  # 8 for the 256-row supertile, 4 for 128-row
    kernel_obj.row_warp_end = ROW_WARP_BEGIN + n_row_warps
    kernel_obj.mma_warp = kernel_obj.row_warp_end
    kernel_obj.tma_warp = kernel_obj.mma_warp + 1
    kernel_obj.fused_tpb = 32 * (kernel_obj.tma_warp + 1)  # 448 / 320 threads
    kernel_obj.row_threads = 32 * n_row_warps  # == kw: 1 M-row per thread
    kernel_obj.row_warp_end_w = ROW_WARP_BEGIN_W + n_row_warps
    kernel_obj.tma_warp_w = kernel_obj.row_warp_end_w
    kernel_obj.fused_tpb_w = 32 * (kernel_obj.tma_warp_w + 1)  # 544 / 416 threads
    kernel_obj.row_threads_w = 32 * n_row_warps


def _round_rht_amax(amax):
    """``max|bf16(v)|`` from ``max|v|``: one rounding for a whole reduction.

    The Triton kernels truncate ``tl.dot``'s fp32 accumulator with ``.to(tl.bfloat16)``
    before both the amax and the quantize, matching TransformerEngine, whose RHT output
    is a bf16 tensor. The tcgen05 UMMA accumulator is fp32 and lives in TMEM, so
    consuming it raw would make every columnwise scale and code disagree with Triton.

    Rounding is only observable through two consumers -- the amax and the scaled value --
    so neither needs a rounded copy of the input. Round-to-nearest-even is monotonic in
    magnitude, so rounding an amax once equals rounding all 16 inputs and then reducing
    (NaN survives either order); the scaled values round pairwise inside the multiply
    loop (``rht_acc`` in ``_quant16_from_amax``). Materializing a rounded 16-value copy
    instead cost 29% on the grouped quantize.
    """
    return cutlass.Float32(cutlass.BFloat16(amax))


def _abs_amax16(vals):
    """Max abs over 16 f32 rmem values (the 1x16 block amax)."""
    amax = _abs_f32(vals[0])
    for i in range(1, 16):
        amax = _max_f32(amax, _abs_f32(vals[i]))
    return amax


def _group16_amax(amax, deltas: cutlass.Constexpr = (8, 4, 2, 1)):
    """Reduce a 1x16 block amax to the 16x16 (2D) block amax via a butterfly max over the
    lane group that holds the block's 16 orthogonal 1x16 strips. With one strip per lane
    that group is 16 lanes wide (xor offsets 8/4/2/1); a caller holding two strips per lane
    has already folded one level in-register and passes the 8-lane offsets (4/2/1). Either
    way the offsets stay inside an aligned lane group, so every lane ends with the shared
    block max."""
    for delta in deltas:
        amax = _max_f32(amax, cute.arch.shuffle_sync_bfly(amax, delta))
    return amax


def _enc_from_amax(amax, enc_over_fp4max, dec, fast_math: cutlass.Constexpr = False):
    """Block amax -> (encode multiplier, stored E4M3 scale).

    Split out of ``_quant16_from_amax`` so a caller that shares one 16x16 block amax across
    several 1x16 strips pays for the E4M3 round-trip and the exact reciprocal once.
    """
    # Cap at FP8_E4M3_MAX only, with no lower clamp: TE emits a zero per-vector scale for an
    # all-zero vector and when a nonzero scale underflows in E4M3, so imposing a nonzero floor
    # would diverge from the TE ground truth (mirrors the triton _nvfp4_quantize). A zero
    # pvscale drives the encode reciprocal to FP32_MAX. This exactly preserves an all-zero
    # vector; in the underflow case, dequantization intentionally loses the tiny nonzero data.
    pvscale = _min_f32(amax * enc_over_fp4max, cutlass.Float32(FP8_E4M3_MAX))
    pv_f32 = cute.make_rmem_tensor((4,), cutlass.Float32)
    for i in range(4):
        pv_f32[i] = pvscale
    pv_f8 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
    pv_f8.store(pv_f32.load().to(cutlass.Float8E4M3FN))
    pvscale_fp8 = pv_f8[0]
    pv_back = cute.make_rmem_tensor((4,), cutlass.Float32)
    pv_back.store(pv_f8.load().to(cutlass.Float32))
    denom = pv_back[0] * dec
    enc = _min_f32(
        cute.arch.rcp_approx(denom)
        if cutlass.const_expr(fast_math)
        else _div_rn_f32(cutlass.Float32(1.0), denom),
        cutlass.Float32(FP32_MAX),
    )
    return enc, pvscale_fp8


def _pack16_rn_from_enc(vals, enc, rht_acc: cutlass.Constexpr = False):
    """16 f32 values + encode multiplier -> the two packed-FP4 u32 words, RTNE.

    ``rht_acc`` selects the variant that first rounds the raw accumulator through bfloat16.
    """
    pack8 = (
        _mul_cvt_rn_e2m1x8_acc_f32
        if cutlass.const_expr(rht_acc)
        else _mul_cvt_rn_e2m1x8_f32
    )
    w0 = pack8(
        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], enc
    )
    w1 = pack8(
        vals[8],
        vals[9],
        vals[10],
        vals[11],
        vals[12],
        vals[13],
        vals[14],
        vals[15],
        enc,
    )
    return w0, w1


def _pack16_rs_from_enc(vals, enc, rb, rht_acc: cutlass.Constexpr = False):
    """16 f32 values + encode multiplier + 4 random words -> the two packed-FP4 u32 words, SR.

    The stochastic-rounding twin of ``_pack16_rn_from_enc``. ``rb`` covers the values in
    groups of four (``rb[0]`` for ``vals[0:4]``, ``rb[1]`` for ``vals[4:8]``, ...), which is
    the grouping ``cvt.rs`` consumes.
    """
    pack8 = (
        _mul_cvt_rs_e2m1x8_acc_f32
        if cutlass.const_expr(rht_acc)
        else _mul_cvt_rs_e2m1x8_f32
    )
    w0 = pack8(
        vals[0],
        vals[1],
        vals[2],
        vals[3],
        vals[4],
        vals[5],
        vals[6],
        vals[7],
        enc,
        rb[0],
        rb[1],
    )
    w1 = pack8(
        vals[8],
        vals[9],
        vals[10],
        vals[11],
        vals[12],
        vals[13],
        vals[14],
        vals[15],
        enc,
        rb[2],
        rb[3],
    )
    return w0, w1


def _quant16_from_amax(
    vals,
    amax,
    enc_over_fp4max,
    dec,
    sr: cutlass.Constexpr = False,
    rb=None,
    rht_acc: cutlass.Constexpr = False,
    fast_math: cutlass.Constexpr = False,
):
    """Quantize 16 f32 values to NVFP4 (w0,w1 packed u32, pvscale_fp8) using a given block amax
    (1x16 or a shared 16x16 amax). sr selects stochastic rounding over RTNE.

    ``rht_acc`` marks ``vals`` as a raw tcgen05 RHT accumulator. Exact mode rounds it
    through bfloat16 for TE-default compatibility; fast mode consumes FP32 directly and
    uses TE's approximate FTZ reciprocal. The caller is responsible for rounding
    ``amax`` in exact mode (see ``_round_rht_amax``)."""
    enc, pvscale_fp8 = _enc_from_amax(amax, enc_over_fp4max, dec, fast_math)
    # Fast math consumes the FP32 accumulator directly, so it takes the plain primitive
    # even when vals is a raw RHT accumulator: the bfloat16 round-through is exact-mode only.
    use_acc = cutlass.const_expr(rht_acc and not fast_math)
    if cutlass.const_expr(sr):
        w0, w1 = _pack16_rs_from_enc(vals, enc, rb, use_acc)
    else:
        w0, w1 = _pack16_rn_from_enc(vals, enc, use_acc)
    return w0, w1, pvscale_fp8


def _quant16(
    vals,
    enc_over_fp4max,
    dec,
    sr: cutlass.Constexpr = False,
    rb=None,
    rht_acc: cutlass.Constexpr = False,
    fast_math: cutlass.Constexpr = False,
):
    """1x16 NVFP4 quantize -> (w0,w1 packed u32, pvscale_fp8). vals: 16-elem f32 rmem.

    ``rht_acc=True`` for a raw RHT accumulator: the block amax rounds once (monotonic),
    the values round pairwise in the multiply loop."""
    amax = _abs_amax16(vals)
    if cutlass.const_expr(rht_acc and not fast_math):
        amax = _round_rht_amax(amax)
    return _quant16_from_amax(
        vals, amax, enc_over_fp4max, dec, sr, rb, rht_acc, fast_math
    )


class _Tcgen05RowColFused:
    def __init__(
        self,
        swizzle_sf: bool = True,
        sr: bool = False,
        apply_rht: bool = True,
        grouped: bool = False,
        col_groups_per_supertile: int = 16,
        fast_math: bool = False,
    ):
        # swizzle_sf=True: cutlass NVFP4 swizzled SF (GEMM-ready, TMA-coalesced store).
        # False: plain (N,M//16)/(M,N//16) SF (row SF falls back to a strided SIMT store); its
        # col-SF box is col_groups_per_supertile bytes wide, so it requires 16 (TMA 16B min).
        # sr=True: stochastic rounding (cvt.rs) in the FP4 cast; False: round-to-nearest.
        # apply_rht=True: columnwise path = NVFP4(RHT(A.t())) via the tcgen05 UMMA (the B operand is
        # the Hadamard matrix). False (weight quantize): plain NVFP4(A.t()) — the col warps read the
        # transposed A from SMEM directly (no MMA/TMEM/acc-pipeline, no B). That weight path also
        # emits 2D 16x16 block scaling.
        # Activations (apply_rht=True) keep the standard 1x16 scaling. The RHT path is compiled
        # separately, so its codegen is unchanged.
        # grouped=True: A is E equal-sized experts stacked into (E*M, N), each with its own
        # global amax. Experts are uniform and contiguous and M % kw == 0, so a work tile never
        # straddles two of them: the rowwise outputs are byte-identical to the ungrouped
        # (E*M, N) result (the 128x4 SF swizzle is per-128-row-block) and only the columnwise
        # stores need an expert offset on their tile coordinate. Compiled as its own variant so
        # the two shipped ungrouped kernels keep their codegen exactly, and so grouped never has
        # to be crossed with sr (whose col seed would alias across experts) or with apply_rht.
        # col_groups_per_supertile=16 (256-row supertile) serves M % 256; 8 serves M % 128.
        assert not (col_groups_per_supertile < 16 and not swizzle_sf), (
            "swizzle_sf=False requires col_groups_per_supertile=16"
        )
        self.swizzle_sf = swizzle_sf
        self.sr = sr
        self.apply_rht = apply_rht
        self.grouped = grouped
        self.fast_math = fast_math
        _set_supertile_geometry(self, col_groups_per_supertile)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mFP4: cute.Tensor,
        mSF: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        global_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        GRID: cutlass.Int32,
        NUM_EXPERTS: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.c_layout = utils.LayoutEnum.from_tensor(mFP4)

        mma_op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            MMA_TILER,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))

        # --- wide A: K = 16*col_groups -> col_groups k-blocks (M-blocks); MN_SW128 swizzle ---
        a_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        a_shape = tiled_mma.partition_shape_A(
            cute.dice((M_TILE, N_TILE, self.kw), (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            a_atom, cute.append(a_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )
        # clean (M_mma=128, KW, STAGE) view of the SAME bytes for the row read
        # (same atom + swizzle -> identical physical mapping to a_smem_layout_staged).
        a_clean_layout = cute.tile_to_shape(
            a_atom, (M_TILE, self.kw, NUM_AB_STAGE), order=(0, 1, 2)
        )

        # --- narrow B: one 16x16 RHT, 1 k-block ---
        b_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.BFloat16
        )
        b_shape = tiled_mma.partition_shape_B(cute.dice(MMA_TILER, (None, 1, 1)))
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            b_atom, cute.append(b_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            a_smem_layout,
            (M_TILE, N_TILE, self.kw),
            tiled_mma,
            (1, 1, 1, 1),
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            b_smem_layout,
            MMA_TILER,
            tiled_mma,
            (1, 1, 1, 1),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(
                cute.append(acc_shape, self.col_groups_per_supertile), NUM_ACC_STAGE
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        # Weight mode (apply_rht=False) skips the B (Hadamard) load — no MMA — so don't count it
        # in the TMA tx_count, or the AB-full barrier would wait on bytes that never arrive.
        num_tma_load_bytes = (
            M_TILE * self.kw + (N_TILE * K if cutlass.const_expr(self.apply_rht) else 0)
        ) * 2

        # COL FP4 store: super-tile = 2U u32 wide
        fp4_smem_layout = cute.make_layout(
            (M_TILE, 2 * self.col_groups_per_supertile),
            stride=(2 * self.col_groups_per_supertile, 1),
        )
        tma_atom_fp4, tma_tensor_fp4 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mFP4,
            fp4_smem_layout,
            (M_TILE, 2 * self.col_groups_per_supertile),
        )
        # COL SF store (TMA). swizzled: (1, SF_GCOL*SF_BLK) box over flat (N//128, (M//64)*SF_BLK);
        # plain: (M_TILE, col_groups) box over (N, M//16).
        if cutlass.const_expr(self.swizzle_sf):
            col_sf_box = (1, self.sf_gcol * SF_BLK)
        else:
            col_sf_box = (M_TILE, self.col_groups_per_supertile)
        sf_smem_layout = cute.make_layout(col_sf_box, stride=(col_sf_box[1], 1))
        tma_atom_sf, tma_tensor_sf = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mSF,
            sf_smem_layout,
            col_sf_box,
        )

        # ROW FP4 store: super-tile = (KW M-rows, M_TILE//8 u32), inner = 64B wide -> TMA-ok
        row_fp4_smem_layout = cute.make_layout(
            (self.kw, M_TILE // 8), stride=(M_TILE // 8, 1)
        )
        tma_atom_row_fp4, tma_tensor_row_fp4 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mRowFP4,
            row_fp4_smem_layout,
            (self.kw, M_TILE // 8),
        )

        # ROW SF store. swizzled: TMA, (SF_RBLK, SF_RGRP*SF_BLK) box over flat (M//128, (N//64)*SF_BLK).
        # plain: strided SIMT, so this atom is unused (alias the FP4 atom).
        if cutlass.const_expr(self.swizzle_sf):
            row_sf_box = (self.sf_rblk, SF_RGRP * SF_BLK)
            row_sf_smem_layout = cute.make_layout(row_sf_box, stride=(row_sf_box[1], 1))
            tma_atom_row_sf, tma_tensor_row_sf = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mRowSF,
                row_sf_smem_layout,
                row_sf_box,
            )
        else:
            tma_atom_row_sf, tma_tensor_row_sf = (
                tma_atom_row_fp4,
                tma_tensor_row_fp4,
            )  # unused

        num_tiles_m = N // cutlass.Int32(M_TILE)  # N output-row tiles (M_mma=128)
        num_tiles_ns = M // cutlass.Int32(
            N_TILE * self.col_groups_per_supertile
        )  # M contraction block-groups (K=16U)
        num_super = num_tiles_m * num_tiles_ns
        # grouped: M is the stacked E*M_expert extent, so pid_ns // tiles_ns_per_expert is the
        # expert and num_tiles_m is the per-expert stride of the columnwise output's row tiles.
        tiles_ns_per_expert = num_tiles_ns // NUM_EXPERTS

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mFP4,
            mSF,
            global_amax_t,
            sr_rng_t,
            tma_atom_fp4,
            tma_tensor_fp4,
            tma_atom_sf,
            tma_tensor_sf,
            mRowFP4,
            mRowSF,
            row_amax_t,
            tma_atom_row_fp4,
            tma_tensor_row_fp4,
            row_fp4_smem_layout,
            tma_atom_row_sf,
            tma_tensor_row_sf,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            fp4_smem_layout,
            sf_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_tiles_ns,
            num_super,
            GRID,
            tiles_ns_per_expert,
            num_tiles_m,
        ).launch(
            grid=(GRID, 1, 1),
            block=(
                self.fused_tpb
                if cutlass.const_expr(self.apply_rht)
                else self.fused_tpb_w,
                1,
                1,
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mFP4: cute.Tensor,
        mSF: cute.Tensor,
        global_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        tma_atom_fp4: cute.CopyAtom,
        mFP4_tma: cute.Tensor,
        tma_atom_sf: cute.CopyAtom,
        mSF_tma: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        tma_atom_row_fp4: cute.CopyAtom,
        mRowFP4_tma: cute.Tensor,
        row_fp4_smem_layout: cute.Layout,
        tma_atom_row_sf: cute.CopyAtom,
        mRowSF_tma: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        fp4_smem_layout: cute.Layout,
        sf_smem_layout: cute.Layout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_tiles_ns: cutlass.Int32,
        num_super: cutlass.Int32,
        GRID: cutlass.Int32,
        tiles_ns_per_expert: cutlass.Int32,
        col_tiles_per_expert: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        start_pid, _, _ = cute.arch.block_idx()

        # Warp layout differs by mode: RHT uses the 4-col/8-row + MMA split; weight mode drops the
        # MMA and balances col=8/row=8 (col now does the heavy transposed read, not a TMEM epilogue).
        if cutlass.const_expr(self.apply_rht):
            _COL_END, _ROW_BEG, _ROW_END, _TMA_W = (
                COL_WARP_END,
                ROW_WARP_BEGIN,
                self.row_warp_end,
                self.tma_warp,
            )
            _COL_THR, _ROW_THR = COL_THREADS, self.row_threads
        else:
            _COL_END, _ROW_BEG, _ROW_END, _TMA_W = (
                COL_WARP_END_W,
                ROW_WARP_BEGIN_W,
                self.row_warp_end_w,
                self.tma_warp_w,
            )
            _COL_THR, _ROW_THR = COL_THREADS_W, self.row_threads_w

        if warp_idx == _TMA_W:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_fp4)
            cpasync.prefetch_descriptor(tma_atom_sf)
            cpasync.prefetch_descriptor(tma_atom_row_fp4)
            if cutlass.const_expr(self.swizzle_sf):
                cpasync.prefetch_descriptor(tma_atom_row_sf)

        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_AB_STAGE * 2]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_ACC_STAGE * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # AB pipeline: TMA producer. RHT mode consumers = MMA (1 umma arrive) + ROW_THREADS;
        # weight mode (no MMA) consumers = COL_THREADS_W + ROW_THREADS_W (both warp groups read sA).
        if cutlass.const_expr(self.apply_rht):
            ab_cons_count = 1 + self.row_threads
        else:
            ab_cons_count = COL_THREADS_W + self.row_threads_w
        ab_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, ab_cons_count)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=NUM_AB_STAGE,
            producer_group=ab_prod_grp,
            consumer_group=ab_cons_grp,
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        acc_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, N_COL_WARPS)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=NUM_ACC_STAGE,
            producer_group=acc_prod_grp,
            consumer_group=acc_cons_grp,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 * (N_COL_WARPS + 1),  # col + mma
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR,
            num_threads=COL_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # --- SMEM: allocate A raw, build two views (MMA swizzle-on-ptr + row swizzle-in-layout) ---
        a_cosize = cute.cosize(a_smem_layout_staged.outer)
        raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        # row view: SAME swizzled pointer, clean (M_mma=128, KW, STAGE) *outer* layout
        # -> swizzle applied identically to sA (both PDSL), just a different logical grouping.
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)

        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sFP4 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=fp4_smem_layout,
            byte_alignment=128,
        )
        # COL SF SMEM. swizzled: raw bytes with a 4D write-view (group, 32, 16) + a 2D TMA-view
        # over the same memory; plain: a single (M_TILE, col_groups) tile (write == TMA view).
        if cutlass.const_expr(self.swizzle_sf):
            raw_csf = smem.allocate_array(
                cutlass.Float8E4M3FN, self.sf_gcol * SF_BLK, byte_alignment=128
            )
            sSF_w = cute.make_tensor(
                raw_csf,
                cute.make_layout(
                    (1, self.sf_gcol, 32, 16),
                    stride=(self.sf_gcol * SF_BLK, SF_BLK, 16, 1),
                ),
            )
            sSF = cute.make_tensor(
                raw_csf, sf_smem_layout
            )  # (1, SF_GCOL*SF_BLK) TMA view
        else:
            sSF = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=sf_smem_layout,
                byte_alignment=128,
            )
            sSF_w = sSF  # (M_TILE, col_groups) write == TMA

        # double-buffered row FP4 staging: overlap TMA store with next iter's compute
        row_fp4_staged = cute.make_layout(
            (self.kw, M_TILE // 8, ROW_FP4_STAGES),
            stride=(M_TILE // 8, 1, self.kw * (M_TILE // 8)),
        )
        sRowFP4 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=row_fp4_staged,
            byte_alignment=128,
        )
        # ROW SF SMEM (swizzled only; plain row SF is a strided SIMT store, no staging).
        # Double-buffered (same as row FP4) so its TMA store overlaps the next iter's compute.
        sRowSF_w = None
        sRowSF = None
        if cutlass.const_expr(self.swizzle_sf):
            _rsf_stage = self.sf_rblk * SF_RGRP * SF_BLK
            raw_rsf = smem.allocate_array(
                cutlass.Float8E4M3FN, _rsf_stage * ROW_FP4_STAGES, byte_alignment=128
            )
            sRowSF_w = cute.make_tensor(
                raw_rsf,
                cute.make_layout(
                    (self.sf_rblk, SF_RGRP, 32, 16, ROW_FP4_STAGES),
                    stride=(SF_RGRP * SF_BLK, SF_BLK, 16, 1, _rsf_stage),
                ),
            )
            sRowSF = cute.make_tensor(
                raw_rsf,
                cute.make_layout(
                    (self.sf_rblk, SF_RGRP * SF_BLK, ROW_FP4_STAGES),
                    stride=(SF_RGRP * SF_BLK, 1, _rsf_stage),
                ),
            )

        # --- global -> mma partition (wide A tiler = (M_TILE, KW)) ---
        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((M_TILE, N_TILE, self.kw), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)

        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, M, col_groups, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)

        def _global_scale(amax):
            is_zero = amax == cutlass.Float32(0.0)
            safe = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), amax))
            c = _min_f32(
                _div_rn_f32(cutlass.Float32(FP8_E4M3_MAX * FP4_E2M1_MAX), safe),
                cutlass.Float32(FP32_MAX),
            )
            c = cutlass.Float32(
                cutlass.select_(c == cutlass.Float32(0.0), cutlass.Float32(1.0), c)
            )
            enc = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), c))
            dec = _div_rn_f32(cutlass.Float32(1.0), enc)
            return enc, dec, enc * cutlass.Float32(1.0 / FP4_E2M1_MAX)

        # Ungrouped: one global scale pair for the whole launch. Grouped: the epilogues overwrite
        # these per work tile from that tile's expert; the hoisted pair still has to be computed
        # (and typed) before the loops, since the DSL rejects a name that enters a dynamic `for`
        # untyped and gains a type inside it.
        _, g_dec, enc_over_fp4max = _global_scale(global_amax_t[0])  # col (RHT)
        _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[0])  # row (plain)
        col_state, row_state = None, None
        if cutlass.const_expr(self.sr):
            col_state = philox_prep(
                cutlass.Uint32(sr_rng_t[0]),
                cutlass.Uint32(sr_rng_t[1]),
                cutlass.Uint32(sr_rng_t[2]),
            )
            row_state = philox_prep(
                cutlass.Uint32(sr_rng_t[4]),
                cutlass.Uint32(sr_rng_t[5]),
                cutlass.Uint32(sr_rng_t[6]),
            )
        # Tile geometry for the stochastic-rounding counter, on a 128x128 grid. Counters
        # are derived from global (token, hidden) coordinates rather than from this
        # kernel's supertile or its traversal order, so the 256-row supertile spanning two
        # token tiles costs nothing and the stream stays a pure function of position.
        # col_tiles_per_expert is num_tiles_m (N // M_TILE), the hidden-tile count -- its
        # grouped name is moot here, since sr and grouped are never compiled together.
        tri_tiles_hid = col_tiles_per_expert

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        rem = num_super - start_pid
        num_iters = cutlass.select_(
            rem > cutlass.Int32(0),
            (rem + GRID - cutlass.Int32(1)) // GRID,
            cutlass.Int32(0),
        )

        # ==================== TMA warp (AB producer) ====================
        if warp_idx == _TMA_W:
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, pid_m, pid_ns, 0)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                if cutlass.const_expr(
                    self.apply_rht
                ):  # B (Hadamard) only needed for the MMA
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, 0)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
            ab_producer.tail()

        # ==================== MMA warp (AB consumer, acc producer) ====================
        if warp_idx == self.mma_warp and cutlass.const_expr(self.apply_rht):
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_ACC_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                ab_handle = ab_consumer.wait_and_advance()
                acc_pipeline.producer_acquire(acc_producer_state)
                for u in cutlass.range_constexpr(self.col_groups_per_supertile):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                        tCrA[(None, None, u, ab_handle.index)],
                        tCrB[(None, None, 0, ab_handle.index)],
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                    )
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                ab_handle.release()  # 1 UMMA arrive on AB empty barrier
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== ROW warps (AB consumer, read raw sA) ====================
        if warp_idx >= _ROW_BEG and warp_idx < _ROW_END:
            k_row = tidx - _ROW_BEG * cutlass.Int32(
                32
            )  # 0..KW-1 = M-position within super-tile
            row_store_barrier = pipeline.NamedBarrier(
                barrier_id=ROW_STORE_BAR, num_threads=_ROW_THR
            )
            row_ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_AB_STAGE
            )
            blk = cute.make_rmem_tensor((16,), cutlass.Float32)
            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                m0 = pid_ns * cutlass.Int32(self.kw)  # global M-row base
                buf = local_iter % ROW_FP4_STAGES  # double-buffer index
                if cutlass.const_expr(self.grouped):
                    # The rowwise outputs are the stacked (E*M, N) result verbatim, so only the
                    # scale is expert-dependent here.
                    _, r_dec, r_enc_over_fp4max = _global_scale(
                        row_amax_t[pid_ns // tiles_ns_per_expert]
                    )

                ab_pipeline.consumer_wait(row_ab_state)  # wait sA full
                stage = row_ab_state.index

                # read this thread's M-row (128 N values across the m_mma grain), 8 SF-blocks.
                # The A atom is MN_SW128, so the N grain is the contiguous mode: each block of
                # 16 is one vector copy, not 16 scalar swizzled loads (same shape as the grouped
                # kernel's row epilogue).
                sA_row = sA_clean[(None, k_row, stage)]
                for b in cutlass.range_constexpr(M_TILE // 16):  # 8 blocks of 16 N
                    cute.autovec_copy(cute.local_tile(sA_row, (16,), (b,)), rBlk)
                    rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                    for j in cutlass.range_constexpr(8):
                        blk[2 * j] = _bf16lo_to_f32(rWords[j])
                        blk[2 * j + 1] = _bf16hi_to_f32(rWords[j])
                    row_rb = None
                    if cutlass.const_expr(self.sr):
                        # This thread owns supertile row k_row, i.e. token tile
                        # pid_ns*(kw/128) + k_row//128 at local token k_row%128. One draw
                        # per 16-element block, indexed by its position in the rowwise
                        # (tokens, hidden) tile.
                        tile_id = (
                            pid_ns * cutlass.Int32(self.kw // M_TILE)
                            + k_row // cutlass.Int32(M_TILE)
                        ) * tri_tiles_hid + pid_m
                        row_rb = philox4_all(
                            row_state,
                            tile_id * cutlass.Int32(TILE_BLOCKS)
                            + (k_row % cutlass.Int32(M_TILE))
                            * cutlass.Int32(M_TILE // 16)
                            + cutlass.Int32(b),
                        )
                    amax = _abs_amax16(blk)
                    if cutlass.const_expr(not self.apply_rht):
                        amax = _group16_amax(amax)
                    w0, w1, sf = _quant16_from_amax(
                        blk,
                        amax,
                        r_enc_over_fp4max,
                        r_dec,
                        self.sr,
                        row_rb,
                        fast_math=self.fast_math,
                    )
                    sRowFP4[k_row, b * 2, buf] = w0
                    sRowFP4[k_row, b * 2 + 1, buf] = w1
                    if cutlass.const_expr(self.swizzle_sf):
                        # swizzled SF[r=m0+k_row, c=pid_m*8+b] -> [r//128, c//4, r%32, (r%128//32)*4 + c%4]
                        sRowSF_w[
                            k_row // cutlass.Int32(128),
                            b // 4,
                            k_row % cutlass.Int32(32),
                            ((k_row // cutlass.Int32(32)) % cutlass.Int32(4))
                            * cutlass.Int32(4)
                            + (b % 4),
                            buf,
                        ] = sf
                    else:
                        mRowSF[
                            m0 + k_row,
                            pid_m * cutlass.Int32(M_TILE // 16) + cutlass.Int32(b),
                        ] = sf

                # all M-rows read -> release AB buffer (1 thread arrive each)
                cute.arch.mbarrier_arrive(
                    ab_pipeline.sync_object_empty.get_barrier(stage)
                )
                row_ab_state.advance()

                # TMA-store the (KW, M_TILE//8) row FP4 tile from buffer `buf`.
                # wait_group(1) keeps <=1 store in flight -> this store overlaps the next
                # iter's read/quant; the 2nd barrier makes its completion (via the *next*
                # iter's wait_group) visible before buf is reused two iters later.
                cute.arch.fence_proxy("async.shared", space="cta")
                row_store_barrier.arrive_and_wait()
                if warp_idx == cutlass.Int32(_ROW_BEG):
                    gRowFP4 = cute.local_tile(
                        mRowFP4_tma, (self.kw, M_TILE // 8), (pid_ns, pid_m)
                    )
                    tRs, tRg = cpasync.tma_partition(
                        tma_atom_row_fp4,
                        0,
                        cta_layout,
                        cute.group_modes(sRowFP4[(None, None, buf)], 0, 2),
                        cute.group_modes(gRowFP4, 0, 2),
                    )
                    cute.copy(tma_atom_row_fp4, tRs, tRg)
                    if cutlass.const_expr(self.swizzle_sf):
                        gRowSF = cute.local_tile(
                            mRowSF_tma,
                            (self.sf_rblk, SF_RGRP * SF_BLK),
                            (pid_ns, pid_m),
                        )
                        tRSs, tRSg = cpasync.tma_partition(
                            tma_atom_row_sf,
                            0,
                            cta_layout,
                            cute.group_modes(sRowSF[(None, None, buf)], 0, 2),
                            cute.group_modes(gRowSF, 0, 2),
                        )
                        cute.copy(tma_atom_row_sf, tRSs, tRSg)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                row_store_barrier.arrive_and_wait()
            if warp_idx == cutlass.Int32(_ROW_BEG):
                cute.arch.cp_async_bulk_wait_group(0, read=True)  # drain last store

        # ==================== COL epilogue warps (acc consumer, TMEM) ====================
        if warp_idx < COL_WARP_END and cutlass.const_expr(self.apply_rht):
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                MMA_TILER,
                self.c_layout,
                cutlass.Float32,
                cutlass.Float32,
                MMA_TILER[:2],
                False,
            )
            tAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tAcc_epi = cute.flat_divide(tAcc, MMA_TILER[:2])
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)
            # Swizzled col SF: the write-view's last mode is contiguous and u indexes it,
            # so four consecutive groups land in four adjacent bytes. Stage them in a
            # register tile and commit one 4B store instead of four scattered STS.U8.
            rSF4 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)

            epi_store_barrier = pipeline.NamedBarrier(
                barrier_id=EPI_STORE_BAR, num_threads=COL_THREADS
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_ACC_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                pid_m_col, pid_ns_col = pid_m, pid_ns
                if cutlass.const_expr(self.grouped):
                    # col outputs are (E*N, M//8) and (E*(N//128), (M//64)*512): shift the row
                    # tile by this expert's block of N tiles, rebase the col tile inside it.
                    e = pid_ns // tiles_ns_per_expert
                    pid_m_col = pid_m + e * col_tiles_per_expert
                    pid_ns_col = pid_ns - e * tiles_ns_per_expert
                    _, g_dec, enc_over_fp4max = _global_scale(global_amax_t[e])
                acc_idx = acc_consumer_state.index

                acc_pipeline.consumer_wait(acc_consumer_state)
                for u in cutlass.range_constexpr(self.col_groups_per_supertile):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_base[(None, None, None, 0, 0, u, acc_idx)],
                        tTR_rAcc,
                    )
                    vals = tTR_rAcc.load().reshape((16,))
                    col_rb = None
                    if cutlass.const_expr(self.sr):
                        # This thread's 16 tokens start at supertile offset u*16, so they
                        # sit in token tile pid_ns*(kw/128) + u//8 at local token
                        # (u%8)*16. One draw per 16-element block, indexed by its position
                        # in the columnwise (hidden, tokens) tile.
                        tile_id = (
                            pid_ns * cutlass.Int32(self.kw // M_TILE)
                            + cutlass.Int32(u // (M_TILE // 16))
                        ) * tri_tiles_hid + pid_m
                        col_rb = philox4_all(
                            col_state,
                            tile_id * cutlass.Int32(TILE_BLOCKS)
                            + tidx * cutlass.Int32(M_TILE // 16)
                            + cutlass.Int32(u % (M_TILE // 16)),
                        )
                    w0, w1, pvscale_fp8 = _quant16(
                        vals,
                        enc_over_fp4max,
                        g_dec,
                        self.sr,
                        col_rb,
                        rht_acc=True,
                        fast_math=self.fast_math,
                    )

                    sFP4[tidx, u * 2] = w0
                    sFP4[tidx, u * 2 + 1] = w1
                    if cutlass.const_expr(self.swizzle_sf):
                        # swizzled SF[r=pid_m*128+tidx, c=pid_ns*16+u] -> [r//128, c//4, r%32, (r%128//32)*4 + c%4]
                        # c%4 is the contiguous mode, so u..u+3 are adjacent bytes: stage
                        # four and store them as one word.
                        rSF4[u % 4] = pvscale_fp8
                        if cutlass.const_expr(u % 4 == 3):
                            cute.autovec_copy(
                                rSF4,
                                cute.local_tile(
                                    sSF_w[(0, u // 4, tidx % cutlass.Int32(32), None)],
                                    (4,),
                                    (tidx // cutlass.Int32(32),),
                                ),
                            )
                    else:
                        sSF_w[tidx, u] = pvscale_fp8

                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                cute.arch.fence_proxy("async.shared", space="cta")
                epi_store_barrier.arrive_and_wait()
                if warp_idx == cutlass.Int32(0):
                    gFP4 = cute.local_tile(
                        mFP4_tma,
                        (M_TILE, 2 * self.col_groups_per_supertile),
                        (pid_m_col, pid_ns_col),
                    )
                    tSs, tSg = cpasync.tma_partition(
                        tma_atom_fp4,
                        0,
                        cta_layout,
                        cute.group_modes(sFP4, 0, 2),
                        cute.group_modes(gFP4, 0, 2),
                    )
                    cute.copy(tma_atom_fp4, tSs, tSg)
                    if cutlass.const_expr(self.swizzle_sf):
                        gSF = cute.local_tile(
                            mSF_tma, (1, self.sf_gcol * SF_BLK), (pid_m_col, pid_ns_col)
                        )
                    else:
                        gSF = cute.local_tile(
                            mSF_tma,
                            (M_TILE, self.col_groups_per_supertile),
                            (pid_m_col, pid_ns_col),
                        )
                    tSFs, tSFg = cpasync.tma_partition(
                        tma_atom_sf,
                        0,
                        cta_layout,
                        cute.group_modes(sSF, 0, 2),
                        cute.group_modes(gSF, 0, 2),
                    )
                    cute.copy(tma_atom_sf, tSFs, tSFg)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epi_store_barrier.arrive_and_wait()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ========= COL weight-mode warps: plain NVFP4(A.t()), read transposed A from SMEM =========
        # No MMA: the col warps are AB consumers (like the row warps) and read A.t() directly from
        # sA_clean — the same swizzled bytes the row path reads, in the transposed grain.
        if warp_idx < _COL_END and cutlass.const_expr(not self.apply_rht):
            col_store_barrier = pipeline.NamedBarrier(
                barrier_id=EPI_STORE_BAR, num_threads=_COL_THR
            )
            col_ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_AB_STAGE
            )
            # 8 col warps (256 threads) cover the 128 N-rows two rows at a time: a thread owns
            # the adjacent pair (nrow, nrow+1) and a quarter of the col-group blocks. N is the
            # contiguous SMEM mode, so the pair is one 32-bit load rather than two 16-bit ones
            # (a warp then moves the full 128B instead of 64B), and because the pair sits in the
            # same 16x16 block both rows share one amax, one E4M3 scale and one reciprocal.
            nrow = (tidx % cutlass.Int32(64)) * cutlass.Int32(2)
            u_quarter = tidx // cutlass.Int32(64)
            blk0 = cute.make_rmem_tensor((16,), cutlass.Float32)
            blk1 = cute.make_rmem_tensor((16,), cutlass.Float32)
            rPair = cute.make_rmem_tensor((2,), cutlass.BFloat16)
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                pid_m_col, pid_ns_col = pid_m, pid_ns
                if cutlass.const_expr(self.grouped):
                    # col outputs are (E*N, M//8) and (E*(N//128), (M//64)*512): shift the row
                    # tile by this expert's block of N tiles, rebase the col tile inside it.
                    e = pid_ns // tiles_ns_per_expert
                    pid_m_col = pid_m + e * col_tiles_per_expert
                    pid_ns_col = pid_ns - e * tiles_ns_per_expert
                    _, g_dec, enc_over_fp4max = _global_scale(global_amax_t[e])

                ab_pipeline.consumer_wait(col_ab_state)  # wait sA full
                stage = col_ab_state.index
                for u_local in cutlass.range_constexpr(
                    self.col_groups_per_supertile // 4
                ):
                    u = cutlass.Int32(u_local) + u_quarter * cutlass.Int32(
                        self.col_groups_per_supertile // 4
                    )
                    for i in cutlass.range_constexpr(16):
                        mpos = u * cutlass.Int32(16) + cutlass.Int32(
                            i
                        )  # M-position (0..255)
                        # transposed read: A.t()[N-rows nrow, nrow+1][M-pos=mpos]
                        cute.autovec_copy(
                            cute.local_tile(
                                sA_clean[(None, (mpos % 8, mpos // 8), (0, stage))],
                                (2,),
                                (nrow // 2,),
                            ),
                            rPair,
                        )
                        blk0[i] = rPair[0].to(cutlass.Float32)
                        blk1[i] = rPair[1].to(cutlass.Float32)
                    # The pair is two of the 16 strips of one 16x16 block: fold them in-register,
                    # then butterfly over the 8 lanes holding the other 14.
                    amax = _group16_amax(
                        _max_f32(_abs_amax16(blk0), _abs_amax16(blk1)), (4, 2, 1)
                    )
                    # Weight mode is RTNE only (asserted at compile), so no SR draw here.
                    enc, sf = _enc_from_amax(amax, enc_over_fp4max, g_dec)
                    w0, w1 = _pack16_rn_from_enc(blk0, enc)
                    v0, v1 = _pack16_rn_from_enc(blk1, enc)
                    sFP4[nrow, u * 2] = w0
                    sFP4[nrow, u * 2 + 1] = w1
                    sFP4[nrow + 1, u * 2] = v0
                    sFP4[nrow + 1, u * 2 + 1] = v1
                    if cutlass.const_expr(self.swizzle_sf):
                        sSF_w[
                            0,
                            u // cutlass.Int32(4),
                            nrow % cutlass.Int32(32),
                            (nrow // cutlass.Int32(32)) * cutlass.Int32(4)
                            + (u % cutlass.Int32(4)),
                        ] = sf
                        sSF_w[
                            0,
                            u // cutlass.Int32(4),
                            (nrow + 1) % cutlass.Int32(32),
                            ((nrow + 1) // cutlass.Int32(32)) * cutlass.Int32(4)
                            + (u % cutlass.Int32(4)),
                        ] = sf
                    else:
                        sSF_w[nrow, u] = sf
                        sSF_w[nrow + 1, u] = sf

                cute.arch.mbarrier_arrive(
                    ab_pipeline.sync_object_empty.get_barrier(stage)
                )
                col_ab_state.advance()

                cute.arch.fence_proxy("async.shared", space="cta")
                col_store_barrier.arrive_and_wait()
                if warp_idx == cutlass.Int32(0):
                    gFP4 = cute.local_tile(
                        mFP4_tma,
                        (M_TILE, 2 * self.col_groups_per_supertile),
                        (pid_m_col, pid_ns_col),
                    )
                    tSs, tSg = cpasync.tma_partition(
                        tma_atom_fp4,
                        0,
                        cta_layout,
                        cute.group_modes(sFP4, 0, 2),
                        cute.group_modes(gFP4, 0, 2),
                    )
                    cute.copy(tma_atom_fp4, tSs, tSg)
                    if cutlass.const_expr(self.swizzle_sf):
                        gSF = cute.local_tile(
                            mSF_tma, (1, self.sf_gcol * SF_BLK), (pid_m_col, pid_ns_col)
                        )
                    else:
                        gSF = cute.local_tile(
                            mSF_tma,
                            (M_TILE, self.col_groups_per_supertile),
                            (pid_m_col, pid_ns_col),
                        )
                    tSFs, tSFg = cpasync.tma_partition(
                        tma_atom_sf,
                        0,
                        cta_layout,
                        cute.group_modes(sSF, 0, 2),
                        cute.group_modes(gSF, 0, 2),
                    )
                    cute.copy(tma_atom_sf, tSFs, tSFg)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                col_store_barrier.arrive_and_wait()


class _Tcgen05RhtAmax:
    """One-pass tensor-core RHT amax over the dual-consumer A load.

    Each epilogue reduces to a global max-abs instead of quantizing:
      col_amax = max|RHT(A.t())|  (TMEM accumulator),  row_amax = max|A|  (raw sA).
    The 16x16 RHT runs on tensor cores, so the pass is HBM-bound. Requires
    M % (16 * col_groups_per_supertile) == 0 (16 -> M % 256, 8 -> M % 128), N % 128 == 0.
    """

    def __init__(self, col_groups_per_supertile: int = 16):
        _set_supertile_geometry(self, col_groups_per_supertile)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        col_amax_t: cute.Tensor,
        row_amax_t: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        GRID: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mma_op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            MMA_TILER,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))

        a_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        a_shape = tiled_mma.partition_shape_A(
            cute.dice((M_TILE, N_TILE, self.kw), (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            a_atom, cute.append(a_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )
        a_clean_layout = cute.tile_to_shape(
            a_atom, (M_TILE, self.kw, NUM_AB_STAGE), order=(0, 1, 2)
        )

        b_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.BFloat16
        )
        b_shape = tiled_mma.partition_shape_B(cute.dice(MMA_TILER, (None, 1, 1)))
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            b_atom, cute.append(b_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            a_smem_layout,
            (M_TILE, N_TILE, self.kw),
            tiled_mma,
            (1, 1, 1, 1),
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            b_smem_layout,
            MMA_TILER,
            tiled_mma,
            (1, 1, 1, 1),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(
                cute.append(acc_shape, self.col_groups_per_supertile), NUM_ACC_STAGE
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        num_tma_load_bytes = (M_TILE * self.kw + N_TILE * K) * 2
        num_tiles_ns = M // cutlass.Int32(N_TILE * self.col_groups_per_supertile)
        num_super = (N // cutlass.Int32(M_TILE)) * num_tiles_ns

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            col_amax_t,
            row_amax_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_tiles_ns,
            num_super,
            GRID,
        ).launch(grid=(GRID, 1, 1), block=(self.fused_tpb, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        col_amax_t: cute.Tensor,
        row_amax_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_tiles_ns: cutlass.Int32,
        num_super: cutlass.Int32,
        GRID: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        start_pid, _, _ = cute.arch.block_idx()
        lane = tidx % cutlass.Int32(32)

        if warp_idx == self.tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_AB_STAGE * 2]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_ACC_STAGE * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # AB pipeline: TMA producer; consumers = MMA (1 umma arrive) + ROW_THREADS (thread arrives)
        ab_cons_count = 1 + self.row_threads
        ab_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, ab_cons_count)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=NUM_AB_STAGE,
            producer_group=ab_prod_grp,
            consumer_group=ab_cons_grp,
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        acc_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, N_COL_WARPS)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=NUM_ACC_STAGE,
            producer_group=acc_prod_grp,
            consumer_group=acc_cons_grp,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 * (N_COL_WARPS + 1),  # col + mma
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR,
            num_threads=COL_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # SMEM: A raw + two views (MMA swizzle-on-ptr + row clean view), B
        a_cosize = cute.cosize(a_smem_layout_staged.outer)
        raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)
        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((M_TILE, N_TILE, self.kw), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)

        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        rem = num_super - start_pid
        num_iters = cutlass.select_(
            rem > cutlass.Int32(0),
            (rem + GRID - cutlass.Int32(1)) // GRID,
            cutlass.Int32(0),
        )

        # ==================== TMA warp (AB producer) ====================
        if warp_idx == self.tma_warp:
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, pid_m, pid_ns, 0)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()

        # ==================== MMA warp (AB consumer, acc producer) ====================
        if warp_idx == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_ACC_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                ab_handle = ab_consumer.wait_and_advance()
                acc_pipeline.producer_acquire(acc_producer_state)
                for u in cutlass.range_constexpr(self.col_groups_per_supertile):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                        tCrA[(None, None, u, ab_handle.index)],
                        tCrB[(None, None, 0, ab_handle.index)],
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                    )
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                ab_handle.release()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== ROW warps (AB consumer, read raw sA) -> row amax ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < self.row_warp_end:
            k_row = tidx - cutlass.Int32(
                ROW_WARP_BEGIN * 32
            )  # 0..KW-1 = M-position within super-tile
            row_ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_AB_STAGE
            )
            thread_row_max = cutlass.Float32(0.0)
            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            for local_iter in cutlass.range(num_iters):
                ab_pipeline.consumer_wait(row_ab_state)
                stage = row_ab_state.index
                # read this thread's M-row (128 N values across the m_mma grain). The A
                # atom is MN_SW128, so the N grain is the contiguous mode: each block of
                # 16 is one vector copy, not 16 scalar swizzled loads (the same shape the
                # fused kernel's row epilogue and the grouped amax already use).
                sA_row = sA_clean[(None, k_row, stage)]
                for b in cutlass.range_constexpr(M_TILE // 16):  # 8 blocks of 16 N
                    cute.autovec_copy(cute.local_tile(sA_row, (16,), (b,)), rBlk)
                    rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                    for j in cutlass.range_constexpr(8):
                        thread_row_max = _max_f32(
                            thread_row_max, _abs_f32(_bf16lo_to_f32(rWords[j]))
                        )
                        thread_row_max = _max_f32(
                            thread_row_max, _abs_f32(_bf16hi_to_f32(rWords[j]))
                        )
                cute.arch.mbarrier_arrive(
                    ab_pipeline.sync_object_empty.get_barrier(stage)
                )
                row_ab_state.advance()
            for offset in [16, 8, 4, 2, 1]:
                thread_row_max = _max_f32(
                    thread_row_max, cute.arch.shuffle_sync_bfly(thread_row_max, offset)
                )
            if lane == cutlass.Int32(0):
                _atom_max_f32_nonneg(row_amax_t.iterator, thread_row_max)

        # ==================== COL epilogue warps (acc consumer, TMEM) -> col amax ====================
        if warp_idx < COL_WARP_END:
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                MMA_TILER,
                self.c_layout,
                cutlass.Float32,
                cutlass.Float32,
                MMA_TILER[:2],
                False,
            )
            tAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tAcc_epi = cute.flat_divide(tAcc, MMA_TILER[:2])
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_ACC_STAGE
            )
            thread_col_max = cutlass.Float32(0.0)
            for local_iter in cutlass.range(num_iters):
                acc_idx = acc_consumer_state.index
                acc_pipeline.consumer_wait(acc_consumer_state)
                for u in cutlass.range_constexpr(self.col_groups_per_supertile):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_base[(None, None, None, 0, 0, u, acc_idx)],
                        tTR_rAcc,
                    )
                    vals = tTR_rAcc.load().reshape((16,))
                    for i in cutlass.range_constexpr(16):
                        thread_col_max = _max_f32(thread_col_max, _abs_f32(vals[i]))
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            # One bf16 rounding for the whole reduction; see _round_rht_amax.
            thread_col_max = _round_rht_amax(thread_col_max)
            for offset in [16, 8, 4, 2, 1]:
                thread_col_max = _max_f32(
                    thread_col_max, cute.arch.shuffle_sync_bfly(thread_col_max, offset)
                )
            if lane == cutlass.Int32(0):
                _atom_max_f32_nonneg(col_amax_t.iterator, thread_col_max)

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ---------------------------------------------------------------------------
# Public entry: device amax + fused kernel -> reads A in place, writes fresh outputs.
# Small per-device state (RHT/identity/RNG buffers, compiled kernels) is cached; under
# CUDA graphs it must be pre-allocated via cutedsl_prepare_for_cuda_graph before capture.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _get_rht_buffer(sign_vector, device_idx):
    """Tiny (16,16,1) RHT torch buffer, cached per (sign_vector, device).

    maxsize=None keeps every key resident; the key count is bounded by the number of distinct
    sign vectors in use (each buffer is a 16x16 bf16 tensor). Transposed (H^T) so the MMA
    operand layout yields v @ H.
    """
    device = torch.device("cuda", device_idx)
    rht_nk = (
        get_rht_matrix(sign_vector, device, torch.bfloat16, HADAMARD_DIM)
        .t()
        .contiguous()
    )
    return rht_nk.reshape(N_TILE, K, 1)


@functools.lru_cache(maxsize=None)
def _get_identity_buffer(device_idx):
    """(16,16,1) placeholder for the kernel's ``B`` (Hadamard) operand on the weight-quantize
    (``apply_rht=False``) path. That path has no MMA, so B is never loaded; this only supplies a
    valid tensor for the B TMA-atom shape."""
    device = torch.device("cuda", device_idx)
    eye = torch.eye(HADAMARD_DIM, dtype=torch.bfloat16, device=device).contiguous()
    return eye.reshape(N_TILE, K, 1)


@functools.lru_cache(maxsize=None)
def _get_sr_rng_buffer(device_idx):
    """Persistent (8,) int32 stochastic-rounding Philox state buffer, cached per device.

    Holds the caller's ``[col_seed, col_offset, row_seed, row_offset]`` int64s viewed as
    little-endian 32-bit halves, which is the form Philox keys and counters take:
    ``[col_seed_lo, col_seed_hi, col_off_lo, col_off_hi, row_seed_lo, ...]``.

    The fused kernel reads its SR state from this *stable* address. A fresh per-call tensor would be
    an untracked live allocation in the CUDA-graph pool (``torch.compile(mode="reduce-overhead")``)
    AND a correctness hazard — the captured kernel would read a recycled address on replay. The SR
    path copies the per-call state in (the copy is captured, so each graph replay re-reads the
    freshly-advanced offset); the RTNE path never reads it (the kernel guards the read behind
    ``const_expr(sr)``), so its value is irrelevant there.
    """
    return torch.zeros(8, dtype=torch.int32, device=torch.device("cuda", device_idx))


@functools.lru_cache(maxsize=None)
def _compile_amax_tc_kernel(device_idx, col_groups_per_supertile=16):
    """Compile the tensor-core RHT amax with symbolic shapes (cached per device+u).

    The symbolic (sym_int) shapes make the compiled kernel serve any (M % (16*u), N % 128);
    M/N/GRID are runtime Int32 args. Not keyed on the sign vector, which is a runtime launch
    buffer (the compile uses a fake), so the compiled kernel is identical for every sign vector.
    """
    device = torch.device("cuda", device_idx)
    # aT = A.t().unsqueeze(-1): (N, M, 1), dim0 (N) contiguous -> stride (1, N, 1).
    m_sym = cute.sym_int(
        divisibility=N_TILE * col_groups_per_supertile
    )  # M % 256 or M % 128
    n_sym = cute.sym_int(divisibility=M_TILE)  # N % 128
    fake_aT = make_fake_tensor(
        cutlass.BFloat16, (n_sym, m_sym, 1), stride=(1, cute.sym_int(), 1)
    )
    fake_bT = make_fake_tensor(
        cutlass.BFloat16, (HADAMARD_DIM, HADAMARD_DIM, 1), stride=(HADAMARD_DIM, 1, 1)
    )
    fake_col = make_fake_tensor(cutlass.Float32, (1,), stride=(1,))
    fake_row = make_fake_tensor(cutlass.Float32, (1,), stride=(1,))
    k = _Tcgen05RhtAmax(col_groups_per_supertile=col_groups_per_supertile)
    # c_layout for the TMEM->reg read = layout of the col FP4 output (row-major 2D); the enum
    # depends only on row/col-majorness, so a dummy contiguous tensor suffices.
    dummy = torch.empty((M_TILE, 16), dtype=torch.int32, device=device)
    k.c_layout = utils.LayoutEnum.from_tensor(from_dlpack(dummy))
    return cute.compile(
        k,
        fake_aT,
        fake_bT,
        fake_col,
        fake_row,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_rht_amax_impl(A: torch.Tensor, sign_vector=DEFAULT_SIGN_VECTOR):
    """Global amaxes for NVFP4 two-level scaling.

    Returns (col_amax, row_amax) as scalar (1,) float32 tensors:
      - col_amax = max|RHT(A.t())|  (the columnwise path quantizes RHT data)
      - row_amax = max|A|           (the rowwise path quantizes A directly)

    The column amax is taken over the post-RHT data (not the plain amax) for correctness: RHT
    can raise the per-block max, and a too-small global scale saturates the E4M3 block scales.
    Requires M % 128 == 0, N % 128 == 0.
    """
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    if not A.is_contiguous():
        raise ValueError("A must be row-major (contiguous)")
    M, N = A.shape
    # The tile is N_TILE*col_groups wide in M. Below that divisibility, GRID=0
    # -> a no-op launch that silently returns amax=0.
    if M % 128 != 0:
        raise ValueError(f"M must be divisible by 128, got M={M}")
    if N % 128 != 0:
        raise ValueError(f"N must be divisible by 128, got N={N}")
    col_groups_per_supertile = 16 if M % 256 == 0 else 8
    # This is a non-differentiable op (autograd is owned by the outer linear Function);
    # detach so the input passed to the kernel never carries autograd state.
    A = A.detach()
    dev = A.device
    col_amax = torch.zeros(1, dtype=torch.float32, device=dev)
    row_amax = torch.zeros(1, dtype=torch.float32, device=dev)

    rht_nk = _get_rht_buffer(tuple(sign_vector), dev.index)  # torch buffer (kept alive)

    NUM_SMS = _get_num_sms(dev.index)
    GRID = min(NUM_SMS, (N // M_TILE) * (M // (N_TILE * col_groups_per_supertile)))
    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))

    amax_compiled = _compile_amax_tc_kernel(dev.index, col_groups_per_supertile)
    amax_compiled(
        A.t().unsqueeze(-1),
        rht_nk,
        col_amax,
        row_amax,
        int(M),
        int(N),
        int(GRID),
        stream,
    )
    return col_amax, row_amax


# maxsize=None: the key is
# (device, swizzle, sr, apply_rht, grouped, col_groups_per_supertile, fast_math)
# and every entry is a compiled kernel that a CUDA-graph capture may depend on. An eviction
# would force a lazy recompile mid-capture, so the cache must never evict.
#
# Every parameter is required and every caller passes it positionally, deliberately: an
# lru_cache key is the literal (args, kwargs) shape, so ``f(i, True, False, apply_rht=False)``
# and ``f(i, True, False, False)`` are two entries compiling the same kernel. Defaults here
# once let cutedsl_prepare_for_cuda_graph warm a set of keys no runtime call could hit, which
# silently turned the whole pre-capture warm-up into a no-op.
@functools.lru_cache(maxsize=None)
def _compile_fused_kernel(
    device_idx,
    swizzle,
    sr,
    apply_rht,
    grouped,
    col_groups_per_supertile,
    fast_math,
):
    """Compile the fused kernel with symbolic shapes (cached per device+flags+supertile).

    The symbolic (sym_int) shapes make the compiled kernel serve any (M % (16*u), N % 128); the
    divisibilities below match each output's TMA store box so the atoms tile cleanly. ``swizzle``
    selects the cutlass-swizzled SF layout (op default) vs the plain (N, M//16)/(M, N//16) layout.
    ``sr`` and ``fast_math`` compile separate arithmetic variants. ``apply_rht=False`` compiles the
    no-MMA weight-quantize variant (the col path reads transposed A from SMEM). ``grouped=True``
    compiles the dense-expert variant: M is the stacked E*M_expert extent, the amaxes are (E,), and
    the columnwise stores carry an expert offset. Not keyed on the sign vector / RNG (runtime launch
    buffers).
    """
    # The grouped variant is only reachable from the weight quantize: apply_rht would need a
    # per-expert B operand. Weight mode itself is RTNE only -- the 2D wrappers never expose
    # stochastic rounding -- so its columnwise epilogue takes no SR draw.
    assert not (grouped and apply_rht)
    assert not (sr and not apply_rht), "weight mode (apply_rht=False) is RTNE only"
    assert not (fast_math and not apply_rht), "fast math is only supported for RHT"
    # M % 256 or M % 128
    m_sym = cute.sym_int(divisibility=N_TILE * col_groups_per_supertile)
    n_sym = cute.sym_int(divisibility=M_TILE)  # N % 128
    free = cute.sym_int  # a fresh dynamic stride per call
    # Reusing a sym_int ties the two extents together in the compiled signature. The col outputs
    # are as tall as A is wide (N) ungrouped, but E*N grouped, so they need their own symbol
    # there. The row outputs stay on m_sym: their height is A's width, E*M, either way.
    cn_sym = cute.sym_int(divisibility=M_TILE) if grouped else n_sym

    # aT = A.t().unsqueeze(-1): (N, M, 1), dim0 contiguous; output FP4 tensors row-major.
    fake_aT = make_fake_tensor(
        cutlass.BFloat16, (n_sym, m_sym, 1), stride=(1, free(), 1)
    )
    fake_bT = make_fake_tensor(
        cutlass.BFloat16, (HADAMARD_DIM, HADAMARD_DIM, 1), stride=(HADAMARD_DIM, 1, 1)
    )
    # col_fp4 (N, M//8) u32, store box inner = 2*col_groups; row_fp4 (M, N//8) u32, inner = 16.
    fake_cfp4 = make_fake_tensor(
        cutlass.Uint32,
        (cn_sym, cute.sym_int(divisibility=2 * col_groups_per_supertile)),
        stride=(free(), 1),
    )
    fake_rfp4 = make_fake_tensor(
        cutlass.Uint32,
        (m_sym, cute.sym_int(divisibility=M_TILE // 8)),
        stride=(free(), 1),
    )
    if swizzle:
        # SF flattened to 2D for the TMA atom: col_sf.reshape(N//128, (M//64)*512) has inner
        # inner divisible by (col_groups//4)*512 (from the M bound);
        # row_sf.reshape(M//128, (N//64)*512) outer divisible by supertile M-blocks,
        # inner by 1024 (from N%128).
        fake_csf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (
                cute.sym_int(divisibility=1),
                cute.sym_int(divisibility=(col_groups_per_supertile // 4) * SF_BLK),
            ),
            stride=(free(), 1),
        )
        fake_rsf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (
                cute.sym_int(divisibility=(K * col_groups_per_supertile) // 128),
                cute.sym_int(divisibility=1024),
            ),
            stride=(free(), 1),
        )
    else:
        # plain SF: col (N, M//16), row (M, N//16). Requires col_groups=16 (col box is that many bytes wide).
        fake_csf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (cn_sym, cute.sym_int(divisibility=col_groups_per_supertile)),
            stride=(free(), 1),
        )
        fake_rsf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (m_sym, cute.sym_int(divisibility=M_TILE // 16)),
            stride=(free(), 1),
        )
    # grouped: the amaxes are (E,) rather than a scalar, indexed by the work tile's expert.
    fake_amax = make_fake_tensor(
        cutlass.Float32, (free() if grouped else 1,), stride=(1,)
    )
    # (8,) Philox state: [col_seed_lo/hi, col_off_lo/hi, row_seed_lo/hi, row_off_lo/hi].
    fake_sr_rng = make_fake_tensor(cutlass.Int32, (8,), stride=(1,))
    k = _Tcgen05RowColFused(
        swizzle_sf=swizzle,
        sr=sr,
        apply_rht=apply_rht,
        grouped=grouped,
        col_groups_per_supertile=col_groups_per_supertile,
        fast_math=fast_math,
    )
    return cute.compile(
        k,
        fake_aT,
        fake_bT,
        fake_cfp4,
        fake_csf,
        fake_rfp4,
        fake_rsf,
        fake_amax,
        fake_amax,
        fake_sr_rng,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_rht_quantize_row_col_impl(
    A: torch.Tensor,
    col_global_amax: torch.Tensor,
    row_global_amax: torch.Tensor,
    sign_vector=DEFAULT_SIGN_VECTOR,
    stochastic_rounding: bool = False,
    sr_rng: Optional[torch.Tensor] = None,
    *,
    swizzle_scale_factors: bool = True,
    compute_rowwise: bool = True,
    apply_rht: bool = True,
    use_fast_math: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused RHT + NVFP4 E2M1 columnwise quantize with rowwise quantize.

    The two global amaxes are taken as input: the caller computes them first (via
    ``cutedsl_rht_amax``, optionally all-reducing for TP). The columnwise path quantizes
    RHT(A.t()) scaled by ``col_global_amax``; the rowwise path quantizes A directly scaled
    by ``row_global_amax``.

    ``apply_rht=False`` selects the no-MMA weight-quantize variant: with no Hadamard rotation the
    columnwise warps read the transposed ``A`` straight from SMEM (a plain ``A.t()`` transpose-
    quantize) instead of an MMA accumulator, with col/row warps balanced. Used by the weight quantize
    (weights are not RHT-rotated); the caller passes ``col_global_amax == row_global_amax == max|A|``
    since ``max|A.t()| == max|A|``.

    Args:
        A: (M, N) bfloat16, row-major. M % 128 == 0 (M % 256 selects the faster
            256-row supertile; other M % 128 shapes compile a 128-row variant),
            N % 128 == 0.
        col_global_amax: scalar float32 = max|RHT(A.t())| (columnwise decode scale).
        row_global_amax: scalar float32 = max|A| (rowwise decode scale).
        sign_vector: RHT sign vector as a list of ints.
        stochastic_rounding: if True, both quant paths round via the Blackwell ``cvt.rs`` HW
            stochastic-rounding cvt seeded by ``sr_rng``. False -> RTNE (default).
        sr_rng: (4,) int64 ``[col_seed, col_offset, row_seed, row_offset]`` Philox state,
            required when ``stochastic_rounding=True``. The offsets are fresh per call so
            CUDA-graph replays advance the stream. Same state the Triton op takes.
        swizzle_scale_factors: cutlass NVFP4 swizzled SF (default, GEMM-ready). False -> plain
            (N,M//16)/(M,N//16) SF, which uses a slower strided row-SF store.
        compute_rowwise: return the rowwise output (default). False -> row_fp4/row_sf returned as
            None. NOTE: the fused kernel always computes + stores the row path; this flag only
            gates the *return*, it does not skip the rowwise work.

    Returns:
        4-tuple (col_fp4, col_sf, row_fp4, row_sf):
          - col_fp4: (N, M//2) uint8 packed FP4 (columnwise).
          - col_sf:  (N//128, M//64, 32, 16) float8_e4m3fn swizzled (or (N, M//16) plain).
          - row_fp4: (M, N//2) uint8 packed FP4 (rowwise), or None if compute_rowwise=False.
          - row_sf:  (M//128, N//64, 32, 16) float8_e4m3fn swizzled (or (M, N//16) plain), or None.
    """
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    if not A.is_contiguous():
        raise ValueError("A must be row-major (contiguous)")
    M, N = A.shape
    if M % 128 != 0:
        raise ValueError(f"M must be divisible by 128, got M={M}")
    if N % 128 != 0:
        raise ValueError(f"N must be divisible by 128, got N={N}")
    col_groups_per_supertile = 16 if M % 256 == 0 else 8
    if col_groups_per_supertile < 16 and not swizzle_scale_factors:
        raise ValueError(
            "swizzle_scale_factors=False requires M % 256 == 0 (the plain col-SF "
            "TMA box is col_groups_per_supertile bytes wide, below TMA's 16B "
            "minimum for the 128-row supertile)"
        )
    # Non-differentiable op (autograd owned by the outer linear Function); detach so the
    # input passed to the kernel never carries autograd state.
    A = A.detach()
    for name, t in (
        ("col_global_amax", col_global_amax),
        ("row_global_amax", row_global_amax),
    ):
        if t.numel() != 1:
            raise ValueError(f"{name} must contain a single element, got {t.numel()}")
        if t.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {t.dtype}")
    dev = A.device
    swizzle = bool(swizzle_scale_factors)
    sr = bool(stochastic_rounding)
    # Persistent buffer (stable address for CUDA-graph capture); see _get_sr_rng_buffer.
    sr_rng_t = _get_sr_rng_buffer(dev.index)
    if sr:
        if sr_rng is None:
            raise ValueError(
                "stochastic_rounding=True requires sr_rng (Philox state tensor)"
            )
        # [col_seed, col_offset, row_seed, row_offset] int64 -> the eight little-endian
        # 32-bit halves Philox keys and counters are built from. Written in-place
        # (captured by the graph) so each replay re-reads the freshly-advanced offset.
        sr_rng_t.copy_(sr_rng[:4].view(torch.int32))

    col_fp4 = torch.empty((N, M // 8), dtype=torch.uint32, device=dev)
    row_fp4 = torch.empty((M, N // 8), dtype=torch.uint32, device=dev)
    if swizzle:
        col_sf = torch.empty(
            (N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
        )
        row_sf = torch.empty(
            (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
        )
        csf_g = col_sf.reshape(
            N // 128, (M // 64) * 32 * 16
        )  # flat 2D for the TMA atom
        rsf_g = row_sf.reshape(M // 128, (N // 64) * 32 * 16)
    else:
        col_sf = torch.empty((N, M // 16), dtype=torch.float8_e4m3fn, device=dev)
        row_sf = torch.empty((M, N // 16), dtype=torch.float8_e4m3fn, device=dev)
        csf_g, rsf_g = col_sf, row_sf

    # The MMA B operand: the Hadamard (RHT) matrix, or an identity for a plain transpose-quantize.
    rht_nk = (
        _get_rht_buffer(tuple(sign_vector), dev.index)
        if apply_rht
        else _get_identity_buffer(dev.index)
    )  # torch buffer (kept alive)
    col_amax_t = col_global_amax.reshape(1)
    row_amax_t = row_global_amax.reshape(1)

    NUM_SMS = _get_num_sms(dev.index)
    GRID = min(NUM_SMS, (N // M_TILE) * (M // (N_TILE * col_groups_per_supertile)))
    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))

    fused = _compile_fused_kernel(
        dev.index,
        swizzle,
        sr,
        bool(apply_rht),
        False,  # grouped
        col_groups_per_supertile,
        bool(use_fast_math),
    )
    fused(
        A.t().unsqueeze(-1),
        rht_nk,
        col_fp4,
        csf_g,
        row_fp4,
        rsf_g,
        row_amax_t,
        col_amax_t,
        sr_rng_t,
        int(M),
        int(N),
        int(GRID),
        1,  # NUM_EXPERTS
        stream,
    )

    col_fp4_u8 = col_fp4.view(torch.uint8)  # (N, M//2)
    if compute_rowwise:
        return col_fp4_u8, col_sf, row_fp4.view(torch.uint8), row_sf
    return col_fp4_u8, col_sf, None, None


def _cutedsl_group_weight_quantize_2d_impl(
    A: torch.Tensor,
    global_amax: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense-expert 2D (16x16) NVFP4 E2M1 weight quantize, no RHT.

    Runs the ``apply_rht=False, grouped=True`` variant of the fused kernel once over the whole
    ``(E, M, N)`` stack. Experts are equal-sized and contiguous, so ``A`` is fed as the ``(E*M, N)``
    view: the rowwise outputs are then the ungrouped result verbatim (the 128x4 SF swizzle is
    per-128-row-block and 128 | M), and only the columnwise outputs need the per-expert tile offset
    the kernel applies. One global amax per expert; the col and row amaxes are the same tensor since
    ``max|A[e].t()| == max|A[e]|``.

    Args:
        A: (E, M, N) bfloat16, contiguous. M % 128 == 0, N % 128 == 0.
        global_amax: (E,) float32 per-expert ``A[e].float().abs().max()``.

    Returns:
        (E, M, N//2) u8 rowwise codes, (E, M//128, N//64, 32, 16) fp8 rowwise swizzled SF,
        (E, N, M//2) u8 colwise codes, (E, N//128, M//64, 32, 16) fp8 colwise swizzled SF.
    """
    E, M, N = A.shape
    # Non-differentiable op (autograd owned by the outer Function); detach so the input passed
    # to the kernel never carries autograd state.
    A = A.detach()
    dev = A.device

    row_fp4 = torch.empty((E, M, N // 8), dtype=torch.uint32, device=dev)
    col_fp4 = torch.empty((E, N, M // 8), dtype=torch.uint32, device=dev)
    row_sf = torch.empty(
        (E, M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )
    col_sf = torch.empty(
        (E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )

    NUM_SMS = _get_num_sms(dev.index)
    # Keyed on the per-expert M, not E*M: a supertile must not straddle two experts,
    # which is what lets the grouped kernel apply one expert offset per work tile.
    col_groups_per_supertile = 16 if M % 256 == 0 else 8
    GRID = min(NUM_SMS, (N // M_TILE) * (E * M // (N_TILE * col_groups_per_supertile)))
    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))

    fused = _compile_fused_kernel(
        dev.index,
        True,  # swizzle
        False,  # sr
        False,  # apply_rht
        True,  # grouped
        col_groups_per_supertile,
        False,  # fast_math
    )
    fused(
        A.view(E * M, N).t().unsqueeze(-1),
        _get_identity_buffer(dev.index),  # unused MMA B operand (no MMA on this path)
        col_fp4.view(E * N, M // 8),
        col_sf.reshape(E * (N // 128), (M // 64) * 32 * 16),
        row_fp4.view(E * M, N // 8),
        row_sf.reshape(E * (M // 128), (N // 64) * 32 * 16),
        global_amax,
        global_amax,
        _get_sr_rng_buffer(dev.index),  # unused (RTNE)
        int(E * M),
        int(N),
        int(GRID),
        int(E),
        stream,
    )
    # uint32 -> uint8 quadruples the last extent: (E, M, N//8) -> (E, M, N//2).
    return row_fp4.view(torch.uint8), row_sf, col_fp4.view(torch.uint8), col_sf
