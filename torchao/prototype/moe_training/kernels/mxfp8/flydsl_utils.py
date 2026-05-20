# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for FlyDSL MXFP8 quantization kernels.

Counterpart of ``cute_utils.py``. Provides:

* Runtime detection (``_flydsl_runtime_available``) so kernel modules import
  cleanly even on hosts without FlyDSL installed.
* Centralized MXFP8 layout constants used by every kernel.
* Kernel-side helpers callable from inside an ``@flyc.kernel`` body — for the
  parts every quant kernel needs to do the same way: deriving the FLOOR- or
  RCEIL-mode E8M0 scale, materializing the FP8 clamp limits, and quantize+pack
  of one 4-element chunk into an i32 (FLOOR uses two ``v_cvt_pk_fp8_f32``;
  RCEIL uses the gfx950 fused ``v_cvt_scalef32_pk_fp8_f32``).

Helpers must be imported at MODULE level in the kernel files (not inside the
factory) so they look like Python globals to the AST rewriter rather than
free-variable closure cells. ``cutedsl_quantize_2d_1x32.py`` uses the same
pattern with ``cute_utils``.
"""

import importlib.util

import torch

# -----------------------------------------------------------------------------
# Runtime detection
# -----------------------------------------------------------------------------

_FLYDSL_RUNTIME_PACKAGES = {
    "flydsl": "flydsl",
    "flydsl.compiler": "flydsl",
    "flydsl.expr": "flydsl",
}


def _missing_flydsl_runtime_packages() -> list[str]:
    missing = []
    for module_name, package_name in _FLYDSL_RUNTIME_PACKAGES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None and package_name not in missing:
            missing.append(package_name)
    return missing


def _flydsl_runtime_available() -> bool:
    return len(_missing_flydsl_runtime_packages()) == 0


# -----------------------------------------------------------------------------
# MXFP8 layout constants (all kernels)
# -----------------------------------------------------------------------------

# OCP MX format spec: each block of 32 elements shares one E8M0 scale.
BLOCK_SIZE = 32

# FP8 E4M3FN max representable value; used as the clamp bound in FLOOR mode.
F8_MAX = 448.0

# E8M0 stores a biased uint8 exponent: stored = unbiased + bias.
E8M0_EXPONENT_BIAS = 127

# log2(F8_MAX) ≈ 8.81 → floor = 8. The FLOOR-mode scale subtracts this from
# the unbiased exponent of amax, putting amax/scale into [256, 512). Values in
# (448, 512) saturate to 448 in the post-quantize clamp; that's the documented
# FLOOR trade-off vs RCEIL.
_FP8_MAX_LOG2_FLOOR = 8

# Width of the cvt_pk_fp8_f32 packing operation (2 cvts give 4 fp8 in one i32).
VEC = 4
CHUNKS_PER_BLOCK = BLOCK_SIZE // VEC  # 8 chunks of 4 per quant block

# AMD wave size on the gfx9xx CDNA architectures we target.
AMD_WAVE_SIZE = 64


# -----------------------------------------------------------------------------
# Kernel-side helpers
# -----------------------------------------------------------------------------

if _flydsl_runtime_available():
    import flydsl.expr as fx
    from flydsl._mlir.dialects.arith import CmpIPredicate as _CmpIPredicate

    # Fused scaled fp8 cvt is not re-exported through flydsl.expr.rocdl yet.
    from flydsl._mlir.dialects.rocdl import (
        cvt_scalef32_pk_fp8_f32 as _scaled_cvt_pk_fp8_f32,
    )
    from flydsl.expr import arith, rocdl, vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T

    def current_stream_fast(device: torch.device) -> "fx.Stream":
        """Build an ``fx.Stream`` for the current PyTorch CUDA stream cheaply.

        ``torch.cuda.current_stream()`` takes ~2.6 µs per call because it
        constructs a Python ``Stream`` wrapper and a ``device`` object. The
        underlying private API ``torch._C._cuda_getCurrentStream(device_idx)``
        returns a ``(stream_ptr, device_idx, device_type)`` tuple in ~0.08 µs,
        and ``fx.Stream`` accepts the raw ``int`` pointer directly. Combined
        wrapper-side cost drops to ~0.2 µs — saves ~2.4 µs per launch, which
        is meaningful at small shapes where the kernel itself is <10 µs.

        The private API may disappear across PyTorch versions; fall back to
        the public ``torch.cuda.current_stream()`` if it does so we degrade
        gracefully rather than break at import time.
        """
        idx = device.index if device.index is not None else 0
        try:
            return fx.Stream(torch._C._cuda_getCurrentStream(idx)[0])
        except AttributeError:
            return fx.Stream(torch.cuda.current_stream(device).cuda_stream)

    def rceil_scale_and_pos_scale(amax_f32):
        """Derive RCEIL-mode E8M0 byte and matching f32 scale magnitude.

        Matches TransformerEngine's ``ptx::float_to_e8m0(amax * (1/F8_MAX))``
        and torchao's ``_to_mx_rceil`` for normal (non-denormal, non-NaN,
        non-Inf) amax values:

            descale     = amax / F8_MAX
            biased_exp  = (descale_bits >> 23) & 0xFF
            E8M0        = biased_exp + (mantissa > 0 ? 1 : 0)   # ceil(log2)
            pos_scale   = bitcast(E8M0 << 23, f32)              # 2^(E8M0-127)

        RCEIL guarantees ``amax / pos_scale ≤ F8_MAX`` exactly, so the gfx950
        ``v_cvt_scalef32_pk_fp8_f32`` instruction (which saturates overflow to
        NaN, not MAX) cannot produce NaN for finite inputs. Paired with
        ``quantize_pack_chunk_to_i32_rceil`` to lower to the 1-op fused cvt.
        FLOOR mode uses the 2-op ``cvt_pk_fp8_f32`` path with an explicit
        ``±F8_MAX`` clamp instead — see ``quantize_pack_chunk_to_i32_floor``.

        Special case: ``capped == 0`` (zero amax) → ``pos_scale_bits == 0`` →
        bitcast to f32 = +0.0; division by zero is undefined, so we select
        1.0 instead (matches TE's ``v_cndmask 1.0, v8, vcc`` pattern).

        Returns:
            Tuple ``(scale_biased_u8, pos_scale_f32)``.
        """
        inv_f8_max = fx.Float32(1.0 / F8_MAX)
        val = ArithValue(amax_f32) * inv_f8_max
        bits = ArithValue(val).bitcast(T.i32)
        biased_exp = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
        mantissa = bits & fx.Int32(0x7FFFFF)
        has_mantissa = arith.cmpi(
            _CmpIPredicate.ne, arith.unwrap(mantissa), arith.unwrap(fx.Int32(0))
        )
        inc_i32 = arith.extui(T.i32, has_mantissa)
        biased_exp_rceil = arith.unwrap(ArithValue(biased_exp) + ArithValue(inc_i32))
        capped = arith.minui(biased_exp_rceil, arith.unwrap(fx.Int32(0xFE)))
        scale_u8 = arith.trunci(T.i8, capped)

        pos_scale_bits = arith.shli(capped, arith.unwrap(fx.Int32(23)))
        is_zero = arith.cmpi(_CmpIPredicate.eq, capped, arith.unwrap(fx.Int32(0)))
        one_bits = arith.unwrap(fx.Int32(0x3F800000))
        selected_bits = arith.select(is_zero, one_bits, pos_scale_bits)
        pos_scale = ArithValue(selected_bits).bitcast(T.f32)
        return scale_u8, pos_scale

    def floor_scale_and_inv_scale(amax_f32):
        """Derive the FLOOR-mode E8M0 byte and the matching inverse scale.

        Algorithm matches ``torchao.prototype.mx_formats.mx_tensor.to_mx`` with
        ``ScaleCalculationMode.FLOOR`` for FP8 E4M3FN (max_pos=448):

            bits         = bitcast(amax, i32)
            E_amax       = ((bits >> 23) & 0xFF) - 127
            scale_unb    = clamp(E_amax - 8, -127, 128)
            scale_biased = scale_unb + 127                  # uint8 E8M0 byte
            inv_scale    = 2 ^ (-scale_unb)                 # f32

        Returns the *inverse* scale (``2^-scale_unb``) so the FLOOR pack path
        (:func:`quantize_pack_chunk_to_i32_floor`) can multiply instead of
        divide before the non-fused ``v_cvt_pk_fp8_f32``. The RCEIL helper
        returns ``pos_scale`` instead because the fused
        ``v_cvt_scalef32_pk_fp8_f32`` takes a divisor.

        Args:
            amax_f32: per-block absolute maximum, as an f32 ArithValue.

        Returns:
            Tuple ``(scale_biased_u8, inv_scale_f32)``.
        """
        bits = ArithValue(amax_f32).bitcast(T.i32)
        exp_biased = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
        e_amax = exp_biased - fx.Int32(E8M0_EXPONENT_BIAS)
        scale_unb = e_amax - fx.Int32(_FP8_MAX_LOG2_FLOOR)
        scale_unb = arith.maxsi(
            arith.unwrap(scale_unb), arith.unwrap(fx.Int32(-E8M0_EXPONENT_BIAS))
        )
        scale_unb = arith.minsi(
            scale_unb, arith.unwrap(fx.Int32(E8M0_EXPONENT_BIAS + 1))
        )

        scale_biased = ArithValue(scale_unb) + fx.Int32(E8M0_EXPONENT_BIAS)
        scale_u8 = arith.trunci(T.i8, arith.unwrap(scale_biased))

        neg_unb_f = arith.sitofp(T.f32, arith.unwrap(fx.Int32(0) - scale_unb))
        inv_scale = fx.math.exp2(neg_unb_f)
        return scale_u8, inv_scale

    def make_fp8_clamp_vectors():
        """Build ``vec<4 x f32>`` constants for ``±F8_MAX`` clamping.

        Returns:
            Tuple ``(f8_min_vec, f8_max_vec)`` of f32x4 vectors broadcasting
            ``-F8_MAX`` and ``+F8_MAX`` respectively. Used as the lhs of
            ``arith.maximumf`` / rhs of ``arith.minimumf`` in the post-scale
            clamp before the FP8 conversion (FLOOR mode).
        """
        f32x4 = T.vec(VEC, T.f32)
        f8_min = vector.broadcast(f32x4, arith.unwrap(fx.Float32(-F8_MAX)))
        f8_max = vector.broadcast(f32x4, arith.unwrap(fx.Float32(F8_MAX)))
        return f8_min, f8_max

    def quantize_pack_chunk_to_i32_rceil(chunk_f32, pos_scale):
        """RCEIL mode: quantize 4 f32 → 4 FP8 E4M3FN packed into one i32.

        Lowers to two ``v_cvt_scalef32_pk_fp8_f32`` instructions (gfx950 fused
        scale + convert with built-in saturate-to-NaN). RCEIL guarantees
        ``amax/scale ≤ F8_MAX`` so no NaN can fire; no explicit clamp needed.

        Args:
            chunk_f32: vec<4 x f32> input chunk.
            pos_scale: scalar f32 = ``2^scale_unb`` (the divisor; the
                instruction computes ``input / pos_scale``).

        Returns:
            i32 ArithValue with bytes ``[qv0, qv1, qv2, qv3]`` (low to high).
        """
        qv0 = vector.extract(chunk_f32, static_position=[0], dynamic_position=[])
        qv1 = vector.extract(chunk_f32, static_position=[1], dynamic_position=[])
        qv2 = vector.extract(chunk_f32, static_position=[2], dynamic_position=[])
        qv3 = vector.extract(chunk_f32, static_position=[3], dynamic_position=[])
        v2i16 = T.vec(2, T.i16)
        zero_i16 = arith.unwrap(fx.Int16(0))
        r = vector.from_elements(v2i16, [zero_i16, zero_i16])
        r = _scaled_cvt_pk_fp8_f32(
            res=v2i16,
            old_vdst=r,
            src0=qv0,
            src1=qv1,
            scale=arith.unwrap(pos_scale),
            dst_lo_hi_sel=False,
        )
        r = _scaled_cvt_pk_fp8_f32(
            res=v2i16,
            old_vdst=r,
            src0=qv2,
            src1=qv3,
            scale=arith.unwrap(pos_scale),
            dst_lo_hi_sel=True,
        )
        r_v1i32 = vector.bitcast(T.vec(1, T.i32), r)
        return vector.extract(r_v1i32, static_position=[0], dynamic_position=[])

    def quantize_pack_chunk_to_i32_floor(chunk_f32, inv_scale, f8_min_vec, f8_max_vec):
        """FLOOR mode: quantize 4 f32 → 4 FP8 E4M3FN packed into one i32.

        FLOOR's ``amax/scale ∈ [256, 512)`` lets the (448, 512) tail overflow
        the FP8 range, so we clamp explicitly with ``±F8_MAX`` before two
        ``v_cvt_pk_fp8_f32`` cvts. ``inv_scale`` comes from
        :func:`floor_scale_and_inv_scale`; ``f8_min_vec`` / ``f8_max_vec``
        from :func:`make_fp8_clamp_vectors`.

        Returns:
            i32 ArithValue with bytes ``[qv0, qv1, qv2, qv3]`` (low to high).
        """
        qv = chunk_f32 * inv_scale
        qv = arith.maximumf(qv, f8_min_vec)
        qv = arith.minimumf(qv, f8_max_vec)
        qv0 = vector.extract(qv, static_position=[0], dynamic_position=[])
        qv1 = vector.extract(qv, static_position=[1], dynamic_position=[])
        qv2 = vector.extract(qv, static_position=[2], dynamic_position=[])
        qv3 = vector.extract(qv, static_position=[3], dynamic_position=[])
        out = arith.unwrap(fx.Int32(0))
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32, src_a=qv0, src_b=qv1, old=out, word_sel=False
        )
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32, src_a=qv2, src_b=qv3, old=out, word_sel=True
        )
        return out
