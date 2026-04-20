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
  parts every quant kernel needs to do the same way: deriving the FLOOR-mode
  E8M0 scale, materializing the FP8 clamp limits, and quantize+pack of one
  4-element chunk into an i32 via two ``v_cvt_pk_fp8_f32`` instructions.

Helpers must be imported at MODULE level in the kernel files (not inside the
factory) so they look like Python globals to the AST rewriter rather than
free-variable closure cells. ``cutedsl_quantize_2d_1x32.py`` uses the same
pattern with ``cute_utils``.
"""

import importlib.util


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
    from flydsl.expr import arith, rocdl, vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.typing import T

    def floor_scale_and_inv_scale(amax_f32):
        """Derive the FLOOR-mode E8M0 byte and the matching inverse scale.

        Algorithm matches ``torchao.prototype.mx_formats.mx_tensor.to_mx`` with
        ``ScaleCalculationMode.FLOOR`` for FP8 E4M3FN (max_pos=448):

            bits         = bitcast(amax, i32)
            E_amax       = ((bits >> 23) & 0xFF) - 127
            scale_unb    = clamp(E_amax - 8, -127, 128)
            scale_biased = scale_unb + 127                  # uint8 E8M0 byte
            inv_scale    = 2 ^ (-scale_unb)                 # f32

        The compiler typically lowers ``scale_unb`` to ~3 ALU ops via constant
        folding (``v_bfe_u32`` + ``v_max_u32`` + ``v_add_u16``), and lowers
        ``inv_scale`` to a single ``v_ldexp_f32`` once it's used in a multiply.

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
        scale_unb = arith.minsi(scale_unb, arith.unwrap(fx.Int32(E8M0_EXPONENT_BIAS + 1)))

        scale_biased = scale_unb + fx.Int32(E8M0_EXPONENT_BIAS)
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

    def quantize_pack_chunk_to_i32(chunk_f32, inv_scale, f8_min_vec, f8_max_vec):
        """Quantize 4 f32 → 4 FP8 E4M3FN packed into one i32.

        Multiplies by ``inv_scale``, clamps to ``±F8_MAX`` (the FLOOR-mode
        post-scale clamp), then issues two ``v_cvt_pk_fp8_f32`` instructions
        to pack 4 results into a 32-bit register. The result can be written
        with one ``buffer_store_dword`` (or fused into a wider store by the
        scheduler).

        Args:
            chunk_f32: vec<4 x f32> input chunk.
            inv_scale: scalar f32 inverse scale.
            f8_min_vec, f8_max_vec: clamp constants from
                :func:`make_fp8_clamp_vectors`.

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
            res=T.i32, src_a=qv0, src_b=qv1, old=out, word_sel=False,
        )
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32, src_a=qv2, src_b=qv3, old=out, word_sel=True,
        )
        return out
