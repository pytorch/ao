# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for FlyDSL MXFP8 quantization kernels.

Counterpart of ``cute_utils.py``. Provides:

* Runtime detection (``_flydsl_runtime_available``) so kernel modules import
  cleanly even on hosts without a supported FlyDSL installed.
* Centralized MXFP8 layout constants used by every kernel.
* Kernel-side helpers callable from inside an ``@flyc.kernel`` body — for the
  parts every quant kernel needs to do the same way: buffer loads/stores,
  the per-chunk amax, deriving the FLOOR- or RCEIL-mode E8M0 scale, and
  quantize+pack of one 4-element chunk into an i32 (FLOOR uses two
  ``v_cvt_pk_fp8_f32``; RCEIL uses the gfx950 fused
  ``v_cvt_scalef32_pk_fp8_f32``).

The helpers use FlyDSL's stable API (``docs/api_stability.md`` in the FlyDSL
repo), available from FlyDSL 0.3.4: buffer access goes through
``fx.rocdl.make_buffer_tensor`` views and ``fx.rocdl.BufferCopy*`` copy atoms,
which replace the ``flydsl.expr.buffer_ops`` and ``flydsl.expr.vector``
modules removed in FlyDSL 0.3.0, and arithmetic goes through ``fx.Int32`` /
``fx.Float32`` / ``fx.Vector``. The one exception is the fused scale+convert,
which FlyDSL does not wrap yet; it comes from the generated ROCDL dialect.

Helpers must be imported at MODULE level in the kernel files (not inside the
factory) so they look like Python globals to the AST rewriter rather than
free-variable closure cells. ``cutedsl_quantize_2d_1x32.py`` uses the same
pattern with ``cute_utils``.
"""

import importlib.metadata
import importlib.util
import os
from contextlib import nullcontext

import torch

# -----------------------------------------------------------------------------
# Runtime detection
# -----------------------------------------------------------------------------

_FLYDSL_RUNTIME_PACKAGES = {
    "flydsl": "flydsl",
    "flydsl.compiler": "flydsl",
    "flydsl.expr": "flydsl",
}

# Oldest FlyDSL whose stable API covers everything these kernels use.
_MIN_FLYDSL_VERSION = (0, 3, 4)


def _flydsl_version():
    """``(version_tuple, version_string)`` of the installed FlyDSL, or None."""
    try:
        version = importlib.metadata.version("flydsl")
    except importlib.metadata.PackageNotFoundError:
        return None
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts), version


def _missing_flydsl_runtime_packages() -> list[str]:
    missing = []
    for module_name, package_name in _FLYDSL_RUNTIME_PACKAGES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None and package_name not in missing:
            missing.append(package_name)
    if not missing:
        found = _flydsl_version()
        if found is not None and found[0] < _MIN_FLYDSL_VERSION:
            want = ".".join(map(str, _MIN_FLYDSL_VERSION))
            missing.append(f"flydsl>={want} (found {found[1]})")
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

# E8M0 byte that encodes NaN, and four FP8 E4M3FN NaNs packed in an i32.
_E8M0_NAN = 0xFF
_FP8_NAN_X4 = 0x7F7F7F7F


def waves_per_eu_hint(env_var: str):
    """Context manager applying the AMDGPU ``waves_per_eu`` hint from ``env_var``.

    A diagnostic tuning knob only. The hint is read at MLIR-compile time, which
    happens lazily on the first launch, so the context must wrap the launch
    call. ``CompilationContext`` is not part of FlyDSL's stable API, so it is
    imported only when the knob is set.
    """
    value = os.environ.get(env_var)
    if value is None:
        return nullcontext()
    from flydsl.compiler.kernel_function import CompilationContext

    return CompilationContext.compile_hints({"waves_per_eu": int(value)})


# -----------------------------------------------------------------------------
# Kernel-side helpers
# -----------------------------------------------------------------------------

if _flydsl_runtime_available():
    import flydsl.expr as fx

    # Not re-exported through flydsl.expr.rocdl yet (FlyDSL 0.3.4.1).
    from flydsl._mlir.dialects.rocdl import (
        cvt_scalef32_pk_fp8_f32 as _scaled_cvt_pk_fp8_f32,
    )
    from flydsl.expr import rocdl
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

    def buffer_view(tensor, vec: int):
        """Buffer-descriptor view of a 1-D tensor in units of ``vec`` elements.

        Column ``i`` of the returned view is elements ``[i * vec, (i + 1) * vec)``.
        """
        return fx.logical_divide(
            fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(vec, 1)
        )

    def buffer_load(view, idx, vec: int, dtype):
        """Load vector ``idx`` of a :func:`buffer_view` as an ``fx.Vector``."""
        atom = fx.make_copy_atom(rocdl.BufferCopy(vec * dtype.width), dtype.width)
        reg = fx.make_rmem_tensor(fx.make_layout(vec, 1), dtype)
        fx.copy(atom, fx.slice(view, (None, idx)), reg)
        return fx.memref_load_vec(reg)

    def buffer_store(values, view, idx, dtype, cache_modifier: int = 0):
        """Store a list of ``vec`` scalars to vector ``idx`` of a :func:`buffer_view`.

        ``cache_modifier`` is the buffer-instruction cache policy (0 = cached,
        2 = non-temporal).
        """
        vec = len(values)
        atom = fx.make_copy_atom(
            rocdl.BufferCopy(vec * dtype.width, cache_modifier), dtype.width
        )
        reg = fx.make_rmem_tensor(fx.make_layout(vec, 1), dtype)
        fx.memref_store_vec(fx.Vector.from_elements(values, dtype), reg)
        fx.copy(atom, reg, fx.slice(view, (None, idx)))

    def store_scale(e8m0, view, idx):
        """Store one E8M0 byte (an ``fx.Int32``) to element ``idx`` of a
        ``buffer_view(t, 1)``."""
        buffer_store([e8m0.to(fx.Uint8)], view, idx, fx.Uint8)

    def abs_max(vec_f32):
        """``max(|v|)`` over a 4-element ``fx.Vector`` of f32, propagating NaN.

        ``Float32.maximumf`` is IEEE ``maximum``: a NaN anywhere yields NaN and
        an Inf yields Inf, both of which the scale helpers below turn into the
        E8M0 NaN byte (NOSAT semantics, as in ``to_mx``). Combine per-chunk
        results with ``maximumf`` too. ``Vector.reduce(ReductionOp.MAX)`` lowers
        to ``maxnumf``, which drops NaN, so it is not used here.
        """
        a = fx.math.absf(vec_f32)
        return a[0].maximumf(a[1]).maximumf(a[2].maximumf(a[3]))

    def rceil_scale_and_pos_scale(amax_f32):
        """Derive RCEIL-mode E8M0 byte and matching f32 scale magnitude.

        Software equivalent of the Blackwell ``cvt.rp.ue8m0x2.f32`` conversion
        with ``NOSAT``, i.e. of ``_f32_to_e8m0_rceil`` in
        ``torchao/prototype/mx_formats/mx_tensor.py``. Keep the two in sync:

            descale     = amax * (1/F8_MAX)
            biased_exp  = (descale_bits >> 23) & 0xFF
            mantissa    = descale_bits & 0x7fffff
            round_up    = biased_exp == 0 ? mantissa > 0x400000 : mantissa != 0
            E8M0        = min(biased_exp + round_up, 0xff)      # ceil(log2)
            pos_scale   = bitcast(E8M0 << 23, f32)              # 2^(E8M0-127)

        RCEIL keeps ``amax / pos_scale`` at or fractionally above ``F8_MAX``
        (the descale rounding can overshoot by at most one FP32 ULP), which is
        far below the (448, 464] round-to-448 window, so the gfx950
        ``v_cvt_scalef32_pk_fp8_f32`` instruction (which saturates overflow to
        NaN, not MAX) cannot produce NaN for finite inputs. Paired with
        ``quantize_pack_chunk_to_i32_rceil`` to lower to the 1-op fused cvt.
        FLOOR mode uses the 2-op ``cvt_pk_fp8_f32`` path with an explicit
        ``±F8_MAX`` clamp instead — see ``quantize_pack_chunk_to_i32_floor``.

        E8M0 byte zero denotes ``2^-127`` rather than numerical zero; byte
        ``0xff`` denotes NaN.

        Returns:
            Tuple ``(e8m0, pos_scale)`` as ``(fx.Int32, fx.Float32)``.
        """
        bits = (amax_f32 * fx.Float32(1.0 / F8_MAX)).bitcast(fx.Int32)
        biased_exp = bits.shrui(fx.Int32(23)) & fx.Int32(0xFF)
        mantissa = bits & fx.Int32(0x7FFFFF)
        # Normal FP32 values round up whenever any mantissa bit is set. FP32
        # subnormals all share biased exponent 0, but E8M0 byte 0 already
        # encodes 2^-127 (mantissa 0x400000), so among subnormals only values
        # strictly above 2^-127 round up to byte 1.
        needs_round_up = (biased_exp == fx.Int32(0)).select(
            mantissa > fx.Int32(0x400000), mantissa != fx.Int32(0)
        )
        e8m0 = biased_exp + needs_round_up.select(fx.Int32(1), fx.Int32(0))
        # Saturating at 0xff yields E8M0 NaN for Inf and NaN amax (both have
        # biased exponent 0xff) and for finite overflow past 2^127 (0xfe + 1),
        # matching the NOSAT hardware.
        e8m0 = (e8m0 > fx.Int32(_E8M0_NAN)).select(fx.Int32(_E8M0_NAN), e8m0)
        pos_scale_bits = (e8m0 == fx.Int32(0)).select(
            fx.Int32(0x00400000), e8m0 << fx.Int32(23)
        )
        pos_scale_bits = (e8m0 == fx.Int32(_E8M0_NAN)).select(
            fx.Int32(0x7FC00000), pos_scale_bits
        )
        return e8m0, pos_scale_bits.bitcast(fx.Float32)

    def floor_scale_and_inv_scale(amax_f32):
        """Derive the FLOOR-mode E8M0 byte and the matching inverse scale.

        Algorithm matches ``torchao.prototype.mx_formats.mx_tensor.to_mx`` with
        ``ScaleCalculationMode.FLOOR`` for FP8 E4M3FN (max_pos=448):

            bits         = bitcast(amax, i32)
            E_amax       = ((bits >> 23) & 0xFF) - 127
            scale_unb    = clamp(E_amax - 8, -127, 128)
            scale_biased = scale_unb + 127                  # uint8 E8M0 byte
            inv_scale    = e8m0_to_f32(254 - scale_biased)  # f32

        The reciprocal is taken as ``254 - scale_biased`` in E8M0 space rather
        than as ``2^-scale_unb``, matching ``_e8m0_scale_to_reciprocal_fp32``
        in ``mx_tensor.py``; that keeps the two endpoints (2^-127 and NaN)
        exact without an ``exp2``.

        Returns the *inverse* scale so the FLOOR pack path
        (:func:`quantize_pack_chunk_to_i32_floor`) can multiply instead of
        divide before the non-fused ``v_cvt_pk_fp8_f32``. The RCEIL helper
        returns ``pos_scale`` instead because the fused
        ``v_cvt_scalef32_pk_fp8_f32`` takes a divisor.

        Args:
            amax_f32: per-block absolute maximum, as an ``fx.Float32``.

        Returns:
            Tuple ``(e8m0, inv_scale)`` as ``(fx.Int32, fx.Float32)``.
        """
        exp_biased = amax_f32.bitcast(fx.Int32).shrui(fx.Int32(23)) & fx.Int32(0xFF)
        scale_unb = exp_biased - fx.Int32(E8M0_EXPONENT_BIAS + _FP8_MAX_LOG2_FLOOR)
        lo = fx.Int32(-E8M0_EXPONENT_BIAS)
        hi = fx.Int32(E8M0_EXPONENT_BIAS + 1)
        scale_unb = (scale_unb < lo).select(lo, scale_unb)
        scale_unb = (scale_unb > hi).select(hi, scale_unb)

        is_special = exp_biased == fx.Int32(0xFF)
        scale_biased = is_special.select(
            fx.Int32(_E8M0_NAN), scale_unb + fx.Int32(E8M0_EXPONENT_BIAS)
        )

        reciprocal_biased = fx.Int32(254) - scale_biased
        # E8M0 byte 0 encodes the FP32 subnormal 2^-127, which the exponent-only
        # shift above cannot build. Currently unreachable, since the clamp above
        # keeps scale_biased <= 246 for finite amax, but every other backend
        # special-cases it and the helper should be correct standalone.
        reciprocal_bits = (reciprocal_biased == fx.Int32(0)).select(
            fx.Int32(0x00400000), reciprocal_biased << fx.Int32(23)
        )
        reciprocal_bits = is_special.select(fx.Int32(0x7FC00000), reciprocal_bits)
        return scale_biased, reciprocal_bits.bitcast(fx.Float32)

    def _canonical_nan(packed_i32, e8m0):
        """Write a NaN-scale block as the canonical FP8 NaN ``0x7f``.

        Every NaN a block can produce lives in a block whose scale is the E8M0
        NaN byte (any NaN or Inf input). The gfx950 converters emit NaN as
        ``0xff``; ``to_mx`` and the CUDA kernels emit ``0x7f``.
        """
        return (e8m0 == fx.Int32(_E8M0_NAN)).select(fx.Int32(_FP8_NAN_X4), packed_i32)

    def quantize_pack_chunk_to_i32_rceil(chunk_f32, pos_scale, e8m0):
        """RCEIL mode: quantize 4 f32 → 4 FP8 E4M3FN packed into one i32.

        Lowers to two ``v_cvt_scalef32_pk_fp8_f32`` instructions (gfx950 fused
        scale + convert with built-in saturate-to-NaN). RCEIL guarantees
        ``amax/scale ≤ F8_MAX`` so no NaN can fire; no explicit clamp needed.

        Args:
            chunk_f32: 4-element ``fx.Vector`` of f32.
            pos_scale: ``fx.Float32`` = ``2^scale_unb`` (the divisor; the
                instruction computes ``input / pos_scale``).
            e8m0: the block's E8M0 byte as ``fx.Int32``.

        Returns:
            ``fx.Int32`` with bytes ``[qv0, qv1, qv2, qv3]`` (low to high).
        """
        v2i16 = T.vec(2, T.i16)
        scale = pos_scale.ir_value()
        r = fx.Vector.from_elements([fx.Int16(0), fx.Int16(0)], fx.Int16)
        r = _scaled_cvt_pk_fp8_f32(
            res=v2i16,
            old_vdst=r.ir_value(),
            src0=chunk_f32[0].ir_value(),
            src1=chunk_f32[1].ir_value(),
            scale=scale,
            dst_lo_hi_sel=False,
        )
        r = _scaled_cvt_pk_fp8_f32(
            res=v2i16,
            old_vdst=r,
            src0=chunk_f32[2].ir_value(),
            src1=chunk_f32[3].ir_value(),
            scale=scale,
            dst_lo_hi_sel=True,
        )
        return _canonical_nan(fx.Vector(r).bitcast(fx.Int32)[0], e8m0)

    def quantize_pack_chunk_to_i32_floor(chunk_f32, inv_scale, e8m0):
        """FLOOR mode: quantize 4 f32 → 4 FP8 E4M3FN packed into one i32.

        FLOOR's ``amax/scale ∈ [256, 512)`` lets the (448, 512) tail overflow
        the FP8 range, so we clamp explicitly with ``±F8_MAX`` before two
        ``v_cvt_pk_fp8_f32`` cvts. ``inv_scale`` comes from
        :func:`floor_scale_and_inv_scale`.

        Returns:
            ``fx.Int32`` with bytes ``[qv0, qv1, qv2, qv3]`` (low to high).
        """
        lo = fx.Float32(-F8_MAX)
        hi = fx.Float32(F8_MAX)
        qv = [(chunk_f32[i] * inv_scale).maximumf(lo).minimumf(hi) for i in range(VEC)]
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32, src_a=qv[0], src_b=qv[1], old=fx.Int32(0), word_sel=False
        )
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32, src_a=qv[2], src_b=qv[3], old=out, word_sel=True
        )
        return _canonical_nan(fx.Int32(out), e8m0)
