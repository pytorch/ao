# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for FlyDSL MXFP8 quantization kernels.

FlyDSL is an MLIR-based DSL for authoring GPU kernels on AMD hardware
(CDNA / RDNA / unified architectures). These utilities mirror cute_utils.py
(the CuteDSL counterpart) but target AMD GPUs via the ROCm/HIP stack.

Initial scope: FLOOR-mode E8M0 scale derivation (software, arch-agnostic).
RCEIL mode is intentionally deferred to a follow-up — it has no direct
hardware intrinsic on AMD equivalent to NVIDIA's ``cvt.rp.satfinite.ue8m0x2.f32``,
so it requires a careful software round-up implementation matching the
PTX behavior bit-for-bit.
"""

import importlib.util

# Runtime package detection (mirror of cute_utils._missing_cutedsl_runtime_packages).
# The flydsl package itself is the only hard requirement; torch/cuda are checked
# elsewhere in the dispatcher.
_FLYDSL_RUNTIME_PACKAGES = {
    "flydsl": "flydsl",
    "flydsl.compiler": "flydsl",
    "flydsl.expr": "flydsl",
}


def _missing_flydsl_runtime_packages() -> list[str]:
    """Check which FlyDSL runtime packages are missing.

    Returns:
        List of missing package names (deduplicated).
    """
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
    """Return True if all FlyDSL runtime packages are importable."""
    return len(_missing_flydsl_runtime_packages()) == 0


# FP8 E4M3FN constants (matches cute_utils.F8_MAX / INV_F8_MAX).
F8_MAX = 448.0
INV_F8_MAX = 1.0 / 448.0

# E8M0 exponent bias (per OCP MX format spec section 6.3).
E8M0_EXPONENT_BIAS = 127

# AMD wave size on CDNA / gfx9xx targets. Used to size half-wave reductions
# (each MXFP8 quant block is 32 elements = half of a 64-lane wave).
AMD_WAVE_SIZE = 64
HALF_WAVE = AMD_WAVE_SIZE // 2  # 32 — matches MXFP8 block size


# The kernel-side helpers below are only valid inside an @flyc.kernel body
# and require FlyDSL to be importable. We guard the import to keep this module
# importable in environments without FlyDSL installed (mirroring cute_utils.py).
if _flydsl_runtime_available():
    import flydsl.expr as fx
    from flydsl.expr import arith, rocdl
    from flydsl.expr.typing import T

    def half_wave_max_f32(x):
        """Reduce f32 ``x`` to the per-half-wave maximum via ``shuffle_xor``.

        Lanes 0..31 reduce together; lanes 32..63 reduce together. Each half
        ends up with the same broadcast value (the half's max).

        AMD ``ds_swizzle_b32`` / ``v_permlanex16`` semantics: when ``width``
        is smaller than the wave size, the high lane bits above ``log2(width)``
        are preserved, so the two halves reduce independently.
        """
        w = x
        for sh in (16, 8, 4, 2, 1):
            peer = w.shuffle_xor(fx.Int32(sh), fx.Int32(HALF_WAVE))
            w = w.maximumf(peer)
        return w

    def compute_scale_floor_f32(amax):
        """FLOOR-mode E8M0 scale derivation (software, arch-agnostic).

        Matches torchao.prototype.mx_formats.mx_tensor.to_mx with
        ScaleCalculationMode.FLOOR for FP8 E4M3 (max_pos=448).

        Reference:
            torchao/csrc/cuda/mx_kernels/mxfp8_quantize.cuh L520-L538
            torchao.prototype.moe_training.kernels.mxfp8.cute_utils.compute_scale_floor

        Algorithm (for amax > 0):
            bits        = bitcast(amax, i32)
            exp_i       = ((bits >> 23) & 0xFF) - 127       # unbiased exponent of amax
            scale_unb   = clamp(exp_i - 8, -127, 128)       # subtract 8 because 448 ≈ 2^8.8
            inv_scale   = 2 ^ (-scale_unb)
            scale_biased = scale_unb + 127                  # E8M0 uint8 value

        For amax == 0, returns (scale_biased=0, inv_scale=1.0).

        Args:
            amax: absolute maximum value of a 32-element block, as f32 ArithValue.

        Returns:
            (scale_biased, inv_scale): i32 ArithValue (storable as uint8) and
            f32 ArithValue.
        """
        # Default outputs for the amax==0 path.
        scale_biased = fx.Int32(0)
        inv_scale = fx.Float32(1.0)

        if amax > fx.Float32(0.0):
            bits = amax.bitcast(T.i32())
            exp_i = ((bits >> fx.Int32(23)) & fx.Int32(0xFF)) - fx.Int32(127)
            scale_unb = exp_i - fx.Int32(8)
            # Clamp to E8M0 representable unbiased range [-127, 128].
            if scale_unb < fx.Int32(-127):
                scale_unb = fx.Int32(-127)
            if scale_unb > fx.Int32(128):
                scale_unb = fx.Int32(128)
            # inv_scale = 2 ^ (-scale_unb). Cast int → f32 for math.exp2.
            neg_unb_f = arith.sitofp(fx.Int32(0) - scale_unb, T.f32())
            inv_scale = fx.math.exp2(neg_unb_f)
            scale_biased = scale_unb + fx.Int32(E8M0_EXPONENT_BIAS)

        return scale_biased, inv_scale

    def pack_4_f32_to_i32_fp8(qv0, qv1, qv2, qv3):
        """Pack 4 quantized f32 values into an i32 holding 4 packed FP8 E4M3FN bytes.

        Uses two ``v_cvt_pk_fp8_f32`` instructions:
        - First call (word_sel=0): pack qv0,qv1 into low half of result i32.
        - Second call (word_sel=1): pack qv2,qv3 into high half of the same i32.

        Available on CDNA3+ (gfx940, gfx942, gfx950) and RDNA4 (gfx1200+).

        Args:
            qv0..qv3: f32 ArithValues, already scaled and clamped to ±F8_MAX.

        Returns:
            i32 ArithValue containing the 4 packed FP8 bytes [qv0|qv1|qv2|qv3].
        """
        out = fx.Int32(0)
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32(),
            src_a=qv0,
            src_b=qv1,
            old=out,
            word_sel=False,  # low word
        )
        out = rocdl.cvt_pk_fp8_f32(
            res=T.i32(),
            src_a=qv2,
            src_b=qv3,
            old=out,
            word_sel=True,  # high word
        )
        return out
