# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for CuTeDSL quantization kernels."""

import importlib.util

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers import arith as _dsl_arith
from cutlass.cutlass_dsl import T, dsl_user_op

# Runtime package detection
_CUTEDSL_RUNTIME_PACKAGES = {
    "cuda.bindings.driver": "cuda-python",
    "cutlass": "nvidia-cutlass-dsl",
    "cutlass.cute": "nvidia-cutlass-dsl",
    "tvm_ffi": "apache-tvm-ffi",
}


def _missing_cutedsl_runtime_packages() -> list[str]:
    """Check which CuTeDSL runtime packages are missing.

    Returns:
        List of missing package names
    """
    missing = []
    for module_name, package_name in _CUTEDSL_RUNTIME_PACKAGES.items():
        if (
            importlib.util.find_spec(module_name) is None
            and package_name not in missing
        ):
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    """Check if all CuTeDSL runtime packages are available.

    Returns:
        True if all required packages are installed
    """
    return len(_missing_cutedsl_runtime_packages()) == 0


# FP8 constants
F8_MAX = cutlass.Float32(448.0)
INV_F8_MAX = cutlass.Float32(1.0 / 448.0)


# PTX inline assembly for RCEIL conversion on Blackwell
@dsl_user_op
def _cvt_rp_satfinite_ue8m0x2_f32(
    a: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint16:
    """PTX inline assembly for RCEIL conversion.

    Uses inline PTX on Blackwell-family targets because CuTeDSL does not
    currently lower this conversion to `cvt.rp.satfinite.ue8m0x2.f32` on its own.
    """
    return cutlass.Uint16(
        llvm.inline_asm(
            T.i16(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            "=h,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# Shared scale computation methods
@cute.jit
def compute_amax(vals_block: cute.Tensor):
    """Compute absolute maximum of a block of values.

    Args:
        vals_block: Tensor of values to compute amax from

    Returns:
        The absolute maximum value as Float32
    """
    vals_vec = vals_block.load()
    abs_vec = cute.where(vals_vec < 0, -vals_vec, vals_vec)
    return cutlass.Float32(
        abs_vec.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    )


@cute.jit
def compute_scale_rceil(
    amax: cutlass.Float32,
    IS_BLACKWELL_VALUE: cutlass.Constexpr[bool],
):
    """Compute scale using RCEIL (round-up) mode.

    Args:
        amax: Absolute maximum value
        IS_BLACKWELL_VALUE: Boolean indicating if Blackwell architecture

    Returns:
        Tuple of (scale_biased, inv_scale)
    """
    descale = amax * INV_F8_MAX
    if cutlass.const_expr(IS_BLACKWELL_VALUE):
        scale_biased = cutlass.Int32(_cvt_rp_satfinite_ue8m0x2_f32(descale))
        inv_scale = cutlass.Float32(1.0)
        if scale_biased == 0xFF:
            inv_scale = cutlass.Float32(0.0)
        elif scale_biased == 0:
            inv_scale = cute.exp2(cutlass.Float32(126.0))
        else:
            inv_scale = cute.exp2(cutlass.Float32(127 - scale_biased))
        return scale_biased, inv_scale

    bits = _dsl_arith.bitcast(descale.ir_value(), _dsl_arith.T.i32())
    exponent = (bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
    mantissa = bits & cutlass.Int32(0x7FFFFF)
    if exponent == 0xFF:
        if mantissa != 0:
            scale_biased = cutlass.Int32(0xFF)
        else:
            scale_biased = cutlass.Int32(0xFE)
    else:
        if mantissa > 0:
            if exponent != 0xFE:
                if exponent == 0:
                    if mantissa > 0x400000:
                        exponent += 1
                else:
                    exponent += 1
        scale_biased = exponent

    inv_scale = cutlass.Float32(1.0)
    if scale_biased == 0xFF:
        inv_scale = cutlass.Float32(0.0)
    elif scale_biased == 0:
        inv_scale = cutlass.Float32(1.0)
    else:
        inv_scale = cute.exp2(cutlass.Float32(127 - scale_biased))
    return scale_biased, inv_scale


@cute.jit
def compute_scale_floor(amax: cutlass.Float32):
    """Compute scale using FLOOR mode.

    Args:
        amax: Absolute maximum value

    Returns:
        Tuple of (scale_biased, inv_scale)
    """
    bits = _dsl_arith.bitcast(amax.ir_value(), _dsl_arith.T.i32())
    exp_i = ((bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)) - cutlass.Int32(127)
    scale_exp_unbiased = exp_i - cutlass.Int32(8)
    if scale_exp_unbiased < -127:
        scale_exp_unbiased = cutlass.Int32(-127)
    if scale_exp_unbiased > 128:
        scale_exp_unbiased = cutlass.Int32(128)
    inv_scale = cute.exp2(cutlass.Float32(-scale_exp_unbiased))
    scale_biased = scale_exp_unbiased + 127
    return scale_biased, inv_scale


@cute.jit
def compute_scale_from_amax(
    amax: cutlass.Float32,
    USE_RCEIL: cutlass.Constexpr[bool],
    IS_BLACKWELL: cutlass.Constexpr[bool],
):
    """Compute scale from absolute maximum using specified mode.

    Args:
        amax: Absolute maximum value
        USE_RCEIL: Constexpr boolean for scaling mode
        IS_BLACKWELL: Boolean indicating if Blackwell architecture

    Returns:
        Tuple of (scale_biased, inv_scale)
    """
    scale_biased = cutlass.Int32(0)
    inv_scale = cutlass.Float32(1.0)
    if amax > 0:
        if cutlass.const_expr(USE_RCEIL):
            scale_biased, inv_scale = compute_scale_rceil(amax, IS_BLACKWELL)
        else:
            scale_biased, inv_scale = compute_scale_floor(amax)
    return scale_biased, inv_scale
