# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for CuTeDSL quantization kernels."""

import importlib.util

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
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError):
            # ModuleNotFoundError: parent module doesn't exist (e.g., 'cuda' on CPU)
            # ValueError: can occur with malformed module names
            spec = None

        if spec is None and package_name not in missing:
            missing.append(package_name)
    return missing


def _cutedsl_runtime_available() -> bool:
    """Check if all CuTeDSL runtime packages are available.

    Returns:
        True if all required packages are installed
    """
    return len(_missing_cutedsl_runtime_packages()) == 0


if _cutedsl_runtime_available():
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith, llvm, nvvm, vector
    from cutlass.cutlass_dsl import T, dsl_user_op

    # FP8 constants
    INV_F8_MAX = cutlass.Float32(1.0 / 448.0)

    @dsl_user_op
    def view_as(x, dtype, *, loc=None, ip=None):
        """Bitcast one scalar to another scalar of equal width."""
        assert type(x).width == dtype.width
        # Use signed IR types even for unsigned CUTLASS types as this is what
        # bitcast wants.
        dst_type = (
            T.i(dtype.width)
            if ir.IntegerType.isinstance(dtype.mlir_type)
            else dtype.mlir_type
        )
        return dtype(
            arith.bitcast(
                dst_type,
                x.ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def unpack(x, dtype, *, loc=None, ip=None):
        """Unpack an integer carrier into a tuple of scalar values."""
        x = cute.typing.as_numeric(x)
        carrier_dtype = type(x)
        assert ir.IntegerType.isinstance(carrier_dtype.mlir_type)
        assert carrier_dtype.width % dtype.width == 0

        num_lanes = carrier_dtype.width // dtype.width
        # Use integer vector lanes because vector<N x FP8> can crash the
        # compiler: https://github.com/NVIDIA/cutlass/issues/3342
        lanes = llvm.bitcast(
            T.vector(num_lanes, T.i(dtype.width)),
            x.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        return tuple(
            view_as(
                cute.typing.as_numeric(
                    vector.extract(
                        lanes,
                        dynamic_position=[],
                        static_position=[i],
                        loc=loc,
                        ip=ip,
                    )
                ),
                dtype,
                loc=loc,
                ip=ip,
            )
            for i in range(num_lanes)
        )

    @dsl_user_op
    def pack(*values, carrier=None, loc=None, ip=None):
        """Pack same-typed scalar values into an integer carrier."""
        assert len(values) > 0

        lane_dtype = type(values[0])
        assert all(type(value) is lane_dtype for value in values)
        # Use integer vector lanes because vector<N x FP8> can crash the
        # compiler: https://github.com/NVIDIA/cutlass/issues/3342
        lane_type = T.i(lane_dtype.width)
        lanes = vector.from_elements(
            T.vector(len(values), lane_type),
            tuple(
                arith.bitcast(
                    lane_type,
                    value.ir_value(loc=loc, ip=ip),
                    loc=loc,
                    ip=ip,
                )
                for value in values
            ),
            loc=loc,
            ip=ip,
        )

        packed_width = len(values) * lane_dtype.width
        packed = llvm.bitcast(T.i(packed_width), lanes, loc=loc, ip=ip)
        if carrier is None:
            return cute.typing.as_numeric(packed)
        else:
            assert ir.IntegerType.isinstance(carrier.mlir_type)
            assert carrier.width == packed_width
            return carrier(packed)

    @dsl_user_op
    def fmax_nan_f32(a, b, *, loc=None, ip=None):
        """Pairwise f32 maximum with NaN propagation."""

        # CUTLASS 4.6 exposes this directly as
        # `cute.arch.fmax(a, b, nan=True)`. Remove this low-level NVVM wrapper
        # once TorchAO's minimum supported nvidia-cutlass-dsl is at least 4.6.
        return cutlass.Float32(
            nvvm.fmax(
                T.f32(),
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                nan=True,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _cvt_f32_to_ue8m0(
        x: cutlass.Float32,
        *,
        rounding_mode,
        loc=None,
        ip=None,
    ) -> cutlass.Float8E8M0FNU:
        """Convert x to a single E8M0 value without saturation."""
        packed = nvvm.cvt_packfloat_f32(
            T.i32(),
            cutlass.Float32(0.0).ir_value(loc=loc, ip=ip),
            x.ir_value(loc=loc, ip=ip),
            cutlass.Int32(0).ir_value(loc=loc, ip=ip),
            nvvm.CVTPackFloatKind.UE8M0x2,
            rnd=rounding_mode,
            sat=nvvm.SaturationModeKind.NONE,
            loc=loc,
            ip=ip,
        )
        return unpack(packed, cutlass.Float8E8M0FNU, loc=loc, ip=ip)[0]

    @dsl_user_op
    def _cvt_ue8m0_to_f32(
        x: cutlass.Float8E8M0FNU,
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Float32:
        """Convert a single E8M0 value to f32 through the supported BF16 path."""
        x_e8m0x2 = pack(x, cutlass.Float8E8M0FNU(0), loc=loc, ip=ip)
        x_u32 = llvm.zext(T.i32(), x_e8m0x2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        bf16x2_bits = nvvm.cvt_packfloat(
            T.i32(),
            x_u32,
            cutlass.Int32(0).ir_value(loc=loc, ip=ip),
            nvvm.CVTPackFloatKind.UE8M0x2,
            nvvm.CVTPackFloatKind.BF16x2,
            rnd=nvvm.RoundingModeKind.RN,
            sat=nvvm.SaturationModeKind.NONE,
            loc=loc,
            ip=ip,
        )
        low_bf16 = unpack(bf16x2_bits, cutlass.BFloat16, loc=loc, ip=ip)[0]
        return low_bf16.to(cutlass.Float32)

    @cute.jit
    def _reciprocal_scale(scale_e8m0: cutlass.Float8E8M0FNU):
        scale_biased = view_as(scale_e8m0, cutlass.Uint8)
        reciprocal_biased = cutlass.Uint8(254) - scale_biased
        return _cvt_ue8m0_to_f32(view_as(reciprocal_biased, cutlass.Float8E8M0FNU))

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
        abs_vec = cute.absf(vals_vec)
        return cutlass.Float32(
            abs_vec.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        )

    @cute.jit
    def compute_scale_rceil(amax: cutlass.Float32):
        """Compute scale using the Blackwell E8M0 round-up conversion.

        Args:
            amax: Absolute maximum value

        Returns:
            Tuple of (scale_biased, inv_scale)
        """
        descale = amax * INV_F8_MAX
        scale_e8m0 = _cvt_f32_to_ue8m0(descale, rounding_mode=nvvm.RoundingModeKind.RP)
        return view_as(scale_e8m0, cutlass.Uint8), _reciprocal_scale(scale_e8m0)

    @cute.jit
    def compute_scale_floor(amax: cutlass.Float32):
        """Compute scale using FLOOR mode.

        Args:
            amax: Absolute maximum value

        Returns:
            Tuple of (scale_biased, inv_scale)
        """
        descale = amax * cutlass.Float32(1.0 / 256.0)
        scale_e8m0 = _cvt_f32_to_ue8m0(descale, rounding_mode=nvvm.RoundingModeKind.RZ)
        return view_as(scale_e8m0, cutlass.Uint8), _reciprocal_scale(scale_e8m0)

    @cute.jit
    def compute_scale_from_amax(
        amax: cutlass.Float32,
        USE_RCEIL: cutlass.Constexpr[bool],
    ):
        """Compute scale from absolute maximum using specified mode.

        Args:
            amax: Absolute maximum value
            USE_RCEIL: Constexpr boolean for scaling mode (True for RCEIL, False for FLOOR)

        Returns:
            Tuple of (scale_biased, inv_scale)
        """
        if cutlass.const_expr(USE_RCEIL):
            return compute_scale_rceil(amax)
        return compute_scale_floor(amax)

    @cute.jit
    def validate_group_sizes(offs: cute.Tensor):
        # Only first thread validates to avoid redundant work
        num_groups = offs.shape[0]

        # Validate first group (from 0 to offs[0])
        if num_groups > 0:
            first_group_size = offs[0]
            cute.testing.assert_(
                first_group_size % 128 == 0,
                "Group sizes must be multiples of 128",
            )

        # Validate subsequent groups
        for i in range(1, num_groups):
            prev_end = offs[i - 1]
            curr_end = offs[i]
            group_size = curr_end - prev_end
            cute.testing.assert_(
                group_size % 128 == 0,
                "Group sizes must be multiples of 128",
            )
