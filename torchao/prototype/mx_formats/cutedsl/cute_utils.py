# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for the MXFP4 + RHT CuTeDSL quantize kernels."""

import importlib.util

import torch

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
    from cutlass._mlir.dialects import llvm
    from cutlass.base_dsl._mlir_helpers import arith as _dsl_arith
    from cutlass.cutlass_dsl import T, dsl_user_op

    # FP4 (E2M1) constants. F4_E2M1_MAX == 6.0.
    INV_F4_E2M1_MAX = cutlass.Float32(1.0 / 6.0)

    # FP4 E8M0 scale constants. NOTE: these are the FP4 values, NOT the FP8
    # ones -- `F4_E2M1_MAX_POW2 == 2` (the FP8 helper uses 8), and the RCEIL
    # descale divisor is `F4_E2M1_MAX == 6.0` (the FP8 helper uses 448).
    F4_E2M1_MAX_POW2 = 2  # log2 of the largest power-of-two <= F4_E2M1_MAX (6.0)
    E8M0_EXPONENT_BIAS = 127
    E8M0_EXPONENT_NAN_VAL = 255

    @dsl_user_op
    def _cvt_rn_satfinite_e2m1x2_f32(
        hi: cutlass.Float32,
        lo: cutlass.Float32,
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Uint8:
        """PTX inline assembly that packs two f32 values into one E2M1x2 byte.

        Uses inline PTX on Blackwell-family targets because CuTeDSL does not
        currently lower the float32 -> E2M1 pair conversion to
        ``cvt.rn.satfinite.e2m1x2.f32`` on its own.

        The PTX result of ``cvt.rn.satfinite.e2m1x2.f32 d, a, b`` is a ``.b8``
        value, but inline-asm output registers must be at least 16-bit and
        ptxas rejects a 16-bit register directly as the ``cvt`` destination. So
        (mirroring cutlass's ``cvt.rn.f16x2.e2m1x2`` wrappers in
        ``cute/arch/nvvm_wrappers.py``) we ``cvt`` into a ``.reg .b8`` and
        assemble it into a ``.b16`` output via ``mov.b16 $0, {d, z}`` with a
        zero high byte. The low byte is then masked out and returned.

        Convention (validated bit-exactly by ``test_mxfp4_rht_cutedsl``): the
        second PTX operand ``b`` (``lo``) lands in the LOW nibble and the first
        operand ``a`` (``hi``) lands in the HIGH nibble, i.e. the returned byte
        is ``(e2m1(hi) << 4) | e2m1(lo)``.
        """
        packed = cutlass.Uint16(
            llvm.inline_asm(
                T.i16(),
                [
                    cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                ],
                "{\n\t"
                ".reg .b8 d, z, w;\n\t"
                ".reg .b16 zero16;\n\t"
                "mov.u16 zero16, 0;\n\t"
                "mov.b16 {z, w}, zero16;\n\t"
                "cvt.rn.satfinite.e2m1x2.f32 d, $1, $2;\n\t"
                "mov.b16 $0, {d, z};\n\t"
                "}",
                "=h,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )
        return cutlass.Uint8(packed & cutlass.Uint16(0xFF))

    @cute.kernel
    def _pack32_e2m1_kernel(
        gx: cute.Tensor,
        gout: cute.Tensor,
    ):
        """One-block, single-thread kernel: pack 32 f32 -> 16 E2M1x2 bytes.

        For ``p`` in ``0..15`` it emits
        ``_cvt_rn_satfinite_e2m1x2_f32(hi=x[2p+1], lo=x[2p])`` so that the even
        column ``2p`` lands in the low nibble and the odd column ``2p+1`` in the
        high nibble of output byte ``p``.
        """
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            for p in cutlass.range_constexpr(16):
                lo = cutlass.Float32(gx[2 * p])
                hi = cutlass.Float32(gx[2 * p + 1])
                gout[p] = _cvt_rn_satfinite_e2m1x2_f32(hi, lo)

    @cute.jit
    def _pack32_e2m1_launch(
        gx: cute.Tensor,
        gout: cute.Tensor,
        stream,
    ):
        _pack32_e2m1_kernel(gx, gout).launch(
            grid=(1, 1, 1),
            block=(32, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    def pack32_e2m1_to_bytes(x: torch.Tensor) -> torch.Tensor:
        """Pack a length-32 fp32 CUDA vector into 16 E2M1x2 bytes.

        Test/validation entry only -- the production kernel inlines the same
        ``_cvt_rn_satfinite_e2m1x2_f32`` logic. Even input columns go to the low
        nibble, odd columns to the high nibble (matching torchao's packed-fp4
        byte order).

        Args:
            x: 1D float32 CUDA tensor of length 32.

        Returns:
            A ``(16,)`` uint8 CUDA tensor.
        """
        import cuda.bindings.driver as cuda
        from cutlass.cute.runtime import from_dlpack

        assert x.is_cuda, "input must be a CUDA tensor"
        assert x.dtype == torch.float32, "input must be float32"
        assert x.numel() == 32, "input must have exactly 32 elements"
        x = x.contiguous()
        out = torch.empty((16,), device=x.device, dtype=torch.uint8)

        gx = from_dlpack(x)
        gout = from_dlpack(out)
        stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
        _pack32_e2m1_launch(gx, gout, stream)
        return out

    # ------------------------------------------------------------------
    # FP4 E8M0 block-scale helpers
    # ------------------------------------------------------------------

    @dsl_user_op
    def _cvt_rp_satfinite_ue8m0x2_f32(
        a: cutlass.Float32,
        *,
        loc=None,
        ip=None,
    ) -> cutlass.Uint16:
        """PTX inline assembly for RCEIL E8M0 conversion (Blackwell SM10.x).

        ``cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, descale`` packs two e8m0
        results into a uint16; the low byte holds the e8m0 of ``descale``.
        Hardware handles NaN -> 255, Inf -> 254, subnormals -> 0.
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

    @cute.jit
    def compute_amax(vals_block: cute.Tensor):
        """Compute the absolute maximum of a block of values as Float32."""
        vals_vec = vals_block.load()
        abs_vec = cute.where(vals_vec < 0, -vals_vec, vals_vec)
        return cutlass.Float32(
            abs_vec.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
        )

    @cute.jit
    def compute_scale_floor_fp4(amax: cutlass.Float32):
        """FP4 FLOOR-mode E8M0 biased scale byte.

        Mirrors torchao eager ``to_mx`` (FLOOR branch) for
        ``elem_dtype=torch.float4_e2m1fn_x2``:

            extracted_pow2     = ((bits(amax) >> 23) & 0xFF) - 127
            scale_unbiased     = extracted_pow2 - F4_E2M1_MAX_POW2    # -2, NOT -8
            scale_unbiased     = clamp(scale_unbiased, -127, 128)
            scale_biased(byte) = scale_unbiased + 127

        Returns the biased E8M0 byte as Int32 (caller stores low 8 bits).
        """
        bits = _dsl_arith.bitcast(amax.ir_value(), _dsl_arith.T.i32())
        exp_i = ((bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)) - cutlass.Int32(
            E8M0_EXPONENT_BIAS
        )
        scale_exp_unbiased = exp_i - cutlass.Int32(F4_E2M1_MAX_POW2)
        if scale_exp_unbiased < -E8M0_EXPONENT_BIAS:
            scale_exp_unbiased = cutlass.Int32(-E8M0_EXPONENT_BIAS)
        if scale_exp_unbiased > E8M0_EXPONENT_BIAS + 1:
            scale_exp_unbiased = cutlass.Int32(E8M0_EXPONENT_BIAS + 1)
        scale_biased = scale_exp_unbiased + E8M0_EXPONENT_BIAS
        return scale_biased

    @cute.jit
    def compute_scale_rceil_fp4(amax: cutlass.Float32):
        """FP4 RCEIL-mode E8M0 biased scale byte.

        Mirrors torchao eager ``_to_mx_rceil`` for
        ``elem_dtype=torch.float4_e2m1fn_x2`` (``max_pos == F4_E2M1_MAX == 6.0``):

            descale = amax / 6.0   (i.e. amax * INV_F4_E2M1_MAX, NOT 1/448)
            biased  = cvt.rp.satfinite.ue8m0x2.f32(0.0, descale)  # low byte

        Hardware saturates / handles NaN -> 255, Inf -> 254. Returns the biased
        E8M0 byte as Int32 (caller stores low 8 bits).
        """
        descale = amax * INV_F4_E2M1_MAX
        scale_biased = cutlass.Int32(_cvt_rp_satfinite_ue8m0x2_f32(descale))
        return scale_biased

    @cute.jit
    def compute_scale_byte_fp4(
        amax: cutlass.Float32,
        USE_RCEIL: cutlass.Constexpr[bool],
    ):
        """Dispatch to the FP4 FLOOR / RCEIL E8M0 biased scale byte.

        Matches eager ``to_mx``: a NaN ``amax`` maps to the E8M0 NaN byte
        (255). For FLOOR a non-NaN ``amax`` always uses the bit-extraction
        path (an all-zero block yields byte ``-2 + 127 = 125``, the same as
        eager, since ``floor(log2(0)) = -127`` clamps and ``0 -> extracted
        pow2 = -127``... handled by clamp). RCEIL uses the saturating PTX cvt,
        which itself maps subnormal/zero descale -> 0.
        """
        # NaN amax -> E8M0 NaN byte, matching eager to_mx.
        scale_biased = cutlass.Int32(E8M0_EXPONENT_NAN_VAL)
        if amax == amax:  # not NaN
            if cutlass.const_expr(USE_RCEIL):
                scale_biased = compute_scale_rceil_fp4(amax)
            else:
                scale_biased = compute_scale_floor_fp4(amax)
        return scale_biased

    @cute.kernel
    def _block_scale_e8m0_fp4_kernel(
        gx: cute.Tensor,
        gscale: cute.Tensor,
        num_blocks: cutlass.Int32,
        USE_RCEIL: cutlass.Constexpr[bool],
    ):
        """One thread per 32-element block: amax -> E8M0 biased scale byte.

        ``gx`` is a 2D ``(num_blocks, 32)`` float32 view; ``gscale`` is a 1D
        ``(num_blocks,)`` uint8 view (the flattened ``(N, K//32)`` scales).
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        block = bidx * bdim + tidx
        if block < num_blocks:
            vals = cute.make_rmem_tensor((32,), cutlass.Float32)
            for j in cutlass.range_constexpr(32):
                vals[j] = cutlass.Float32(gx[block, j])
            amax = compute_amax(vals)
            scale_biased = compute_scale_byte_fp4(amax, USE_RCEIL)
            gscale[block] = cutlass.Uint8(scale_biased & cutlass.Int32(0xFF))

    @cute.jit
    def _block_scale_e8m0_fp4_launch(
        gx: cute.Tensor,
        gscale: cute.Tensor,
        num_blocks: cutlass.Int32,
        stream,
        USE_RCEIL: cutlass.Constexpr[bool],
    ):
        threads = 128
        grid = (num_blocks + threads - 1) // threads
        _block_scale_e8m0_fp4_kernel(gx, gscale, num_blocks, USE_RCEIL).launch(
            grid=(grid, 1, 1),
            block=(threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    def compute_block_scale_e8m0_fp4(x: torch.Tensor, mode: str) -> torch.Tensor:
        """Compute per-32-block E8M0 biased scale bytes for FP4.

        Bit-exact (validated by ``test_mxfp4_rht_cutedsl``) against torchao
        eager ``to_mx(..., elem_dtype=torch.float4_e2m1fn_x2, block_size=32)``
        plain (unswizzled) scales, for FLOOR and RCEIL modes.

        Args:
            x: 2D CUDA tensor ``[N, K]`` (bf16 or fp32) with ``K % 32 == 0``.
            mode: ``"floor"`` or ``"rceil"``.

        Returns:
            ``(N, K // 32)`` ``torch.float8_e8m0fnu`` CUDA tensor of scale bytes.
        """
        import cuda.bindings.driver as cuda
        from cutlass.cute.runtime import from_dlpack

        assert x.is_cuda, "input must be a CUDA tensor"
        assert x.dim() == 2, "input must be 2D [N, K]"
        n, k = x.shape
        assert k % 32 == 0, "K must be divisible by 32"
        mode = mode.lower()
        assert mode in ("floor", "rceil"), f"unsupported scaling mode: {mode}"
        use_rceil = mode == "rceil"

        kb = k // 32
        num_blocks = n * kb
        # float32 [num_blocks, 32] view of the per-block elements.
        x_f32 = x.to(torch.float32).contiguous().reshape(num_blocks, 32)
        scale_u8 = torch.empty((num_blocks,), device=x.device, dtype=torch.uint8)

        gx = from_dlpack(x_f32)
        gscale = from_dlpack(scale_u8)
        stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
        _block_scale_e8m0_fp4_launch(
            gx, gscale, cutlass.Int32(num_blocks), stream, use_rceil
        )
        return scale_u8.view(torch.float8_e8m0fnu).reshape(n, kb)
