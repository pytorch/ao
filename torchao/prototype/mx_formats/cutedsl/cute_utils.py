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
    from cutlass.cutlass_dsl import T, dsl_user_op

    # FP4 (E2M1) constants. F4_E2M1_MAX == 6.0.
    INV_F4_E2M1_MAX = cutlass.Float32(1.0 / 6.0)

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
