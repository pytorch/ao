# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP4 (E2M1, block 32, E8M0) + Random Hadamard Transform CuTeDSL cast.

A max-bandwidth streaming quantize kernel: each thread owns one full 32-element
block (= 16 packed E2M1x2 output bytes = one wide 128-bit store + one E8M0 scale
byte), reads it with forced 128-bit loads, applies the optional register-resident
FWHT(32) + sign transform (``fwht.fwht32_sign``), computes amax + the E8M0 scale,
packs the 16 bytes, and writes them with a single 128-bit store; heavy per-thread
ILP hides the load latency and the e2m1/e8m0 conversions.

* the E2M1 output is two codes per byte -> half-width ``[M, K // 2]`` ``uint8``;
* ``scaling_mode`` ``"floor"`` or ``"rceil"`` (E8M0 via ``compute_scale_byte_fp4``);
  no explicit ``+-6`` clamp is needed -- ``cvt.rn.satfinite`` already saturates;
* the biased E8M0 scale byte is written either in the cuBLAS-blocked padded
  layout (``is_swizzled_scales=True``) or as a plain ``(M, K // 32)`` tensor.

The op is gated behind a Blackwell (SM 10.x) GPU, CUDA >= 12.8, and the CuTeDSL
runtime packages (see ``cutedsl/__init__.py``).
"""

import functools
from typing import Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    _cvt_rn_satfinite_e2m1x2_f32,
    compute_amax,
    compute_scale_byte_fp4,
)
from .fwht import fwht32_sign

# ============================================================================
# Max-bandwidth streaming quantize kernel
# ============================================================================
# The fused MXFP4 (+/- RHT) quantize kernel (no smem, no TMA). Streaming design:
# each thread owns ONE full 32-block (= 32 input
# elems = 16 output bytes = one wide 128-bit ``STG`` + one E8M0 scale byte),
# reads it with forced 128-bit ``LDG`` (``CopyUniversalOp`` num_bits_per_copy=128
# + ``cute.assume`` on the offset -- a plain ``iterator + dynamic_offset``
# silently degrades to scalar loads), runs the FWHT32+sign thread-locally, and
# uses heavy per-thread ILP. No explicit ``+-6`` clamps (``cvt.rn.satfinite``
# already saturates, bit-exact). 2D ``(row, group-in-row)`` map keeps
# ``(row, kb)`` division-free for the swizzled scale offset.
#
# IMPORTANT: compiled with CONCRETE ``from_dlpack`` tensors, not
# ``make_fake_tensor`` -- the symbolic AOT path mis-lowers the single-byte scale
# store. Cached on (dtype, scaling_mode, swizzled, threads, ilp).
#
# NOTE: the RHT path is FWHT32-compute-bound; the plain (no-RHT) path reaches
# the HBM roofline.
_MAXBW_JIT_CACHE: dict = {}


@functools.cache
def _get_maxbw_launch():
    """Define and return the (uncompiled) ``@cute.jit`` max-bandwidth launcher.

    Gated behind the CuTeDSL runtime (cutlass imports + kernel definition live
    inside this function). Reuses the module-level ``compute_amax`` /
    ``compute_scale_byte_fp4`` / ``_cvt_rn_satfinite_e2m1x2_f32`` / ``fwht32_sign``.
    """
    import cutlass
    import cutlass.cute as cute
    import cutlass.cute.nvgpu as nv

    F4_MAX = cutlass.Float32(6.0)

    @cute.kernel
    def _maxbw_kernel(
        gx: cute.Tensor,  # (M*K,) input flat (bf16 or fp32)
        gq: cute.Tensor,  # (M*K // 2,) uint8 flat (packed E2M1x2)
        gscale: cute.Tensor,  # scale buffer flat uint8 (E8M0 bytes)
        gsign: cute.Tensor,  # (32,) int32 sign vector (RHT only)
        M: cutlass.Int32,
        K: cutlass.Int32,
        GPR: cutlass.Int32,  # groups per row = K // 32
        pad_cols: cutlass.Int32,  # padded scale cols = ceil(K // 32, 4) * 4
        ILP: cutlass.Constexpr[int],
        APPLY_RHT: cutlass.Constexpr[bool],
        SWIZZLED: cutlass.Constexpr[bool],
        USE_RCEIL: cutlass.Constexpr[bool],
        DROP_CLAMP: cutlass.Constexpr[bool],
        LDWIDTH: cutlass.Constexpr[int],  # elems per 128-bit load (8 bf16, 4 fp32)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy_init, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        gid = bidx * bdim + tidx
        nthreads_x = cute.arch.grid_dim()[0] * bdim

        in_dt = gx.element_type
        ld = cute.make_copy_atom(nv.CopyUniversalOp(), in_dt, num_bits_per_copy=128)
        st = cute.make_copy_atom(
            nv.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=128
        )

        # number of 128-bit loads to cover 32 input elems
        NLD = 32 // LDWIDTH

        # Load the length-32 sign vector once (RHT only).
        sign_reg = cute.make_rmem_tensor((32,), cutlass.Float32)
        if cutlass.const_expr(APPLY_RHT):
            for j in cutlass.range_constexpr(32):
                sign_reg[j] = cutlass.Float32(gsign[j])

        # 2D grid-stride over (row, gcol). grid.y indexes rows.
        row = bidy_init
        while row < M:
            base = gid
            while base < GPR:
                # ---- issue ALL loads first (ILP) into a flat (ILP, 32) buffer ----
                fragbuf = cute.make_rmem_tensor((ILP * 32,), in_dt)
                for jj in cutlass.range_constexpr(ILP):
                    gc = base + jj * nthreads_x
                    if gc < GPR:
                        off = cute.assume(row * K + gc * cutlass.Int32(32), divby=32)
                        for w in cutlass.range_constexpr(NLD):
                            s = cute.make_tensor(
                                gx.iterator + off + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            f = cute.make_tensor(
                                fragbuf.iterator + jj * 32 + w * LDWIDTH,
                                cute.make_layout((LDWIDTH,), stride=(1,)),
                            )
                            cute.copy(ld, s, f)

                # ---- consume: FWHT/scale/pack + wide store per 32-block ----
                for jj in cutlass.range_constexpr(ILP):
                    gc = base + jj * nthreads_x
                    if gc < GPR:
                        vals = cute.make_rmem_tensor((32,), cutlass.Float32)
                        for i in cutlass.range_constexpr(32):
                            vals[i] = cutlass.Float32(fragbuf[jj * 32 + i])
                        blk = cute.make_tensor(
                            vals.iterator, cute.make_layout((32,), stride=(1,))
                        )
                        if cutlass.const_expr(APPLY_RHT):
                            fwht32_sign(blk, sign_reg)
                        amax = compute_amax(blk)
                        scale_biased, inv = compute_scale_byte_fp4(amax, USE_RCEIL)
                        kb = gc  # one block per group
                        scale_byte = cutlass.Uint8(scale_biased & cutlass.Int32(0xFF))
                        # scale store
                        if cutlass.const_expr(SWIZZLED):
                            r128 = row // cutlass.Int32(128)
                            r32 = row % cutlass.Int32(32)
                            r32_4 = (row // cutlass.Int32(32)) % cutlass.Int32(4)
                            kb4 = kb // cutlass.Int32(4)
                            kbm = kb % cutlass.Int32(4)
                            soff = (
                                r128 * cutlass.Int32(128) * pad_cols
                                + kb4 * cutlass.Int32(512)
                                + r32 * cutlass.Int32(16)
                                + r32_4 * cutlass.Int32(4)
                                + kbm
                            )
                            gscale[soff] = scale_byte
                        else:
                            gscale[row * GPR + kb] = scale_byte
                        packed = cute.make_rmem_tensor((16,), cutlass.Uint8)
                        for p in cutlass.range_constexpr(16):
                            lo = blk[2 * p] * inv
                            hi = blk[2 * p + 1] * inv
                            if cutlass.const_expr(not DROP_CLAMP and not USE_RCEIL):
                                if lo > F4_MAX:
                                    lo = F4_MAX
                                if lo < -F4_MAX:
                                    lo = -F4_MAX
                                if hi > F4_MAX:
                                    hi = F4_MAX
                                if hi < -F4_MAX:
                                    hi = -F4_MAX
                            packed[p] = _cvt_rn_satfinite_e2m1x2_f32(hi, lo)
                        offq = cute.assume(
                            row * (K // cutlass.Int32(2)) + gc * cutlass.Int32(16),
                            divby=16,
                        )
                        d = cute.make_tensor(
                            gq.iterator + offq, cute.make_layout((16,), stride=(1,))
                        )
                        cute.copy(st, packed, d)

                base = base + nthreads_x * cutlass.Int32(ILP)
            row = row + cute.arch.grid_dim()[1]

    @cute.jit
    def _maxbw_launch(
        gx,
        gq,
        gscale,
        gsign,
        M,
        K,
        GPR,
        pad_cols,
        threads,
        ncta_x,
        ncta_y,
        stream,
        ILP: cutlass.Constexpr[int],
        APPLY_RHT: cutlass.Constexpr[bool],
        SWIZZLED: cutlass.Constexpr[bool],
        USE_RCEIL: cutlass.Constexpr[bool],
        DROP_CLAMP: cutlass.Constexpr[bool],
        LDWIDTH: cutlass.Constexpr[int],
    ):
        _maxbw_kernel(
            gx,
            gq,
            gscale,
            gsign,
            M,
            K,
            GPR,
            pad_cols,
            ILP,
            APPLY_RHT,
            SWIZZLED,
            USE_RCEIL,
            DROP_CLAMP,
            LDWIDTH,
        ).launch(
            grid=(ncta_x, ncta_y, 1),
            block=(threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    return _maxbw_launch


def _maxbw_quantize(
    x: torch.Tensor,
    sign_vector,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
    threads: int = 128,
    ilp: int = 2,
    drop_clamp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host wrapper: launch the max-bandwidth MXFP4 (+/- RHT) quantize kernel.

    Output contract matches the warp-specialized TMA path exactly: ``qdata`` is
    row-major ``(M, K // 2)`` uint8 (packed E2M1x2); ``scales`` is
    ``float8_e8m0fnu`` in the cuBLAS-blocked padded layout (``is_swizzled_scales``)
    or a plain ``(M, K // 32)`` tensor.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    assert x.is_cuda and x.dim() == 2 and x.is_contiguous()
    assert x.dtype in (torch.float32, torch.bfloat16)
    M, K = x.shape
    assert M % 128 == 0 and K % 128 == 0
    k_blocks = K // 32
    apply_rht = sign_vector is not None and len(sign_vector) > 0
    if apply_rht:
        assert len(sign_vector) == 32
    use_rceil = scaling_mode.lower() == "rceil"

    q_data = torch.empty_strided(
        (M, K // 2), (K // 2, 1), device=x.device, dtype=torch.uint8
    )
    padded_scale_rows = ceil_div(M, 128) * 128
    padded_scale_cols = ceil_div(k_blocks, 4) * 4
    if is_swizzled_scales:
        scales_u8 = torch.empty(
            (padded_scale_rows * padded_scale_cols,),
            device=x.device,
            dtype=torch.uint8,
        )
    else:
        scales_u8 = torch.empty((M, k_blocks), device=x.device, dtype=torch.uint8)

    sign_src = sign_vector if apply_rht else [0] * 32
    sign_dev = torch.tensor(
        [int(s) for s in sign_src], device=x.device, dtype=torch.int32
    )

    GPR = K // 32
    nthreads_needed = (GPR + ilp - 1) // ilp
    ncta_x = (nthreads_needed + threads - 1) // threads
    ncta_y = M
    ldwidth = 4 if x.dtype == torch.float32 else 8

    launch = _get_maxbw_launch()
    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))

    def _args():
        return (
            from_dlpack(x.view(-1), assumed_align=16),
            from_dlpack(q_data.view(-1), assumed_align=16),
            from_dlpack(scales_u8.view(-1), assumed_align=16),
            from_dlpack(sign_dev, assumed_align=16),
            cutlass.Int32(M),
            cutlass.Int32(K),
            cutlass.Int32(GPR),
            cutlass.Int32(padded_scale_cols),
            threads,
            ncta_x,
            ncta_y,
            stream,
        )

    key = (
        str(x.dtype),
        apply_rht,
        is_swizzled_scales,
        use_rceil,
        drop_clamp,
        threads,
        ilp,
    )
    compiled = _MAXBW_JIT_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            launch,
            *_args(),
            ilp,
            apply_rht,
            is_swizzled_scales,
            use_rceil,
            drop_clamp,
            ldwidth,
        )
        _MAXBW_JIT_CACHE[key] = compiled
    compiled(*_args())

    scales = scales_u8.view(torch.float8_e8m0fnu)
    scales = (
        scales.view(padded_scale_rows, padded_scale_cols)
        if is_swizzled_scales
        else scales.view(M, k_blocks)
    )
    return q_data, scales


def _mxfp4_rht_quantize_cutedsl_impl(
    x: torch.Tensor,
    sign_vector: list[int],
    block_size: int = 32,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host wrapper: launch the fused MXFP4 + RHT CuTeDSL quantize kernel.

    Args:
        x: 2D contiguous bf16/fp32 input ``(M, K)`` with ``M % 128 == 0`` and
            ``K % 128 == 0``.
        sign_vector: length-32 list of ``{-1, +1}`` for the RHT sign multiply.
        block_size: only 32 is supported.
        scaling_mode: ``"floor"`` or ``"rceil"``.
        is_swizzled_scales: write scales in the cuBLAS-blocked padded layout
            (``True``) or a plain ``(M, K // 32)`` tensor (``False``).

    Returns:
        ``(qdata, scales)`` where ``qdata`` is row-major ``(M, K // 2)`` uint8
        (packed E2M1x2) and ``scales`` is ``float8_e8m0fnu`` in the requested
        layout.
    """
    assert x.is_cuda, "Input tensor must be CUDA"
    assert x.dim() == 2, "Input tensor must be 2D"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert block_size == 32, "Only block_size=32 is supported"
    scaling_mode = scaling_mode.lower()
    assert scaling_mode in ("floor", "rceil"), (
        f"Unsupported scaling_mode={scaling_mode!r}; expected 'floor' or 'rceil'"
    )
    assert len(sign_vector) == 32, "sign_vector must have length 32"

    M, K = x.shape
    assert K % 32 == 0, "K must be divisible by 32"
    assert M % 128 == 0, "M must be divisible by 128"
    assert K % 128 == 0, "K must be divisible by 128"

    return _maxbw_quantize(
        x,
        sign_vector,
        scaling_mode=scaling_mode,
        is_swizzled_scales=is_swizzled_scales,
    )


@torch.library.custom_op("torchao::mxfp4_rht_quantize_cutedsl", mutates_args=())
def mxfp4_rht_quantize_cutedsl(
    x: torch.Tensor,
    sign_vector: list[int],
    block_size: int = 32,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _mxfp4_rht_quantize_cutedsl_impl(
        x,
        sign_vector,
        block_size=block_size,
        scaling_mode=scaling_mode,
        is_swizzled_scales=is_swizzled_scales,
    )


@mxfp4_rht_quantize_cutedsl.register_fake
def _(
    x: torch.Tensor,
    sign_vector: list[int],
    block_size: int = 32,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    q = torch.empty_strided(
        (m, k // 2), (k // 2, 1), device=x.device, dtype=torch.uint8
    )  # row-major pinned
    kb = k // block_size
    if is_swizzled_scales:
        scales = x.new_empty(
            (ceil_div(m, 128) * 128, ceil_div(kb, 4) * 4),
            dtype=torch.float8_e8m0fnu,
        )
    else:
        scales = x.new_empty((m, kb), dtype=torch.float8_e8m0fnu)
    return q, scales


def mxfp4_rht_quantize_cutedsl_2d(
    x: torch.Tensor,
    sign_vector,
    block_size: int = 32,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gated public wrapper for the fused MXFP4 + RHT CuTeDSL quantize op.

    Raises ``NotImplementedError`` (with the missing-runtime detail) when the
    CuTeDSL runtime / SM 10.x / CUDA >= 12.8 requirements are not met.
    """
    from torchao.prototype.mx_formats.cutedsl import (
        _mxfp4_rht_cutedsl_kernels_available,
    )

    if not _mxfp4_rht_cutedsl_kernels_available:
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            _missing_cutedsl_runtime_packages,
        )

        raise NotImplementedError(
            "mxfp4_rht_quantize_cutedsl requires CUDA SM10.x, CUDA>=12.8, and: "
            f"{_missing_cutedsl_runtime_packages() or 'nvidia-cutlass-dsl'}"
        )
    return mxfp4_rht_quantize_cutedsl(
        x, list(sign_vector), block_size, scaling_mode, is_swizzled_scales
    )
