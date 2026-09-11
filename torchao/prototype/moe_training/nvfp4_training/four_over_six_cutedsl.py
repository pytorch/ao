# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL NVFP4 four-over-six quantize kernel (SM100+).

Fast path for ``four_over_six_quantize``: an op-for-op reimplementation of the
pure-PyTorch reference body, producing bitwise-identical codes and scales.
The load-bearing arithmetic is pinned with inline PTX so no compiler lowering
choice can change a rounding:

* every division is a real ``div.rn.f32`` (the ``(block_amax / 6) * S_enc``
  association, ``S_enc``/``S_dec``, the encode reciprocals, and the
  dequant-error denominator) — never a reciprocal multiply;
* the E4M3 scale cast is ``cvt.rn.satfinite.e4m3x2.f32`` and its exact decode
  ``cvt.rn.f16x2.e4m3x2``;
* the FP4 cast is ``cvt.rn.satfinite.e2m1x2.f32`` (NaN -> +6) and the error
  path dequantizes with the exact ``cvt.rn.f16x2.e2m1x2``;
* block amaxes use NaN-dropping ``max.f32`` (``fmaxf``: an all-NaN group
  yields amax 0) and the 448 caps use NaN-dropping ``min.f32`` (``fminf``);
* the per-group dequant error is accumulated strictly sequentially in element
  order 0..15 with scalar FP32 round-to-nearest adds (no reduction trees), and
  16x16 tiles reduce their 16 row errors with the exact width-16
  shuffle-down tree ``(((e0+e8)+(e4+e12))+((e2+e10)+(e6+e14))) +
  (((e1+e9)+(e5+e13))+((e3+e11)+(e7+e15)))`` broadcast from the segment base.

Tiling: one (128, 64) input tile per CTA, 128 threads, one tile row per
thread, so lanes 0-15 / 16-31 of each warp hold 16 consecutive rows — the
16-lane segments the 16x16 (2D) mode reduces over. The tile is loaded with
one TMA G2S copy, packed FP4 codes leave via TMA S2G (rows past R are
clipped by the TMA bounds check; zero-filled OOB rows compute garbage that
is never stored), and the four scale bytes per tile row leave as one u32.

Eligibility (the dispatch gate in ``four_over_six_quantize``): SM100+, CUDA
bf16 or fp32 input, contiguous, C % 64 == 0 (the tile width; also makes the
u32 scale store aligned). Ineligible calls fall through to the pure-PyTorch
body. The CuTe DSL runtime packages are assumed to be installed wherever the
gate passes.
"""

import functools
from typing import Tuple

import torch

from torchao.utils import ceil_div, is_sm_at_least_100

TILE_ROWS = 128
TILE_COLS = 64  # 4 1x16 groups per tile row; the C % 64 dispatch gate
_FP32_MAX = torch.finfo(torch.float32).max
# shfl.sync c-operand for width-16 segments: segmask 0x10 | clamp 0x1f, the
# same ((32 - width) << 8) | 0x1f encoding __shfl_down_sync(..., 16) uses.
_SHFL_WIDTH16 = 0x101F


def _cutedsl_quantize_eligible(x: torch.Tensor) -> bool:
    """True iff ``four_over_six_quantize`` may dispatch ``x`` to the CuTe DSL kernel.

    Shape/dtype gates come first so FakeTensor tracing takes the same branch
    as eager. Ineligible inputs silently use the pure-PyTorch body.
    """
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.shape[0] > 0
        and x.shape[0] <= 65535 * TILE_ROWS  # grid.y limit
        and x.shape[1] > 0
        and x.shape[1] % TILE_COLS == 0
        and x.is_contiguous()
        and is_sm_at_least_100()
    )


@functools.cache
def _compile_four_over_six_quantize_cutedsl(
    input_dtype_name: str,
    block_16x16: bool,
    err_mode: str,
    e4m3_scale_bound: int,
    row_scaled: bool,
    device_index: int,
):
    """Compile one (dtype, block, err_mode, bound, row_scaled) kernel variant.

    ``device_index`` is a cache key only (per-device cache separation, as in
    the sibling ``_compile_fused_kernel``); the body never reads it.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.cutlass_dsl import T, dsl_user_op

    from ._cutedsl_kernels_impl import (
        _abs_f32,
        _cvt_rn_satfinite_e2m1x2_f32_pack4,
        _min_f32,
    )

    if input_dtype_name == "torch.float32":
        INPUT_CUTLASS_DTYPE = cutlass.Float32
    elif input_dtype_name == "torch.bfloat16":
        INPUT_CUTLASS_DTYPE = cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported input dtype for CuTe DSL four_over_six_quantize: {input_dtype_name}"
        )

    BLOCK_16X16 = block_16x16
    USE_MSE = err_mode == "mse"
    ROW_SCALED = row_scaled
    # bound * 6: the S_enc numerator AND the slow-path error denominator
    # (1536.0 for bound 256, 2688.0 for bound 448) — both exact in FP32.
    BOUND_TIMES_FP4MAX = float(e4m3_scale_bound) * 6.0

    GROUPS_PER_ROW = TILE_COLS // 16
    CODE_TILE_BYTES = TILE_COLS // 2
    THREADS_PER_BLOCK = TILE_ROWS  # one tile row per thread
    input_elem_bytes = INPUT_CUTLASS_DTYPE.width // 8
    TILE_COPY_BYTES = TILE_ROWS * TILE_COLS * input_elem_bytes

    @dsl_user_op
    def _div_rn_f32(
        a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
    ) -> cutlass.Float32:
        """Correctly-rounded FP32 division (``__fdiv_rn``), pinned with inline PTX.

        The bitwise contract needs real divisions — a reciprocal multiply
        double-rounds and flips candidate picks near error ties — so we never
        rely on how the DSL lowers ``/``.
        """
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
    def _fmax_f32(
        a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
    ) -> cutlass.Float32:
        # Plain max.f32 = fmaxf: NaN inputs are silently dropped, so an
        # all-NaN group yields amax 0. This is NOT the NaN-propagating
        # _max_f32 the RHT kernels use.
        return cutlass.Float32(
            llvm.inline_asm(
                T.f32(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                "max.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _mul_rn_f32(
        a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
    ) -> cutlass.Float32:
        """``__fmul_rn`` pinned with inline PTX so it can never fuse into an FMA.

        Used for the MSE ``diff * diff`` feeding the sequential error adds —
        the one mul+add pair a contraction would double-round.
        """
        return cutlass.Float32(
            llvm.inline_asm(
                T.f32(),
                [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _e4m3x2_rn_satfinite_with_decode(
        c6: cutlass.Float32, c4: cutlass.Float32, *, loc=None, ip=None
    ):
        """E4M3-cast both candidate scales and decode them back, all exactly.

        ``cvt.rn.satfinite.e4m3x2.f32`` packs e4m3(c4) in the high byte and
        e4m3(c6) in the low byte (matching ``__nv_cvt_float_to_fp8`` RN
        satfinite); ``cvt.rn.f16x2.e4m3x2`` + ``cvt.f32.f16`` recover the
        exact FP32 value of each scale byte. Returns (packed_u32, f6, f4).
        """
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
            [c6.ir_value(loc=loc, ip=ip), c4.ir_value(loc=loc, ip=ip)],
            (
                "{\n"
                ".reg .b16 sp, s6, s4;\n"
                ".reg .b32 dp;\n"
                "cvt.rn.satfinite.e4m3x2.f32 sp, $4, $3;\n"
                "cvt.u32.u16 $0, sp;\n"
                "cvt.rn.f16x2.e4m3x2 dp, sp;\n"
                "mov.b32 {s6, s4}, dp;\n"
                "cvt.f32.f16 $1, s6;\n"
                "cvt.f32.f16 $2, s4;\n"
                "}"
            ),
            "=r,=f,=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        pair = cutlass.Uint32(llvm.extractvalue(T.i32(), res, [0]))
        f6 = cutlass.Float32(llvm.extractvalue(T.f32(), res, [1]))
        f4 = cutlass.Float32(llvm.extractvalue(T.f32(), res, [2]))
        return pair, f6, f4

    @dsl_user_op
    def _dequant_e2m1x8_f32(w: cutlass.Uint32, *, loc=None, ip=None):
        """Exact dequant of 8 packed FP4 codes to 8 FP32s in element order.

        ``cvt.rn.f16x2.e2m1x2`` per byte (low nibble -> low half) then exact
        ``cvt.f32.f16`` — the slow-path error dequant.
        """
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * 8),
            [w.ir_value(loc=loc, ip=ip)],
            (
                "{\n"
                ".reg .b8 q0, q1, q2, q3;\n"
                ".reg .b32 r0, r1, r2, r3;\n"
                ".reg .b16 ha, hb;\n"
                "mov.b32 {q0, q1, q2, q3}, $8;\n"
                "cvt.rn.f16x2.e2m1x2 r0, q0;\n"
                "mov.b32 {ha, hb}, r0;\n"
                "cvt.f32.f16 $0, ha;\n"
                "cvt.f32.f16 $1, hb;\n"
                "cvt.rn.f16x2.e2m1x2 r1, q1;\n"
                "mov.b32 {ha, hb}, r1;\n"
                "cvt.f32.f16 $2, ha;\n"
                "cvt.f32.f16 $3, hb;\n"
                "cvt.rn.f16x2.e2m1x2 r2, q2;\n"
                "mov.b32 {ha, hb}, r2;\n"
                "cvt.f32.f16 $4, ha;\n"
                "cvt.f32.f16 $5, hb;\n"
                "cvt.rn.f16x2.e2m1x2 r3, q3;\n"
                "mov.b32 {ha, hb}, r3;\n"
                "cvt.f32.f16 $6, ha;\n"
                "cvt.f32.f16 $7, hb;\n"
                "}"
            ),
            "=f,=f,=f,=f,=f,=f,=f,=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
        return tuple(
            cutlass.Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(8)
        )

    @cute.struct
    class SharedStorage:
        tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
        in_smem: cute.struct.Align[
            cute.struct.MemRange[INPUT_CUTLASS_DTYPE, TILE_ROWS * TILE_COLS], 128
        ]
        out_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, TILE_ROWS * CODE_TILE_BYTES], 128
        ]

    # The helpers below are plain (undecorated) python functions traced inline
    # from the kernel body — the _cutedsl_kernels_impl convention — so their
    # loops unroll at trace time (the sequential error adds stay 16 dependent
    # scalar FP32 RN adds) and python tuples stay indexable.

    def _load_group(sIN, m_rel, k_base):
        """Load one 1x16 group from SMEM, upconverting to FP32 exactly once."""
        raw = cute.make_rmem_tensor((16,), INPUT_CUTLASS_DTYPE)
        cute.autovec_copy(
            cute.make_tensor(
                (sIN.iterator + (m_rel * TILE_COLS + k_base)).align(16),
                cute.make_layout(16),
            ),
            raw,
        )
        vals = cute.make_rmem_tensor((16,), cutlass.Float32)
        for i in range(16):
            vals[i] = cutlass.Float32(raw[i])
        return vals

    def _group_amax(vals):
        """fmaxf chain over |vals| from 0.0 (NaN-dropping; order-insensitive)."""
        amax = cutlass.Float32(0.0)
        for i in range(16):
            amax = _fmax_f32(amax, _abs_f32(vals[i]))
        return amax

    def _tile_reduce_max16(v):
        """16-lane-segment max: shuffle-down offsets 8/4/2/1, broadcast from base."""
        for delta in (8, 4, 2, 1):
            v = _fmax_f32(
                v, cute.arch.shuffle_sync_down(v, delta, mask_and_clamp=_SHFL_WIDTH16)
            )
        return cute.arch.shuffle_sync(v, 0, mask_and_clamp=_SHFL_WIDTH16)

    def _tile_reduce_sum16(v):
        """The pinned error tree: value(l) += value(l+8), +=(l+4), +=(l+2),
        +=(l+1) (clamped shuffle-down, NOT butterfly), broadcast from the
        segment base. Summation shape
        (((e0+e8)+(e4+e12))+((e2+e10)+(e6+e14))) + (((e1+e9)+(e5+e13))+((e3+e11)+(e7+e15)))."""
        for delta in (8, 4, 2, 1):
            v = v + cute.arch.shuffle_sync_down(v, delta, mask_and_clamp=_SHFL_WIDTH16)
        return cute.arch.shuffle_sync(v, 0, mask_and_clamp=_SHFL_WIDTH16)

    def _scale_pair(block_amax, s_enc, s_dec):
        """compute_scale_pair: base = (block_amax / 6) * S_enc — a real div.rn
        THEN mul.rn — the 1.5x map-to-4 expansion, both capped at the full
        448 E4M3 range, and the exact-division encode multipliers."""
        base = _div_rn_f32(block_amax, cutlass.Float32(6.0)) * s_enc
        c6 = _min_f32(base, cutlass.Float32(448.0))
        c4 = _min_f32(base * cutlass.Float32(1.5), cutlass.Float32(448.0))
        pair, f6, f4 = _e4m3x2_rn_satfinite_with_decode(c6, c4)
        inv6 = _min_f32(
            _div_rn_f32(cutlass.Float32(1.0), f6 * s_dec),
            cutlass.Float32(_FP32_MAX),
        )
        inv4 = _min_f32(
            _div_rn_f32(cutlass.Float32(1.0), f4 * s_dec),
            cutlass.Float32(_FP32_MAX),
        )
        b6 = pair & 0xFF
        b4 = (pair >> 8) & 0xFF
        return b6, b4, f6, f4, inv6, inv4

    def _encode_and_error(vals, inv, f_dec, gamax):
        """One candidate: FP4-encode 16 values and accumulate the dequant
        error strictly sequentially in element order 0..15 (16 dependent
        FP32 RN adds — never a reduction tree). Returns (w0, w1, err)."""
        q = cute.make_rmem_tensor((16,), cutlass.Float32)
        for i in range(16):
            q[i] = vals[i] * inv
        # Even element -> low nibble: byte j = cvt(hi=q[2j+1], lo=q[2j]).
        w0 = _cvt_rn_satfinite_e2m1x2_f32_pack4(
            q[0], q[2], q[4], q[6], q[1], q[3], q[5], q[7]
        )
        w1 = _cvt_rn_satfinite_e2m1x2_f32_pack4(
            q[8], q[10], q[12], q[14], q[9], q[11], q[13], q[15]
        )
        err = cutlass.Float32(0.0)
        for half in range(2):
            dq = _dequant_e2m1x8_f32(w0 if half == 0 else w1)
            for i in range(8):
                # val = div.rn(mul.rn(mul.rn(dequant, (f32)scale_byte),
                #              global_amax), 6*bound) in the input domain.
                val = _div_rn_f32(
                    (dq[i] * f_dec) * gamax,
                    cutlass.Float32(BOUND_TIMES_FP4MAX),
                )
                diff = val - vals[half * 8 + i]
                if USE_MSE:
                    err = err + _mul_rn_f32(diff, diff)
                else:
                    err = err + _abs_f32(diff)
        return w0, w1, err

    class FourOverSixQuantizeKernel:
        @cute.kernel
        def kernel(
            self,
            tma_atom_in: cute.CopyAtom,
            tma_tensor_in: cute.Tensor,
            tma_atom_out: cute.CopyAtom,
            tma_tensor_out: cute.Tensor,
            scales_u32: cute.Tensor,
            amax_f32: cute.Tensor,
            R: cutlass.Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            bidx, bidy, _ = cute.arch.block_idx()

            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            tma_mbar_ptr = storage.tma_mbar_ptr.data_ptr()

            smem_layout_in = cute.make_layout(
                (TILE_ROWS, TILE_COLS), stride=(TILE_COLS, 1)
            )
            smem_layout_out = cute.make_layout(
                (TILE_ROWS, CODE_TILE_BYTES), stride=(CODE_TILE_BYTES, 1)
            )
            sIN = storage.in_smem.get_tensor(smem_layout_in)
            sOUT = storage.out_smem.get_tensor(smem_layout_out)
            # u64 view of the code tile: one 8-byte store per 1x16 group.
            sOUT_u64 = cute.recast_tensor(sOUT, cutlass.Uint64)

            if tidx == 0:
                cpasync.prefetch_descriptor(tma_atom_in)
                cpasync.prefetch_descriptor(tma_atom_out)
                cute.arch.mbarrier_init(tma_mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            m_tile = cutlass.Int64(bidy)
            c_tile = cutlass.Int64(bidx)

            gIN_tile = cute.local_tile(
                tma_tensor_in, (TILE_ROWS, TILE_COLS), (m_tile, c_tile)
            )
            if warp_idx == 0:
                cta_layout = cute.make_layout((1,))
                tINs, tINg = cpasync.tma_partition(
                    tma_atom_in,
                    0,
                    cta_layout,
                    cute.group_modes(sIN, 0, 2),
                    cute.group_modes(gIN_tile, 0, 2),
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        tma_mbar_ptr, TILE_COPY_BYTES
                    )
                cute.copy(tma_atom_in, tINg[None], tINs[None], tma_bar_ptr=tma_mbar_ptr)

            m_rel = tidx
            m = m_tile * TILE_ROWS + m_rel

            # Global scale chain, once per thread (per row when row-scaled):
            # S_enc = fminf(bound*6 / global_amax, FLT_MAX), falling back to
            # 1.0 when global_amax == 0 or S_enc == 0; S_dec = 1 / S_enc.
            gamax = cutlass.Float32(0.0)
            if cutlass.const_expr(ROW_SCALED):
                if m < R:
                    gamax = amax_f32[m]
            else:
                gamax = amax_f32[0]
            s_enc_raw = _min_f32(
                _div_rn_f32(cutlass.Float32(BOUND_TIMES_FP4MAX), gamax),
                cutlass.Float32(_FP32_MAX),
            )
            s_enc = s_enc_raw
            if gamax == 0.0:
                s_enc = cutlass.Float32(1.0)
            if s_enc_raw == 0.0:
                s_enc = cutlass.Float32(1.0)
            s_dec = _div_rn_f32(cutlass.Float32(1.0), s_enc)

            cute.arch.mbarrier_wait(tma_mbar_ptr, 0)

            scale_word = cutlass.Uint32(0)
            for gc in cutlass.range_constexpr(GROUPS_PER_ROW):
                vals = _load_group(sIN, m_rel, gc * 16)
                block_amax = _group_amax(vals)
                if cutlass.const_expr(BLOCK_16X16):
                    block_amax = _tile_reduce_max16(block_amax)
                b6, b4, f6, f4, inv6, inv4 = _scale_pair(block_amax, s_enc, s_dec)
                w6_0, w6_1, err6 = _encode_and_error(vals, inv6, f6, gamax)
                w4_0, w4_1, err4 = _encode_and_error(vals, inv4, f4, gamax)
                if cutlass.const_expr(BLOCK_16X16):
                    err6 = _tile_reduce_sum16(err6)
                    err4 = _tile_reduce_sum16(err4)
                # Strict < with map4 on the left: ties and NaN errors pick map6.
                w0 = w6_0
                w1 = w6_1
                sbyte = b6
                if err4 < err6:
                    w0 = w4_0
                    w1 = w4_1
                    sbyte = b4
                sOUT_u64[m_rel, gc] = cutlass.Uint64(w0) | (cutlass.Uint64(w1) << 32)
                scale_word = scale_word | (sbyte << (8 * gc))

            if m < R:
                scales_u32[m, c_tile] = scale_word

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp_idx == 0:
                gOUT_tile = cute.local_tile(
                    tma_tensor_out, (TILE_ROWS, CODE_TILE_BYTES), (m_tile, c_tile)
                )
                cta_layout = cute.make_layout((1,))
                tOUTs, tOUTg = cpasync.tma_partition(
                    tma_atom_out,
                    0,
                    cta_layout,
                    cute.group_modes(sOUT, 0, 2),
                    cute.group_modes(gOUT_tile, 0, 2),
                )
                cute.copy(tma_atom_out, tOUTs[None], tOUTg[None])

        @cute.jit
        def __call__(
            self,
            inp_rc: cute.Tensor,
            out_codes: cute.Tensor,
            scales_u32: cute.Tensor,
            amax_f32: cute.Tensor,
            R: cutlass.Int32,
            r_tiles: cutlass.Int32,
            c_tiles: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            smem_layout_in = cute.make_layout(
                (TILE_ROWS, TILE_COLS), stride=(TILE_COLS, 1)
            )
            smem_layout_out = cute.make_layout(
                (TILE_ROWS, CODE_TILE_BYTES), stride=(CODE_TILE_BYTES, 1)
            )
            g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
                g2s_op,
                inp_rc,
                smem_layout_in,
                (TILE_ROWS, TILE_COLS),
            )
            tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_codes,
                smem_layout_out,
                (TILE_ROWS, CODE_TILE_BYTES),
            )
            self.kernel(
                tma_atom_in,
                tma_tensor_in,
                tma_atom_out,
                tma_tensor_out,
                scales_u32,
                amax_f32,
                R,
            ).launch(
                grid=(c_tiles, r_tiles, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    kernel = FourOverSixQuantizeKernel()

    r = cute.sym_int()
    c = cute.sym_int(divisibility=64)
    ch = cute.sym_int(divisibility=32)
    cs = cute.sym_int()
    fake_inp = make_fake_tensor(
        INPUT_CUTLASS_DTYPE,
        (r, c),
        stride=(cute.sym_int(), cute.sym_int()),
    )
    fake_out = make_fake_tensor(
        cutlass.Uint8,
        (r, ch),
        stride=(cute.sym_int(), cute.sym_int()),
    )
    fake_scales = make_fake_tensor(
        cutlass.Uint32,
        (r, cs),
        stride=(cute.sym_int(), cute.sym_int()),
    )
    fake_amax = make_fake_tensor(
        cutlass.Float32,
        (cute.sym_int(),),
        stride=(cute.sym_int(),),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        inp_rc=fake_inp,
        out_codes=fake_out,
        scales_u32=fake_scales,
        amax_f32=fake_amax,
        R=0,
        r_tiles=1,
        c_tiles=1,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op(
    "torchao::four_over_six_quantize_cutedsl", mutates_args=(), device_types="cuda"
)
def four_over_six_quantize_cutedsl(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    block: str,
    err_mode: str,
    e4m3_scale_bound: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 four-over-six quantize (CuTe DSL, SM100+).

    Bitwise-identical to the pure-PyTorch ``four_over_six_quantize`` body for
    every (block, err_mode, e4m3_scale_bound, per-tensor/row-scaled) mode,
    with one documented exception: NaN inputs take NaN-dropping block
    amaxes and encode to +6 FP4 codes here, while torch's ``amax``
    propagates NaN into the reference's block scales.

    Args:
        x:                (R, C) bfloat16 or float32, contiguous, C % 64 == 0.
        global_amax:      scalar FP32 amax, or (R,) per-row amax (1x16 only).
        block:            "1x16" or "16x16".
        err_mode:         "mae" or "mse".
        e4m3_scale_bound: 256 or 448.

    Returns:
        (codes, scales): (R, C//2) uint8 packed FP4 codes (low nibble = even
        element) and (R, C//16) float8_e4m3fn block scales.
    """
    if not is_sm_at_least_100():
        raise NotImplementedError("four_over_six_quantize_cutedsl requires SM100+")
    if x.ndim != 2:
        raise ValueError("x must be 2-D")
    # Direct torch.ops callers bypass the dispatch gate and the python
    # wrapper, so mirror their checks here as errors rather than TMA/launch
    # failures or silently wrong scales.
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"x must be bfloat16 or float32, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if block not in ("1x16", "16x16"):
        raise ValueError(f"block must be '1x16' or '16x16', got {block!r}")
    if err_mode not in ("mae", "mse"):
        raise ValueError(f"err_mode must be 'mae' or 'mse', got {err_mode!r}")
    if e4m3_scale_bound not in (256, 448):
        raise ValueError(f"e4m3_scale_bound must be 256 or 448, got {e4m3_scale_bound}")
    rows, cols = x.shape
    if rows == 0 or cols == 0:
        raise ValueError(f"x must be non-empty, got shape {tuple(x.shape)}")
    if global_amax.numel() not in (1, rows):
        raise ValueError(
            f"global_amax must have 1 or {rows} elements, got {global_amax.numel()}"
        )
    if rows > 65535 * TILE_ROWS:
        raise ValueError(
            f"R must be at most {65535 * TILE_ROWS} (grid.y limit), got {rows}"
        )
    if cols % TILE_COLS != 0:
        raise ValueError(
            f"four_over_six_quantize_cutedsl requires C % {TILE_COLS} == 0, got {cols}"
        )
    if block == "16x16" and rows % 16:
        raise ValueError(f"16x16 blocks need rows divisible by 16, got {rows}")
    row_scaled = global_amax.dim() == 1 and global_amax.numel() == rows

    codes = torch.empty_strided(
        (rows, cols // 2), (cols // 2, 1), device=x.device, dtype=torch.uint8
    )
    scales_u8 = torch.empty_strided(
        (rows, cols // 16), (cols // 16, 1), device=x.device, dtype=torch.uint8
    )
    scales_u32 = scales_u8.view(torch.uint32)
    # The kernel reads the amax through a raw device pointer, so a CPU scalar
    # amax (fine for the pure-PyTorch body) must be moved to x's device here.
    amax_f32 = (
        global_amax.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    )

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    import cuda.bindings.driver as cuda

    with torch.cuda.device(x.device):
        compiled = _compile_four_over_six_quantize_cutedsl(
            str(x.dtype),
            block == "16x16",
            err_mode,
            int(e4m3_scale_bound),
            row_scaled,
            device_index,
        )
        stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
        compiled(
            x,
            codes,
            scales_u32,
            amax_f32,
            int(rows),
            int(ceil_div(rows, TILE_ROWS)),
            int(cols // TILE_COLS),
            stream,
        )
    return codes, scales_u8.view(torch.float8_e4m3fn)


@four_over_six_quantize_cutedsl.register_fake
def _(x, global_amax, block, err_mode, e4m3_scale_bound):
    rows, cols = x.shape
    codes = x.new_empty((rows, cols // 2), dtype=torch.uint8)
    scales = x.new_empty((rows, cols // 16), dtype=torch.float8_e4m3fn)
    return codes, scales
