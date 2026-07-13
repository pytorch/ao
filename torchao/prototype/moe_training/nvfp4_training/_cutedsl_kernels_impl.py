"""CuteDSL RHT + NVFP4 E2M1 columnwise/rowwise quantization kernels for SM100.

Private impl for the torchao:: ops. Imported lazily by the op wrappers so the top-level
``import cutlass`` only runs when a cutedsl op is actually called.

Two kernels, both built on the same single A load shared by two consumers (the MMA warp
does the columnwise RHT, a row warp group reads the same tile):
  - _Tcgen05RowColFused: quantizes col=RHT(A.t()) and row=A to NVFP4.
  - _Tcgen05RhtAmax: reduces col=max|RHT(A.t())|, row=max|A|.
"""

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import blackwell_helpers as sm100_utils
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from .hadamard_utils import get_rht_matrix

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
FP32_MAX = torch.finfo(torch.float32).max
FP8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny  # smallest normal E4M3 (0.015625)

HADAMARD_DIM = 16


# ---------------------------------------------------------------------------
# CuteDSL inline PTX ops
# ---------------------------------------------------------------------------


@dsl_user_op
def _cvt_rn_satfinite_e2m1x2_f32_pack4(
    lo0: cutlass.Float32,
    lo1: cutlass.Float32,
    lo2: cutlass.Float32,
    lo3: cutlass.Float32,
    hi0: cutlass.Float32,
    hi1: cutlass.Float32,
    hi2: cutlass.Float32,
    hi3: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Pack 4 (lo, hi) FP32 pairs into 4 FP4 bytes via inline PTX."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                lo0.ir_value(loc=loc, ip=ip),
                lo1.ir_value(loc=loc, ip=ip),
                lo2.ir_value(loc=loc, ip=ip),
                lo3.ir_value(loc=loc, ip=ip),
                hi0.ir_value(loc=loc, ip=ip),
                hi1.ir_value(loc=loc, ip=ip),
                hi2.ir_value(loc=loc, ip=ip),
                hi3.ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n"
                ".reg .b8 b0, b1, b2, b3;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;\n"
                "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;\n"
                "mov.b32 $0, {b0, b1, b2, b3};\n"
                "}"
            ),
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_rs_satfinite_e2m1x4_f32_pack4(
    lo0: cutlass.Float32,
    lo1: cutlass.Float32,
    lo2: cutlass.Float32,
    lo3: cutlass.Float32,
    hi0: cutlass.Float32,
    hi1: cutlass.Float32,
    hi2: cutlass.Float32,
    hi3: cutlass.Float32,
    rb0: cutlass.Uint32,
    rb1: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Stochastic-rounding analog of _cvt_rn_satfinite_e2m1x2_f32_pack4: same (lo,hi) arg order and
    same packed-FP4 output, but rounds with the hardware cvt.rs using random bits rb0/rb1 (one 32-bit
    word per 4-FP4 half). The {$6,$2,$5,$1}/{$8,$4,$7,$3} lane order reproduces the rn path's nibbles."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                lo0.ir_value(loc=loc, ip=ip),
                lo1.ir_value(loc=loc, ip=ip),
                lo2.ir_value(loc=loc, ip=ip),
                lo3.ir_value(loc=loc, ip=ip),
                hi0.ir_value(loc=loc, ip=ip),
                hi1.ir_value(loc=loc, ip=ip),
                hi2.ir_value(loc=loc, ip=ip),
                hi3.ir_value(loc=loc, ip=ip),
                rb0.ir_value(loc=loc, ip=ip),
                rb1.ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n"
                ".reg .b16 h0, h1;\n"
                "cvt.rs.satfinite.e2m1x4.f32 h0, {$6, $2, $5, $1}, $9;\n"
                "cvt.rs.satfinite.e2m1x4.f32 h1, {$8, $4, $7, $3}, $10;\n"
                "mov.b32 $0, {h0, h1};\n"
                "}"
            ),
            "=r,f,f,f,f,f,f,f,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _hash_u32(x: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    """murmur3 32-bit finalizer: a well-mixed (seed, counter) -> uniform u32 for stochastic rounding."""
    x = x ^ (x >> 16)
    x = x * cutlass.Uint32(0x85EBCA6B)
    x = x ^ (x >> 13)
    x = x * cutlass.Uint32(0xC2B2AE35)
    x = x ^ (x >> 16)
    return x


@dsl_user_op
def _min_f32(
    a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _rcp_approx_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Fast approximate reciprocal via PTX rcp.approx.f32.

    Less precise than 1.0/x but acceptable for FP4 quantization scale factors.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip)],
            "rcp.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _max_f32(
    a: cutlass.Float32, b: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    # max.NaN.f32 returns NaN if either operand is NaN, so the amax reductions propagate NaN
    # (matching triton_rht_amax). The non-NaN max.f32 variant would silently drop it.
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "max.NaN.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _min_xorsign_abs_f32(
    a: cutlass.Float32, limit: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    """Emit PTX min.xorsign.abs.f32 for symmetric clamp to +/-limit."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), limit.ir_value(loc=loc, ip=ip)],
            "min.xorsign.abs.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _atom_max_f32_nonneg(
    addr: cutlass.Pointer, val: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Float32:
    """Atomic max on global memory for non-negative float32.

    For non-negative floats, bit patterns are ordered the same as unsigned integers,
    so we can use atom.global.max.u32 on the reinterpreted bits.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [addr.llvm_ptr, val.ir_value(loc=loc, ip=ip)],
            (
                "{\n"
                ".reg .b32 v_bits, old_bits;\n"
                "mov.b32 v_bits, $2;\n"
                "atom.global.max.u32 old_bits, [$1], v_bits;\n"
                "mov.b32 $0, old_bits;\n"
                "}"
            ),
            "=f,l,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _abs_f32(a: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Absolute value of float32."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ---------------------------------------------------------------------------
# Hadamard matrix
# ---------------------------------------------------------------------------


DEFAULT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)


# ---------------------------------------------------------------------------
# Compilation and caching
# ---------------------------------------------------------------------------


# Device-scoped cache to avoid redundant per-call work in the hot path
@functools.lru_cache(maxsize=4)
def _get_num_sms(device_idx: int) -> int:
    return torch.cuda.get_device_properties(device_idx).multi_processor_count


# ---------------------------------------------------------------------------
# Fused row+col tcgen05 RHT+NVFP4 kernel (single A load, dual-consumer sA)
# ---------------------------------------------------------------------------
M_TILE = 128
N_TILE = 16
K = 16
U = 16  # adjacent col-groups per super-tile
KW = K * U  # wide A load K (= M-positions per super-tile)
MMA_TILER = (M_TILE, N_TILE, K)  # instruction atom stays 128x16x16

NUM_AB_STAGE = 2
NUM_ACC_STAGE = 1

# --- warp specialization (RHT mode, 14 warps): col is a cheap TMEM epilogue (MMA does the work) ---
N_COL_WARPS = 4
N_ROW_WARPS = 8
COL_WARP_END = N_COL_WARPS  # warps 0..3 = col
ROW_WARP_BEGIN = N_COL_WARPS  # 4
ROW_WARP_END = N_COL_WARPS + N_ROW_WARPS  # 12
MMA_WARP = ROW_WARP_END  # 12
TMA_WARP = MMA_WARP + 1  # 13
N_WARPS = TMA_WARP + 1  # 14
FUSED_TPB = 32 * N_WARPS  # 448
COL_THREADS = 32 * N_COL_WARPS  # 128
ROW_THREADS = 32 * N_ROW_WARPS  # 256  (== KW rows for U=16: 1 row/thread)

# --- warp specialization (weight mode, apply_rht=False, no MMA): col now does the heavy transposed
# SMEM read itself, so col and row get EQUAL warps (their per-tile outputs are equal-sized). Col's 8
# warps cover the 128 N-rows with 2 threads/row (each does half the U u-blocks). ---
COL_WARP_END_W = 8  # warps 0..7 = col (2 threads / N-row)
ROW_WARP_BEGIN_W = 8
ROW_WARP_END_W = 16  # warps 8..15 = row (1 thread / M-row, KW=256)
TMA_WARP_W = 16
FUSED_TPB_W = 32 * (TMA_WARP_W + 1)  # 544
COL_THREADS_W = 32 * COL_WARP_END_W  # 256
ROW_THREADS_W = 32 * (ROW_WARP_END_W - ROW_WARP_BEGIN_W)  # 256

TMEM_ALLOC_BAR = 1
TMEM_DEALLOC_BAR = 2
EPI_STORE_BAR = 3
ROW_STORE_BAR = 4
ROW_FP4_STAGES = 2

# Swizzled scale-factor layout (cutlass NVFP4): SF[r,c] -> [r//128, c//4, r%32, (r%128//32)*4 + c%4].
# Per super-tile, the SF tile has a 32x16 (=16B-wide) inner -> TMA-storable. Block inner = 32*16.
SF_BLK = 32 * 16  # 512 fp8 per (128-row x 4-col) swizzle block
SF_GCOL = U // 4  # 4  : M-groups (c//4) per col super-tile (16 SF cols)
SF_RBLK = KW // 128  # 2  : M-blocks (r//128) per row super-tile (256 M rows)
SF_RGRP = (M_TILE // 16) // 4  # 2  : N-groups (c//4) per row super-tile (8 SF cols)


def _pack16(q, sr: cutlass.Constexpr, rng_base):
    """Pack 16 scaled f32 -> (w0, w1) packed-FP4 u32. RTNE, or stochastic rounding (hardware
    cvt.rs) when sr, with four decorrelated random words derived from rng_base via _hash_u32."""
    if cutlass.const_expr(sr):
        rb0 = _hash_u32(rng_base)
        rb1 = _hash_u32(rng_base ^ cutlass.Uint32(0x9E3779B9))
        rb2 = _hash_u32(rng_base ^ cutlass.Uint32(0x7F4A7C15))
        rb3 = _hash_u32(rng_base ^ cutlass.Uint32(0xBB67AE85))
        w0 = _cvt_rs_satfinite_e2m1x4_f32_pack4(
            q[0], q[2], q[4], q[6], q[1], q[3], q[5], q[7], rb0, rb1
        )
        w1 = _cvt_rs_satfinite_e2m1x4_f32_pack4(
            q[8], q[10], q[12], q[14], q[9], q[11], q[13], q[15], rb2, rb3
        )
    else:
        w0 = _cvt_rn_satfinite_e2m1x2_f32_pack4(
            q[0], q[2], q[4], q[6], q[1], q[3], q[5], q[7]
        )
        w1 = _cvt_rn_satfinite_e2m1x2_f32_pack4(
            q[8], q[10], q[12], q[14], q[9], q[11], q[13], q[15]
        )
    return w0, w1


def _quant16(vals, enc_over_fp4max, dec, sr: cutlass.Constexpr = False, rng_base=None):
    """16-value NVFP4 quantize -> (w0,w1 packed u32, pvscale_fp8). vals: 16-elem f32 rmem.
    sr selects stochastic rounding (rng_base = the per-block RNG seed) over RTNE."""
    amax = _abs_f32(vals[0])
    for i in range(1, 16):
        amax = _max_f32(amax, _abs_f32(vals[i]))
    # Clamp to [eps, max]: a zero/near-zero block stores eps, not 0 (matches triton, and keeps
    # the decode reciprocal finite).
    pvscale = _min_f32(amax * enc_over_fp4max, cutlass.Float32(FP8_E4M3_MAX))
    pvscale = _max_f32(pvscale, cutlass.Float32(FP8_E4M3_EPS))
    pv_f32 = cute.make_rmem_tensor((4,), cutlass.Float32)
    for i in range(4):
        pv_f32[i] = pvscale
    pv_f8 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
    pv_f8.store(pv_f32.load().to(cutlass.Float8E4M3FN))
    pvscale_fp8 = pv_f8[0]
    pv_back = cute.make_rmem_tensor((4,), cutlass.Float32)
    pv_back.store(pv_f8.load().to(cutlass.Float32))
    denom = pv_back[0] * dec
    enc = _min_f32(_rcp_approx_f32(denom), cutlass.Float32(FP32_MAX))
    q = cute.make_rmem_tensor((16,), cutlass.Float32)
    for i in range(16):
        q[i] = _min_xorsign_abs_f32(vals[i] * enc, cutlass.Float32(FP4_E2M1_MAX))
    w0, w1 = _pack16(q, sr, rng_base)
    return w0, w1, pvscale_fp8


class _Tcgen05RowColFused:
    def __init__(
        self, swizzle_sf: bool = True, sr: bool = False, apply_rht: bool = True
    ):
        # swizzle_sf=True: cutlass NVFP4 swizzled SF (GEMM-ready, TMA-coalesced store).
        # False: plain (N,M//16)/(M,N//16) SF (row SF falls back to a strided SIMT store).
        # sr=True: stochastic rounding (cvt.rs) in the FP4 cast; False: round-to-nearest.
        # apply_rht=True: columnwise path = NVFP4(RHT(A.t())) via the tcgen05 UMMA (the B operand is
        # the Hadamard matrix). False (weight quantize): plain NVFP4(A.t()) — the col warps read the
        # transposed A from SMEM directly (no MMA/TMEM/acc-pipeline, no B), so it's a 2D block-scaling
        # quantize. The RHT path is compiled separately, so its codegen is unchanged.
        self.swizzle_sf = swizzle_sf
        self.sr = sr
        self.apply_rht = apply_rht

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mFP4: cute.Tensor,
        mSF: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        global_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        GRID: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.c_layout = utils.LayoutEnum.from_tensor(mFP4)

        mma_op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            MMA_TILER,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))

        # --- wide A: K = 16*U -> U k-blocks (M-blocks); MN_SW128 swizzle ---
        a_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        a_shape = tiled_mma.partition_shape_A(
            cute.dice((M_TILE, N_TILE, KW), (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            a_atom, cute.append(a_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )
        # clean (M_mma=128, KW=256, STAGE) view of the SAME bytes for the row read
        # (same atom + swizzle -> identical physical mapping to a_smem_layout_staged).
        a_clean_layout = cute.tile_to_shape(
            a_atom, (M_TILE, KW, NUM_AB_STAGE), order=(0, 1, 2)
        )

        # --- narrow B: one 16x16 RHT, 1 k-block ---
        b_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.BFloat16
        )
        b_shape = tiled_mma.partition_shape_B(cute.dice(MMA_TILER, (None, 1, 1)))
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            b_atom, cute.append(b_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            a_smem_layout,
            (M_TILE, N_TILE, KW),
            tiled_mma,
            (1, 1, 1, 1),
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            b_smem_layout,
            MMA_TILER,
            tiled_mma,
            (1, 1, 1, 1),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(cute.append(acc_shape, U), NUM_ACC_STAGE)
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        # Weight mode (apply_rht=False) skips the B (Hadamard) load — no MMA — so don't count it
        # in the TMA tx_count, or the AB-full barrier would wait on bytes that never arrive.
        num_tma_load_bytes = (
            M_TILE * KW + (N_TILE * K if cutlass.const_expr(self.apply_rht) else 0)
        ) * 2

        # COL FP4 store: super-tile = 2U u32 wide
        fp4_smem_layout = cute.make_layout((M_TILE, 2 * U), stride=(2 * U, 1))
        tma_atom_fp4, tma_tensor_fp4 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mFP4,
            fp4_smem_layout,
            (M_TILE, 2 * U),
        )
        # COL SF store (TMA). swizzled: (1, SF_GCOL*SF_BLK) box over flat (N//128, (M//64)*SF_BLK);
        # plain: (M_TILE, U) box over (N, M//16).
        if cutlass.const_expr(self.swizzle_sf):
            col_sf_box = (1, SF_GCOL * SF_BLK)
        else:
            col_sf_box = (M_TILE, U)
        sf_smem_layout = cute.make_layout(col_sf_box, stride=(col_sf_box[1], 1))
        tma_atom_sf, tma_tensor_sf = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mSF,
            sf_smem_layout,
            col_sf_box,
        )

        # ROW FP4 store: super-tile = (KW M-rows, M_TILE//8 u32) = (256,16) = 64B wide -> TMA-ok
        row_fp4_smem_layout = cute.make_layout(
            (KW, M_TILE // 8), stride=(M_TILE // 8, 1)
        )
        tma_atom_row_fp4, tma_tensor_row_fp4 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mRowFP4,
            row_fp4_smem_layout,
            (KW, M_TILE // 8),
        )

        # ROW SF store. swizzled: TMA, (SF_RBLK, SF_RGRP*SF_BLK) box over flat (M//128, (N//64)*SF_BLK).
        # plain: strided SIMT, so this atom is unused (alias the FP4 atom).
        if cutlass.const_expr(self.swizzle_sf):
            row_sf_box = (SF_RBLK, SF_RGRP * SF_BLK)
            row_sf_smem_layout = cute.make_layout(row_sf_box, stride=(row_sf_box[1], 1))
            tma_atom_row_sf, tma_tensor_row_sf = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mRowSF,
                row_sf_smem_layout,
                row_sf_box,
            )
        else:
            tma_atom_row_sf, tma_tensor_row_sf = (
                tma_atom_row_fp4,
                tma_tensor_row_fp4,
            )  # unused

        num_tiles_m = N // cutlass.Int32(M_TILE)  # N output-row tiles (M_mma=128)
        num_tiles_ns = M // cutlass.Int32(
            N_TILE * U
        )  # M contraction block-groups (K=16U)
        num_super = num_tiles_m * num_tiles_ns

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mFP4,
            mSF,
            global_amax_t,
            sr_rng_t,
            tma_atom_fp4,
            tma_tensor_fp4,
            tma_atom_sf,
            tma_tensor_sf,
            mRowFP4,
            mRowSF,
            row_amax_t,
            tma_atom_row_fp4,
            tma_tensor_row_fp4,
            row_fp4_smem_layout,
            tma_atom_row_sf,
            tma_tensor_row_sf,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            fp4_smem_layout,
            sf_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_tiles_ns,
            num_super,
            GRID,
        ).launch(
            grid=(GRID, 1, 1),
            block=(
                FUSED_TPB if cutlass.const_expr(self.apply_rht) else FUSED_TPB_W,
                1,
                1,
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mFP4: cute.Tensor,
        mSF: cute.Tensor,
        global_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        tma_atom_fp4: cute.CopyAtom,
        mFP4_tma: cute.Tensor,
        tma_atom_sf: cute.CopyAtom,
        mSF_tma: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        tma_atom_row_fp4: cute.CopyAtom,
        mRowFP4_tma: cute.Tensor,
        row_fp4_smem_layout: cute.Layout,
        tma_atom_row_sf: cute.CopyAtom,
        mRowSF_tma: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        fp4_smem_layout: cute.Layout,
        sf_smem_layout: cute.Layout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_tiles_ns: cutlass.Int32,
        num_super: cutlass.Int32,
        GRID: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        start_pid, _, _ = cute.arch.block_idx()

        # Warp layout differs by mode: RHT uses the 4-col/8-row + MMA split; weight mode drops the
        # MMA and balances col=8/row=8 (col now does the heavy transposed read, not a TMEM epilogue).
        if cutlass.const_expr(self.apply_rht):
            _COL_END, _ROW_BEG, _ROW_END, _TMA_W = (
                COL_WARP_END,
                ROW_WARP_BEGIN,
                ROW_WARP_END,
                TMA_WARP,
            )
            _COL_THR, _ROW_THR = COL_THREADS, ROW_THREADS
        else:
            _COL_END, _ROW_BEG, _ROW_END, _TMA_W = (
                COL_WARP_END_W,
                ROW_WARP_BEGIN_W,
                ROW_WARP_END_W,
                TMA_WARP_W,
            )
            _COL_THR, _ROW_THR = COL_THREADS_W, ROW_THREADS_W

        if warp_idx == _TMA_W:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_fp4)
            cpasync.prefetch_descriptor(tma_atom_sf)
            cpasync.prefetch_descriptor(tma_atom_row_fp4)
            if cutlass.const_expr(self.swizzle_sf):
                cpasync.prefetch_descriptor(tma_atom_row_sf)

        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_AB_STAGE * 2]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_ACC_STAGE * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # AB pipeline: TMA producer. RHT mode consumers = MMA (1 umma arrive) + ROW_THREADS;
        # weight mode (no MMA) consumers = COL_THREADS_W + ROW_THREADS_W (both warp groups read sA).
        if cutlass.const_expr(self.apply_rht):
            ab_cons_count = 1 + ROW_THREADS
        else:
            ab_cons_count = COL_THREADS_W + ROW_THREADS_W
        ab_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, ab_cons_count)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=NUM_AB_STAGE,
            producer_group=ab_prod_grp,
            consumer_group=ab_cons_grp,
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        acc_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, N_COL_WARPS)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=NUM_ACC_STAGE,
            producer_group=acc_prod_grp,
            consumer_group=acc_cons_grp,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 * (N_COL_WARPS + 1),  # col + mma
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR,
            num_threads=COL_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # --- SMEM: allocate A raw, build two views (MMA swizzle-on-ptr + row swizzle-in-layout) ---
        a_cosize = cute.cosize(a_smem_layout_staged.outer)
        raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        # row view: SAME swizzled pointer, clean (M_mma=128, KW=256, STAGE) *outer* layout
        # -> swizzle applied identically to sA (both PDSL), just a different logical grouping.
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)

        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sFP4 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=fp4_smem_layout,
            byte_alignment=128,
        )
        # COL SF SMEM. swizzled: raw bytes with a 4D write-view (group, 32, 16) + a 2D TMA-view
        # over the same memory; plain: a single (M_TILE, U) tile (write == TMA view).
        if cutlass.const_expr(self.swizzle_sf):
            raw_csf = smem.allocate_array(
                cutlass.Float8E4M3FN, SF_GCOL * SF_BLK, byte_alignment=128
            )
            sSF_w = cute.make_tensor(
                raw_csf,
                cute.make_layout(
                    (1, SF_GCOL, 32, 16), stride=(SF_GCOL * SF_BLK, SF_BLK, 16, 1)
                ),
            )
            sSF = cute.make_tensor(
                raw_csf, sf_smem_layout
            )  # (1, SF_GCOL*SF_BLK) TMA view
        else:
            sSF = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=sf_smem_layout,
                byte_alignment=128,
            )
            sSF_w = sSF  # (M_TILE, U) write == TMA

        # double-buffered row FP4 staging: overlap TMA store with next iter's compute
        row_fp4_staged = cute.make_layout(
            (KW, M_TILE // 8, ROW_FP4_STAGES),
            stride=(M_TILE // 8, 1, KW * (M_TILE // 8)),
        )
        sRowFP4 = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=row_fp4_staged,
            byte_alignment=128,
        )
        # ROW SF SMEM (swizzled only; plain row SF is a strided SIMT store, no staging).
        # Double-buffered (same as row FP4) so its TMA store overlaps the next iter's compute.
        sRowSF_w = None
        sRowSF = None
        if cutlass.const_expr(self.swizzle_sf):
            _rsf_stage = SF_RBLK * SF_RGRP * SF_BLK
            raw_rsf = smem.allocate_array(
                cutlass.Float8E4M3FN, _rsf_stage * ROW_FP4_STAGES, byte_alignment=128
            )
            sRowSF_w = cute.make_tensor(
                raw_rsf,
                cute.make_layout(
                    (SF_RBLK, SF_RGRP, 32, 16, ROW_FP4_STAGES),
                    stride=(SF_RGRP * SF_BLK, SF_BLK, 16, 1, _rsf_stage),
                ),
            )
            sRowSF = cute.make_tensor(
                raw_rsf,
                cute.make_layout(
                    (SF_RBLK, SF_RGRP * SF_BLK, ROW_FP4_STAGES),
                    stride=(SF_RGRP * SF_BLK, 1, _rsf_stage),
                ),
            )

        # --- global -> mma partition (wide A tiler = (M_TILE, KW)) ---
        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((M_TILE, N_TILE, KW), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)

        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, M, U, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)

        def _global_scale(amax):
            is_zero = amax == cutlass.Float32(0.0)
            safe = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), amax))
            c = _min_f32(
                cutlass.Float32(FP8_E4M3_MAX * FP4_E2M1_MAX) / safe,
                cutlass.Float32(FP32_MAX),
            )
            c = cutlass.Float32(
                cutlass.select_(c == cutlass.Float32(0.0), cutlass.Float32(1.0), c)
            )
            enc = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), c))
            dec = cutlass.Float32(1.0) / enc
            return enc, dec, enc * cutlass.Float32(1.0 / FP4_E2M1_MAX)

        g_enc, g_dec, enc_over_fp4max = _global_scale(global_amax_t[0])  # col (RHT)
        r_enc, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[0])  # row (plain)
        sr_rng = cutlass.Uint32(0)
        if cutlass.const_expr(self.sr):
            sr_rng = cutlass.Uint32(
                sr_rng_t[0]
            )  # per-call stochastic-rounding RNG base

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        rem = num_super - start_pid
        num_iters = cutlass.select_(
            rem > cutlass.Int32(0),
            (rem + GRID - cutlass.Int32(1)) // GRID,
            cutlass.Int32(0),
        )

        # ==================== TMA warp (AB producer) ====================
        if warp_idx == _TMA_W:
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, pid_m, pid_ns, 0)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                if cutlass.const_expr(
                    self.apply_rht
                ):  # B (Hadamard) only needed for the MMA
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, 0)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
            ab_producer.tail()

        # ==================== MMA warp (AB consumer, acc producer) ====================
        if warp_idx == MMA_WARP and cutlass.const_expr(self.apply_rht):
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_ACC_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                ab_handle = ab_consumer.wait_and_advance()
                acc_pipeline.producer_acquire(acc_producer_state)
                for u in cutlass.range_constexpr(U):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                        tCrA[(None, None, u, ab_handle.index)],
                        tCrB[(None, None, 0, ab_handle.index)],
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                    )
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                ab_handle.release()  # 1 UMMA arrive on AB empty barrier
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== ROW warps (AB consumer, read raw sA) ====================
        if warp_idx >= _ROW_BEG and warp_idx < _ROW_END:
            k_row = tidx - _ROW_BEG * cutlass.Int32(
                32
            )  # 0..KW-1 = M-position within super-tile
            row_store_barrier = pipeline.NamedBarrier(
                barrier_id=ROW_STORE_BAR, num_threads=_ROW_THR
            )
            row_ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_AB_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                m0 = pid_ns * cutlass.Int32(KW)  # global M-row base
                buf = local_iter % ROW_FP4_STAGES  # double-buffer index

                ab_pipeline.consumer_wait(row_ab_state)  # wait sA full
                stage = row_ab_state.index

                # read this thread's M-row (128 N values across the m_mma grain), 8 SF-blocks
                kc = k_row % cutlass.Int32(8)
                kd = k_row // cutlass.Int32(8)
                for b in cutlass.range_constexpr(M_TILE // 16):  # 8 blocks of 16 N
                    blk = cute.make_rmem_tensor((16,), cutlass.Float32)
                    for j in cutlass.range_constexpr(16):
                        m_mma = b * 16 + j  # N position (0..127), python int
                        blk[j] = sA_clean[
                            ((m_mma % 64, m_mma // 64), (kc, kd), (0, stage))
                        ].to(cutlass.Float32)
                    row_rng = None
                    if cutlass.const_expr(self.sr):
                        # row-stream RNG seed for this (global M-row, SF-col block); 0x00B0.. tags row
                        row_rng = (
                            sr_rng
                            ^ (cutlass.Uint32(m0 + k_row) << cutlass.Uint32(8))
                            ^ cutlass.Uint32(
                                pid_m * cutlass.Int32(M_TILE // 16) + cutlass.Int32(b)
                            )
                            ^ cutlass.Uint32(0x00B00200)
                        )
                    w0, w1, sf = _quant16(
                        blk, r_enc_over_fp4max, r_dec, self.sr, row_rng
                    )
                    sRowFP4[k_row, b * 2, buf] = w0
                    sRowFP4[k_row, b * 2 + 1, buf] = w1
                    if cutlass.const_expr(self.swizzle_sf):
                        # swizzled SF[r=m0+k_row, c=pid_m*8+b] -> [r//128, c//4, r%32, (r%128//32)*4 + c%4]
                        sRowSF_w[
                            k_row // cutlass.Int32(128),
                            b // 4,
                            k_row % cutlass.Int32(32),
                            ((k_row // cutlass.Int32(32)) % cutlass.Int32(4))
                            * cutlass.Int32(4)
                            + (b % 4),
                            buf,
                        ] = sf
                    else:
                        mRowSF[
                            m0 + k_row,
                            pid_m * cutlass.Int32(M_TILE // 16) + cutlass.Int32(b),
                        ] = sf

                # all M-rows read -> release AB buffer (1 thread arrive each)
                cute.arch.mbarrier_arrive(
                    ab_pipeline.sync_object_empty.get_barrier(stage)
                )
                row_ab_state.advance()

                # TMA-store the (KW, M_TILE//8) row FP4 tile from buffer `buf`.
                # wait_group(1) keeps <=1 store in flight -> this store overlaps the next
                # iter's read/quant; the 2nd barrier makes its completion (via the *next*
                # iter's wait_group) visible before buf is reused two iters later.
                cute.arch.fence_proxy("async.shared", space="cta")
                row_store_barrier.arrive_and_wait()
                if warp_idx == cutlass.Int32(_ROW_BEG):
                    gRowFP4 = cute.local_tile(
                        mRowFP4_tma, (KW, M_TILE // 8), (pid_ns, pid_m)
                    )
                    tRs, tRg = cpasync.tma_partition(
                        tma_atom_row_fp4,
                        0,
                        cta_layout,
                        cute.group_modes(sRowFP4[(None, None, buf)], 0, 2),
                        cute.group_modes(gRowFP4, 0, 2),
                    )
                    cute.copy(tma_atom_row_fp4, tRs, tRg)
                    if cutlass.const_expr(self.swizzle_sf):
                        gRowSF = cute.local_tile(
                            mRowSF_tma, (SF_RBLK, SF_RGRP * SF_BLK), (pid_ns, pid_m)
                        )
                        tRSs, tRSg = cpasync.tma_partition(
                            tma_atom_row_sf,
                            0,
                            cta_layout,
                            cute.group_modes(sRowSF[(None, None, buf)], 0, 2),
                            cute.group_modes(gRowSF, 0, 2),
                        )
                        cute.copy(tma_atom_row_sf, tRSs, tRSg)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                row_store_barrier.arrive_and_wait()
            if warp_idx == cutlass.Int32(_ROW_BEG):
                cute.arch.cp_async_bulk_wait_group(0, read=True)  # drain last store

        # ==================== COL epilogue warps (acc consumer, TMEM) ====================
        if warp_idx < COL_WARP_END and cutlass.const_expr(self.apply_rht):
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                MMA_TILER,
                self.c_layout,
                cutlass.Float32,
                cutlass.Float32,
                MMA_TILER[:2],
                False,
            )
            tAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tAcc_epi = cute.flat_divide(tAcc, MMA_TILER[:2])
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)

            epi_store_barrier = pipeline.NamedBarrier(
                barrier_id=EPI_STORE_BAR, num_threads=COL_THREADS
            )
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_ACC_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                acc_idx = acc_consumer_state.index

                acc_pipeline.consumer_wait(acc_consumer_state)
                for u in cutlass.range_constexpr(U):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_base[(None, None, None, 0, 0, u, acc_idx)],
                        tTR_rAcc,
                    )
                    vals = tTR_rAcc.load().reshape((16,))
                    col_rng = None
                    if cutlass.const_expr(self.sr):
                        # col-stream RNG seed for this (global row, u-block); 0x00C0.. tags col stream
                        col_rng = (
                            sr_rng
                            ^ (
                                cutlass.Uint32(pid_m * cutlass.Int32(M_TILE) + tidx)
                                << cutlass.Uint32(8)
                            )
                            ^ cutlass.Uint32(u)
                            ^ cutlass.Uint32(0x00C01000)
                        )
                    w0, w1, pvscale_fp8 = _quant16(
                        vals, enc_over_fp4max, g_dec, self.sr, col_rng
                    )

                    sFP4[tidx, u * 2] = w0
                    sFP4[tidx, u * 2 + 1] = w1
                    if cutlass.const_expr(self.swizzle_sf):
                        # swizzled SF[r=pid_m*128+tidx, c=pid_ns*16+u] -> [r//128, c//4, r%32, (r%128//32)*4 + c%4]
                        sSF_w[
                            0,
                            u // 4,
                            tidx % cutlass.Int32(32),
                            (tidx // cutlass.Int32(32)) * cutlass.Int32(4) + (u % 4),
                        ] = pvscale_fp8
                    else:
                        sSF_w[tidx, u] = pvscale_fp8

                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                cute.arch.fence_proxy("async.shared", space="cta")
                epi_store_barrier.arrive_and_wait()
                if warp_idx == cutlass.Int32(0):
                    gFP4 = cute.local_tile(mFP4_tma, (M_TILE, 2 * U), (pid_m, pid_ns))
                    tSs, tSg = cpasync.tma_partition(
                        tma_atom_fp4,
                        0,
                        cta_layout,
                        cute.group_modes(sFP4, 0, 2),
                        cute.group_modes(gFP4, 0, 2),
                    )
                    cute.copy(tma_atom_fp4, tSs, tSg)
                    if cutlass.const_expr(self.swizzle_sf):
                        gSF = cute.local_tile(
                            mSF_tma, (1, SF_GCOL * SF_BLK), (pid_m, pid_ns)
                        )
                    else:
                        gSF = cute.local_tile(mSF_tma, (M_TILE, U), (pid_m, pid_ns))
                    tSFs, tSFg = cpasync.tma_partition(
                        tma_atom_sf,
                        0,
                        cta_layout,
                        cute.group_modes(sSF, 0, 2),
                        cute.group_modes(gSF, 0, 2),
                    )
                    cute.copy(tma_atom_sf, tSFs, tSFg)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epi_store_barrier.arrive_and_wait()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ========= COL weight-mode warps: plain NVFP4(A.t()), read transposed A from SMEM =========
        # No MMA: the col warps are AB consumers (like the row warps) and read A.t() directly from
        # sA_clean — the same swizzled bytes the row path reads, in the transposed grain.
        if warp_idx < _COL_END and cutlass.const_expr(not self.apply_rht):
            col_store_barrier = pipeline.NamedBarrier(
                barrier_id=EPI_STORE_BAR, num_threads=_COL_THR
            )
            col_ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_AB_STAGE
            )
            # 8 col warps (256 threads) cover the 128 N-rows with 2 threads/row: nrow = the N-row,
            # u_half selects which half of the U u-blocks this thread owns (each does U//2 blocks).
            nrow = tidx % cutlass.Int32(M_TILE)
            u_half = tidx // cutlass.Int32(M_TILE)
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns

                ab_pipeline.consumer_wait(col_ab_state)  # wait sA full
                stage = col_ab_state.index
                for u_local in cutlass.range_constexpr(U // 2):
                    u = cutlass.Int32(u_local) + u_half * cutlass.Int32(U // 2)
                    blk = cute.make_rmem_tensor((16,), cutlass.Float32)
                    for i in cutlass.range_constexpr(16):
                        mpos = u * cutlass.Int32(16) + cutlass.Int32(
                            i
                        )  # M-position (0..255)
                        # transposed read: A.t()[N-row=nrow, M-pos=mpos]
                        blk[i] = sA_clean[
                            ((nrow % 64, nrow // 64), (mpos % 8, mpos // 8), (0, stage))
                        ].to(cutlass.Float32)
                    col_rng = None
                    if cutlass.const_expr(self.sr):
                        col_rng = (
                            sr_rng
                            ^ (
                                cutlass.Uint32(pid_m * cutlass.Int32(M_TILE) + nrow)
                                << cutlass.Uint32(8)
                            )
                            ^ cutlass.Uint32(u)
                            ^ cutlass.Uint32(0x00C01000)
                        )
                    w0, w1, sf = _quant16(blk, enc_over_fp4max, g_dec, self.sr, col_rng)
                    sFP4[nrow, u * 2] = w0
                    sFP4[nrow, u * 2 + 1] = w1
                    if cutlass.const_expr(self.swizzle_sf):
                        sSF_w[
                            0,
                            u // cutlass.Int32(4),
                            nrow % cutlass.Int32(32),
                            (nrow // cutlass.Int32(32)) * cutlass.Int32(4)
                            + (u % cutlass.Int32(4)),
                        ] = sf
                    else:
                        sSF_w[nrow, u] = sf

                cute.arch.mbarrier_arrive(
                    ab_pipeline.sync_object_empty.get_barrier(stage)
                )
                col_ab_state.advance()

                cute.arch.fence_proxy("async.shared", space="cta")
                col_store_barrier.arrive_and_wait()
                if warp_idx == cutlass.Int32(0):
                    gFP4 = cute.local_tile(mFP4_tma, (M_TILE, 2 * U), (pid_m, pid_ns))
                    tSs, tSg = cpasync.tma_partition(
                        tma_atom_fp4,
                        0,
                        cta_layout,
                        cute.group_modes(sFP4, 0, 2),
                        cute.group_modes(gFP4, 0, 2),
                    )
                    cute.copy(tma_atom_fp4, tSs, tSg)
                    if cutlass.const_expr(self.swizzle_sf):
                        gSF = cute.local_tile(
                            mSF_tma, (1, SF_GCOL * SF_BLK), (pid_m, pid_ns)
                        )
                    else:
                        gSF = cute.local_tile(mSF_tma, (M_TILE, U), (pid_m, pid_ns))
                    tSFs, tSFg = cpasync.tma_partition(
                        tma_atom_sf,
                        0,
                        cta_layout,
                        cute.group_modes(sSF, 0, 2),
                        cute.group_modes(gSF, 0, 2),
                    )
                    cute.copy(tma_atom_sf, tSFs, tSFg)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                col_store_barrier.arrive_and_wait()


class _Tcgen05RhtAmax:
    """One-pass tensor-core RHT amax over the dual-consumer A load.

    Each epilogue reduces to a global max-abs instead of quantizing:
      col_amax = max|RHT(A.t())|  (TMEM accumulator),  row_amax = max|A|  (raw sA).
    The 16x16 RHT runs on tensor cores, so the pass is HBM-bound. Requires
    M % 256 == 0, N % 128 == 0.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        col_amax_t: cute.Tensor,
        row_amax_t: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        GRID: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        mma_op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            MMA_TILER,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))

        a_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        a_shape = tiled_mma.partition_shape_A(
            cute.dice((M_TILE, N_TILE, KW), (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            a_atom, cute.append(a_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )
        a_clean_layout = cute.tile_to_shape(
            a_atom, (M_TILE, KW, NUM_AB_STAGE), order=(0, 1, 2)
        )

        b_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.BFloat16
        )
        b_shape = tiled_mma.partition_shape_B(cute.dice(MMA_TILER, (None, 1, 1)))
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            b_atom, cute.append(b_shape, NUM_AB_STAGE), order=(1, 2, 3)
        )

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            a_smem_layout,
            (M_TILE, N_TILE, KW),
            tiled_mma,
            (1, 1, 1, 1),
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            b_smem_layout,
            MMA_TILER,
            tiled_mma,
            (1, 1, 1, 1),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(cute.append(acc_shape, U), NUM_ACC_STAGE)
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        num_tma_load_bytes = (M_TILE * KW + N_TILE * K) * 2
        num_tiles_ns = M // cutlass.Int32(N_TILE * U)
        num_super = (N // cutlass.Int32(M_TILE)) * num_tiles_ns

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            col_amax_t,
            row_amax_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_tiles_ns,
            num_super,
            GRID,
        ).launch(grid=(GRID, 1, 1), block=(FUSED_TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        col_amax_t: cute.Tensor,
        row_amax_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_tiles_ns: cutlass.Int32,
        num_super: cutlass.Int32,
        GRID: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        start_pid, _, _ = cute.arch.block_idx()
        lane = tidx % cutlass.Int32(32)

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_AB_STAGE * 2]
            acc_full_mbar: cute.struct.MemRange[cutlass.Int64, NUM_ACC_STAGE * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # AB pipeline: TMA producer; consumers = MMA (1 umma arrive) + ROW_THREADS (thread arrives)
        ab_cons_count = 1 + ROW_THREADS
        ab_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, ab_cons_count)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar.data_ptr(),
            num_stages=NUM_AB_STAGE,
            producer_group=ab_prod_grp,
            consumer_group=ab_cons_grp,
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        acc_prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, N_COL_WARPS)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar.data_ptr(),
            num_stages=NUM_ACC_STAGE,
            producer_group=acc_prod_grp,
            consumer_group=acc_cons_grp,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 * (N_COL_WARPS + 1),  # col + mma
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR,
            num_threads=COL_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # SMEM: A raw + two views (MMA swizzle-on-ptr + row clean view), B
        a_cosize = cute.cosize(a_smem_layout_staged.outer)
        raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)
        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((M_TILE, N_TILE, KW), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)

        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        rem = num_super - start_pid
        num_iters = cutlass.select_(
            rem > cutlass.Int32(0),
            (rem + GRID - cutlass.Int32(1)) // GRID,
            cutlass.Int32(0),
        )

        # ==================== TMA warp (AB producer) ====================
        if warp_idx == TMA_WARP:
            for local_iter in cutlass.range(num_iters):
                super_id = start_pid + local_iter * GRID
                pid_m = super_id // num_tiles_ns
                pid_ns = super_id % num_tiles_ns
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, pid_m, pid_ns, 0)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()

        # ==================== MMA warp (AB consumer, acc producer) ====================
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_ACC_STAGE
            )
            for local_iter in cutlass.range(num_iters):
                ab_handle = ab_consumer.wait_and_advance()
                acc_pipeline.producer_acquire(acc_producer_state)
                for u in cutlass.range_constexpr(U):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                        tCrA[(None, None, u, ab_handle.index)],
                        tCrB[(None, None, 0, ab_handle.index)],
                        tCtAcc_base[(None, None, None, u, acc_producer_state.index)],
                    )
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                ab_handle.release()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== ROW warps (AB consumer, read raw sA) -> row amax ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < ROW_WARP_END:
            k_row = tidx - cutlass.Int32(
                ROW_WARP_BEGIN * 32
            )  # 0..KW-1 = M-position within super-tile
            row_ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_AB_STAGE
            )
            thread_row_max = cutlass.Float32(0.0)
            kc = k_row % cutlass.Int32(8)
            kd = k_row // cutlass.Int32(8)
            for local_iter in cutlass.range(num_iters):
                ab_pipeline.consumer_wait(row_ab_state)
                stage = row_ab_state.index
                for b in cutlass.range_constexpr(M_TILE // 16):  # 8 blocks of 16 N
                    for j in cutlass.range_constexpr(16):
                        m_mma = b * 16 + j  # N position (0..127)
                        v = sA_clean[
                            ((m_mma % 64, m_mma // 64), (kc, kd), (0, stage))
                        ].to(cutlass.Float32)
                        thread_row_max = _max_f32(thread_row_max, _abs_f32(v))
                cute.arch.mbarrier_arrive(
                    ab_pipeline.sync_object_empty.get_barrier(stage)
                )
                row_ab_state.advance()
            for offset in [16, 8, 4, 2, 1]:
                thread_row_max = _max_f32(
                    thread_row_max, cute.arch.shuffle_sync_bfly(thread_row_max, offset)
                )
            if lane == cutlass.Int32(0):
                _atom_max_f32_nonneg(row_amax_t.iterator, thread_row_max)

        # ==================== COL epilogue warps (acc consumer, TMEM) -> col amax ====================
        if warp_idx < COL_WARP_END:
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                MMA_TILER,
                self.c_layout,
                cutlass.Float32,
                cutlass.Float32,
                MMA_TILER[:2],
                False,
            )
            tAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tAcc_epi = cute.flat_divide(tAcc, MMA_TILER[:2])
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_ACC_STAGE
            )
            thread_col_max = cutlass.Float32(0.0)
            for local_iter in cutlass.range(num_iters):
                acc_idx = acc_consumer_state.index
                acc_pipeline.consumer_wait(acc_consumer_state)
                for u in cutlass.range_constexpr(U):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_base[(None, None, None, 0, 0, u, acc_idx)],
                        tTR_rAcc,
                    )
                    vals = tTR_rAcc.load().reshape((16,))
                    for i in cutlass.range_constexpr(16):
                        thread_col_max = _max_f32(thread_col_max, _abs_f32(vals[i]))
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            for offset in [16, 8, 4, 2, 1]:
                thread_col_max = _max_f32(
                    thread_col_max, cute.arch.shuffle_sync_bfly(thread_col_max, offset)
                )
            if lane == cutlass.Int32(0):
                _atom_max_f32_nonneg(col_amax_t.iterator, thread_col_max)

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ---------------------------------------------------------------------------
# Public entry: device amax + fused kernel -> reads A in place, writes fresh outputs.
# Small per-device state (RHT/identity/RNG buffers, compiled kernels) is cached; under
# CUDA graphs it must be pre-allocated via cutedsl_prepare_for_cuda_graph before capture.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _get_rht_buffer(sign_vector, device_idx):
    """Tiny (16,16,1) RHT torch buffer, cached per (sign_vector, device).

    maxsize=None keeps every key resident; the key count is bounded by the number of distinct
    sign vectors in use (each buffer is a 16x16 bf16 tensor). Transposed (H^T) so the MMA
    operand layout yields v @ H.
    """
    device = torch.device("cuda", device_idx)
    rht_nk = (
        get_rht_matrix(sign_vector, device, torch.bfloat16, HADAMARD_DIM)
        .t()
        .contiguous()
    )
    return rht_nk.reshape(N_TILE, K, 1)


@functools.lru_cache(maxsize=None)
def _get_identity_buffer(device_idx):
    """(16,16,1) placeholder for the kernel's ``B`` (Hadamard) operand on the weight-quantize
    (``apply_rht=False``) path. That path has no MMA, so B is never loaded; this only supplies a
    valid tensor for the B TMA-atom shape."""
    device = torch.device("cuda", device_idx)
    eye = torch.eye(HADAMARD_DIM, dtype=torch.bfloat16, device=device).contiguous()
    return eye.reshape(N_TILE, K, 1)


@functools.lru_cache(maxsize=None)
def _get_sr_rng_buffer(device_idx):
    """Persistent (1,) int32 stochastic-rounding RNG-base buffer, cached per device.

    The fused kernel reads its SR seed from this *stable* address. A fresh per-call tensor would be
    an untracked live allocation in the CUDA-graph pool (``torch.compile(mode="reduce-overhead")``)
    AND a correctness hazard — the captured kernel would read a recycled address on replay. The SR
    path copies the per-call ``(seed ^ offset)`` value in (the copy is captured, so each graph
    replay re-reads the freshly-advanced offset); the RTNE path never reads it (the kernel guards
    the read behind ``const_expr(sr)``), so its value is irrelevant there.
    """
    return torch.zeros(1, dtype=torch.int32, device=torch.device("cuda", device_idx))


@functools.lru_cache(maxsize=8)
def _compile_amax_tc_kernel(device_idx):
    """Compile the tensor-core RHT amax with symbolic shapes (cached per device).

    The symbolic (sym_int) shapes make the compiled kernel serve any (M % 256, N % 128); M/N/GRID
    are runtime Int32 args. Not keyed on the sign vector, which is a runtime launch buffer (the
    compile uses a fake), so the compiled kernel is identical for every sign vector.
    """
    device = torch.device("cuda", device_idx)
    # aT = A.t().unsqueeze(-1): (N, M, 1), dim0 (N) contiguous -> stride (1, N, 1).
    m_sym = cute.sym_int(divisibility=N_TILE * U)  # M % 256
    n_sym = cute.sym_int(divisibility=M_TILE)  # N % 128
    fake_aT = make_fake_tensor(
        cutlass.BFloat16, (n_sym, m_sym, 1), stride=(1, cute.sym_int(), 1)
    )
    fake_bT = make_fake_tensor(
        cutlass.BFloat16, (HADAMARD_DIM, HADAMARD_DIM, 1), stride=(HADAMARD_DIM, 1, 1)
    )
    fake_col = make_fake_tensor(cutlass.Float32, (1,), stride=(1,))
    fake_row = make_fake_tensor(cutlass.Float32, (1,), stride=(1,))
    k = _Tcgen05RhtAmax()
    # c_layout for the TMEM->reg read = layout of the col FP4 output (row-major 2D); the enum
    # depends only on row/col-majorness, so a dummy contiguous tensor suffices.
    dummy = torch.empty((M_TILE, 16), dtype=torch.int32, device=device)
    k.c_layout = utils.LayoutEnum.from_tensor(from_dlpack(dummy))
    return cute.compile(
        k,
        fake_aT,
        fake_bT,
        fake_col,
        fake_row,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_rht_amax_impl(A: torch.Tensor, sign_vector=DEFAULT_SIGN_VECTOR):
    """Global amaxes for NVFP4 two-level scaling.

    Returns (col_amax, row_amax) as scalar (1,) float32 tensors:
      - col_amax = max|RHT(A.t())|  (the columnwise path quantizes RHT data)
      - row_amax = max|A|           (the rowwise path quantizes A directly)

    The column amax is taken over the post-RHT data (not the plain amax) for correctness: RHT
    can raise the per-block max, and a too-small global scale saturates the E4M3 block scales.
    Requires M % 256 == 0, N % 128 == 0.
    """
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    if not A.is_contiguous():
        raise ValueError("A must be row-major (contiguous)")
    M, N = A.shape
    # The tile is N_TILE*U=256 wide in M. Without M%256, M=128 gives GRID=0
    # -> a no-op launch that silently returns amax=0.
    if M % 256 != 0:
        raise ValueError(f"M must be divisible by 256, got M={M}")
    if N % 128 != 0:
        raise ValueError(f"N must be divisible by 128, got N={N}")
    # This is a non-differentiable op (autograd is owned by the outer linear Function);
    # detach so the input passed to the kernel never carries autograd state.
    A = A.detach()
    dev = A.device
    col_amax = torch.zeros(1, dtype=torch.float32, device=dev)
    row_amax = torch.zeros(1, dtype=torch.float32, device=dev)

    rht_nk = _get_rht_buffer(tuple(sign_vector), dev.index)  # torch buffer (kept alive)

    NUM_SMS = _get_num_sms(dev.index)
    GRID = min(NUM_SMS, (N // M_TILE) * (M // (N_TILE * U)))
    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))

    amax_compiled = _compile_amax_tc_kernel(dev.index)
    amax_compiled(
        A.t().unsqueeze(-1),
        rht_nk,
        col_amax,
        row_amax,
        int(M),
        int(N),
        int(GRID),
        stream,
    )
    return col_amax, row_amax


@functools.lru_cache(maxsize=16)
def _compile_fused_kernel(device_idx, swizzle, sr, apply_rht=True):
    """Compile the fused kernel with symbolic shapes (cached per device+swizzle+sr+apply_rht).

    The symbolic (sym_int) shapes make the compiled kernel serve any (M % 256, N % 128); the
    divisibilities below match each output's TMA store box so the atoms tile cleanly. ``swizzle``
    selects the cutlass-swizzled SF layout (op default) vs the plain (N, M//16)/(M, N//16) layout.
    ``sr`` compiles the stochastic-rounding (cvt.rs) variant as a separate kernel, leaving the RTNE
    forward path untouched. ``apply_rht=False`` compiles the no-MMA weight-quantize variant (the col
    path reads transposed A from SMEM). Not keyed on the sign vector / RNG (runtime launch buffers).
    """
    m_sym = cute.sym_int(divisibility=N_TILE * U)  # M % 256
    n_sym = cute.sym_int(divisibility=M_TILE)  # N % 128
    free = cute.sym_int  # a fresh dynamic stride per call

    # aT = A.t().unsqueeze(-1): (N, M, 1), dim0 contiguous; output FP4 tensors row-major.
    fake_aT = make_fake_tensor(
        cutlass.BFloat16, (n_sym, m_sym, 1), stride=(1, free(), 1)
    )
    fake_bT = make_fake_tensor(
        cutlass.BFloat16, (HADAMARD_DIM, HADAMARD_DIM, 1), stride=(HADAMARD_DIM, 1, 1)
    )
    # col_fp4 (N, M//8) u32, store box inner = 2*U = 32; row_fp4 (M, N//8) u32, inner = 16.
    fake_cfp4 = make_fake_tensor(
        cutlass.Uint32, (n_sym, cute.sym_int(divisibility=2 * U)), stride=(free(), 1)
    )
    fake_rfp4 = make_fake_tensor(
        cutlass.Uint32,
        (m_sym, cute.sym_int(divisibility=M_TILE // 8)),
        stride=(free(), 1),
    )
    if swizzle:
        # SF flattened to 2D for the TMA atom: col_sf.reshape(N//128, (M//64)*512) has inner
        # divisible by 2048 (from M%256); row_sf.reshape(M//128, (N//64)*512) inner by 1024.
        fake_csf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (cute.sym_int(divisibility=1), cute.sym_int(divisibility=2048)),
            stride=(free(), 1),
        )
        fake_rsf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (cute.sym_int(divisibility=2), cute.sym_int(divisibility=1024)),
            stride=(free(), 1),
        )
    else:
        # plain SF: col (N, M//16), row (M, N//16).
        fake_csf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (n_sym, cute.sym_int(divisibility=U)),
            stride=(free(), 1),
        )
        fake_rsf = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (m_sym, cute.sym_int(divisibility=M_TILE // 16)),
            stride=(free(), 1),
        )
    fake_amax = make_fake_tensor(cutlass.Float32, (1,), stride=(1,))
    fake_sr_rng = make_fake_tensor(cutlass.Int32, (1,), stride=(1,))
    k = _Tcgen05RowColFused(swizzle_sf=swizzle, sr=sr, apply_rht=apply_rht)
    return cute.compile(
        k,
        fake_aT,
        fake_bT,
        fake_cfp4,
        fake_csf,
        fake_rfp4,
        fake_rsf,
        fake_amax,
        fake_amax,
        fake_sr_rng,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_rht_quantize_row_col_impl(
    A: torch.Tensor,
    col_global_amax: torch.Tensor,
    row_global_amax: torch.Tensor,
    sign_vector=DEFAULT_SIGN_VECTOR,
    stochastic_rounding: bool = False,
    sr_rng: Optional[torch.Tensor] = None,
    *,
    swizzle_scale_factors: bool = True,
    compute_rowwise: bool = True,
    apply_rht: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused RHT + NVFP4 E2M1 columnwise quantize with rowwise quantize.

    The two global amaxes are taken as input: the caller computes them first (via
    ``cutedsl_rht_amax``, optionally all-reducing for TP). The columnwise path quantizes
    RHT(A.t()) scaled by ``col_global_amax``; the rowwise path quantizes A directly scaled
    by ``row_global_amax``.

    ``apply_rht=False`` selects the no-MMA weight-quantize variant: with no Hadamard rotation the
    columnwise warps read the transposed ``A`` straight from SMEM (a plain ``A.t()`` transpose-
    quantize) instead of an MMA accumulator, with col/row warps balanced. Used by the weight quantize
    (weights are not RHT-rotated); the caller passes ``col_global_amax == row_global_amax == max|A|``
    since ``max|A.t()| == max|A|``.

    Args:
        A: (M, N) bfloat16, row-major. M % 256 == 0, N % 128 == 0.
        col_global_amax: scalar float32 = max|RHT(A.t())| (columnwise decode scale).
        row_global_amax: scalar float32 = max|A| (rowwise decode scale).
        sign_vector: RHT sign vector as a list of ints.
        stochastic_rounding: if True, both quant paths round via the Blackwell ``cvt.rs`` HW
            stochastic-rounding cvt seeded by ``sr_rng``. False -> RTNE (default).
        sr_rng: int RNG base tensor (1-elem) required when ``stochastic_rounding=True``. The kernel
            decorrelates col vs row and per-block internally, so a single base suffices; the caller
            mixes a fixed seed with a fresh per-call offset for CUDA-graph-replay advancement.
        swizzle_scale_factors: cutlass NVFP4 swizzled SF (default, GEMM-ready). False -> plain
            (N,M//16)/(M,N//16) SF, which uses a slower strided row-SF store.
        compute_rowwise: return the rowwise output (default). False -> row_fp4/row_sf returned as
            None. NOTE: the fused kernel always computes + stores the row path; this flag only
            gates the *return*, it does not skip the rowwise work.

    Returns:
        4-tuple (col_fp4, col_sf, row_fp4, row_sf):
          - col_fp4: (N, M//2) uint8 packed FP4 (columnwise).
          - col_sf:  (N//128, M//64, 32, 16) float8_e4m3fn swizzled (or (N, M//16) plain).
          - row_fp4: (M, N//2) uint8 packed FP4 (rowwise), or None if compute_rowwise=False.
          - row_sf:  (M//128, N//64, 32, 16) float8_e4m3fn swizzled (or (M, N//16) plain), or None.
    """
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    if not A.is_contiguous():
        raise ValueError("A must be row-major (contiguous)")
    M, N = A.shape
    if M % 256 != 0:
        raise ValueError(f"M must be divisible by 256, got M={M}")
    if N % 128 != 0:
        raise ValueError(f"N must be divisible by 128, got N={N}")
    # Non-differentiable op (autograd owned by the outer linear Function); detach so the
    # input passed to the kernel never carries autograd state.
    A = A.detach()
    for name, t in (
        ("col_global_amax", col_global_amax),
        ("row_global_amax", row_global_amax),
    ):
        if t.numel() != 1:
            raise ValueError(f"{name} must contain a single element, got {t.numel()}")
        if t.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {t.dtype}")
    dev = A.device
    swizzle = bool(swizzle_scale_factors)
    sr = bool(stochastic_rounding)
    # Persistent buffer (stable address for CUDA-graph capture); see _get_sr_rng_buffer.
    sr_rng_t = _get_sr_rng_buffer(dev.index)
    if sr:
        if sr_rng is None:
            raise ValueError(
                "stochastic_rounding=True requires sr_rng (RNG base tensor)"
            )
        # Single non-negative int32 device scalar; the kernel decorrelates col vs row + per-block
        # internally via XOR tags, so one base value suffices. Written in-place (captured by the
        # graph) so each replay re-reads the freshly-advanced offset.
        sr_rng_t.copy_((sr_rng.reshape(1).to(torch.int64) & 0x7FFFFFFF).to(torch.int32))

    col_fp4 = torch.empty((N, M // 8), dtype=torch.uint32, device=dev)
    row_fp4 = torch.empty((M, N // 8), dtype=torch.uint32, device=dev)
    if swizzle:
        col_sf = torch.empty(
            (N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
        )
        row_sf = torch.empty(
            (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
        )
        csf_g = col_sf.reshape(
            N // 128, (M // 64) * 32 * 16
        )  # flat 2D for the TMA atom
        rsf_g = row_sf.reshape(M // 128, (N // 64) * 32 * 16)
    else:
        col_sf = torch.empty((N, M // 16), dtype=torch.float8_e4m3fn, device=dev)
        row_sf = torch.empty((M, N // 16), dtype=torch.float8_e4m3fn, device=dev)
        csf_g, rsf_g = col_sf, row_sf

    # The MMA B operand: the Hadamard (RHT) matrix, or an identity for a plain transpose-quantize.
    rht_nk = (
        _get_rht_buffer(tuple(sign_vector), dev.index)
        if apply_rht
        else _get_identity_buffer(dev.index)
    )  # torch buffer (kept alive)
    col_amax_t = col_global_amax.reshape(1)
    row_amax_t = row_global_amax.reshape(1)

    NUM_SMS = _get_num_sms(dev.index)
    GRID = min(NUM_SMS, (N // M_TILE) * (M // (N_TILE * U)))
    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))

    fused = _compile_fused_kernel(dev.index, swizzle, sr, bool(apply_rht))
    fused(
        A.t().unsqueeze(-1),
        rht_nk,
        col_fp4,
        csf_g,
        row_fp4,
        rsf_g,
        row_amax_t,
        col_amax_t,
        sr_rng_t,
        int(M),
        int(N),
        int(GRID),
        stream,
    )

    col_fp4_u8 = col_fp4.view(torch.uint8)  # (N, M//2)
    if compute_rowwise:
        return col_fp4_u8, col_sf, row_fp4.view(torch.uint8), row_sf
    return col_fp4_u8, col_sf, None, None
