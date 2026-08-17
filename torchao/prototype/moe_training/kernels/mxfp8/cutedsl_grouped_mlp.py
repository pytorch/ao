# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 routed-expert grouped-MLP kernels for SM100 (CuTe DSL, public API only).

Three physically fused kernels, one launch each:

* ``launch_grouped_gemm_swiglu_fwd``  -- FC1 ragged grouped GEMM + SwiGLU +
  rowwise (1x32) and columnwise (32x1) MXFP8 RCEIL quantization, plus the BF16
  pre-activation save.
* ``launch_grouped_gemm_dswiglu_bwd`` -- FC2 dgrad ragged grouped GEMM +
  dSwiGLU + the same dual quantization of the FC1 input gradient.
* ``launch_grouped_gemm_wgrad``       -- generic ragged-K grouped weight
  gradient, BF16 output; called once for FC1 and once for FC2.

They share one blockscaled tcgen05 mainloop. Three structural invariants, all
descending from every per-expert row count being a multiple of 128:

* No per-group tensormaps: one host-built static TMA descriptor per operand;
  per-expert selection is an integer coordinate (an L coordinate for the 3-D
  weights, a K-tile index base for wgrad's ragged contraction -- exact
  because ``tile_atom_to_shape_SF``'s K-tile mode is uniform).
* No tile scheduler: forward/backward enumerate all ``[0, R/128)`` M tiles
  and the wgrad grid ``(N/128, K/128, G)`` is fully static; only wgrad's
  K-loop trip count is data-dependent.
* No special tail path: an inactive tile runs with ``k_cnt == 0`` -- no TMA
  loads, a register-zeroed accumulator, and the unmodified epilogue emits the
  zero bytes the contract requires. Stores are never predicated; the one
  epilogue-side gmem input (the backward kernel's saved ``z_bf16``) is loaded
  only when ``k_cnt > 0`` because its tail rows are read-forbidden.

Offset contract (documented caller invariants -- the offset VALUES live on
device and cannot be checked on the host without a synchronization): offsets
are exclusive per-expert end indices, int32, CUDA, 1-D, contiguous,
nondecreasing, every per-expert row count a multiple of 128, and
``offsets[-1] <= R``. The launchers validate all metadata and reject the rest
of the malformed-input space with ``ValueError``; set
``TORCHAO_MXFP8_VALIDATE_OFFSETS=1`` to additionally validate the values on
the host while debugging, at the cost of a D2H copy. There is deliberately no
device-side assertion in the default build: assertions are compiled out of
CuTe DSL kernels unless ``CUTE_DSL_ENABLE_ASSERTIONS=1``, so they cannot serve
as a production guard, and with malformed offsets the wgrad kernel returns a
wrong result rather than faulting -- "it did not crash" is not evidence the
offsets were valid.
"""

import functools
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_validation import (
    _is_fake,
    validate_allocated_rows,
    validate_blocked_scales,
    validate_destination,
    validate_feature_dims,
    validate_group_offsets,
    validate_grouped_operand,
)

__all__ = [
    "launch_grouped_gemm_swiglu_fwd",
    "launch_grouped_gemm_dswiglu_bwd",
    "launch_grouped_gemm_wgrad",
]

# --- Frozen configuration: one tiling, one pipeline shape, one warp assignment.

# MXFP8 scaling block: 32 values share one E8M0 scale.
_SF_VEC_SIZE = 32
# cta_tile_m = 128 is required by CtaGroup.ONE and by the no-partial-M-tile
# argument. cta_tile_k = 128 is pinned: the wgrad kernel selects an expert's K
# range with an integer K-tile index base, exact only because every group
# boundary (a multiple of 128 rows) is a multiple of the K tile.
_CTA_M = 128
_CTA_N = 128
_CTA_K = 128
_MMA_TILER = (_CTA_M, _CTA_N, _CTA_K)
# One stage: A tile + B tile (E4M3) + one 512-byte scale atom per operand.
# This is also the exact tx_count the TMA pipeline barrier must expect.
_AB_STAGE_BYTES = _CTA_M * _CTA_K + _CTA_N * _CTA_K + 2 * (128 * (_CTA_K // 32))
_NUM_AB_STAGE = 6
_NUM_ACC_STAGE = 1
# Warp 0 loads (TMA), warp 1 issues the MMA, warps 4-7 run the epilogue, warps
# 2-3 idle. The epilogue must start on a warp quad (warp id multiple of 4):
# tcgen05.ld selects its TMEM datapath sub-partition from the physical warp
# id, and a misaligned epilogue block would read every 128-row tile with its
# 32-row groups rotated.
_THREADS = 256
_TMA_WARP_ID = 0
_MMA_WARP_ID = 1
_FIRST_EPI_WARP = 4
_NUM_EPI_THREADS = 128
_FIRST_EPI_THREAD = 32 * _FIRST_EPI_WARP
# Named barrier ids (0 is left free for the DSL's own use).
_EPI_FINAL_BARRIER_ID = 1
_TMEM_ALLOC_BARRIER_ID = 2
_EPI_STAGE_BARRIER_ID = 3
# sm_100 usable dynamic shared memory per CTA, (228 - 1) KiB.
_SMEM_CAPACITY_BYTES = 232448
# The TMEM allocator requires a power-of-two multiple of 32 columns; shared
# memory already pins us to one CTA per SM, so taking the whole array is free.
_TMEM_TOTAL_COLS = 512


@dataclass(frozen=True)
class _KernelConfig:
    """Per-kernel trace-time constants (everything else is module-frozen)."""

    # Accumulator columns handed to the epilogue per subtile. 64 for the
    # forward (an adjacent gate/up accumulator pair per output column, so 64
    # accumulator columns are 32 output columns = one 1x32 block per row), 32
    # for the backward and for wgrad.
    epi_n_acc: int
    # False: offsets partition the GEMM M axis (forward/backward). True: they
    # partition the contraction (wgrad).
    ragged_k: bool
    # Columns of the [128, cols] BF16 epilogue staging tile (0 = no staging).
    # Padded to an odd count so columnwise reads don't serialize on banks.
    epi_smem_cols: int


_SWIGLU_FWD_CONFIG = _KernelConfig(epi_n_acc=64, ragged_k=False, epi_smem_cols=33)
_DSWIGLU_BWD_CONFIG = _KernelConfig(epi_n_acc=32, ragged_k=False, epi_smem_cols=65)
_WGRAD_CONFIG = _KernelConfig(epi_n_acc=32, ragged_k=True, epi_smem_cols=0)


# --- Host-side operand views: pure torch restrides into the (MN, K, L) GEMM
# domain, K contiguous.


def activation_gemm_view(t: torch.Tensor) -> torch.Tensor:
    """``[MN, K]`` K-contiguous -> ``(MN, K, 1)`` with a defined batch stride."""
    mn, k = t.shape
    if t.stride() != (k, 1):
        raise ValueError(
            f"GEMM operand must be K-contiguous with stride {(k, 1)}, got {t.stride()}"
        )
    return torch.as_strided(t, (mn, k, 1), (k, 1, mn * k))


def weight_gemm_view(w: torch.Tensor) -> torch.Tensor:
    """``[G, K, N]`` stride ``(K*N, 1, K)`` -> ``(N, K, G)``: expert becomes L."""
    g, k, n = w.shape
    if w.stride() != (k * n, 1, k):
        raise ValueError(
            f"grouped weight must have stride {(k * n, 1, k)}, got {w.stride()}"
        )
    return w.permute(2, 1, 0)


@dataclass
class TileCoords:
    """Per-CTA tile description handed to the epilogue. All fields CTA-uniform.

    ``k_cnt == 0`` marks both an inactive tail tile (ragged M) and a
    zero-token expert (ragged K); the accumulator arrives zeroed and the
    epilogue must store unconditionally.
    """

    expert: Int32
    row_base: Int32
    col_base: Int32
    k_cnt: Int32


def _s2t_copy_and_partition(sSF: cute.Tensor, tSF: cute.Tensor):
    """SMEM -> TMEM scale-factor copy, issued once per K tile from the MMA warp.

    ``Cp4x32x128bOp`` must be issued as a plain ``cute.copy`` -- the DSL
    inserts the single-thread election itself, and wrapping it in
    ``elect_one()`` deadlocks.
    """
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sSF.element_type
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t, thr_copy_s2t.partition_S(tCsSF_compact)
    )
    tCtSF_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_s2t, tCtSF_s2t


# --- Quantization math: public conversions only; every step mirrors the torchao
# reference (`to_mx(..., RCEIL)` + `to_blocked`) so the fused outputs are
# byte-identical to the standalone quantizers on the same BF16 input.


@cute.jit
def _blocked_scale_idx(row: Int32, scale_col: Int32, ncb: Int32) -> Int32:
    """Flat byte index in the tcgen05 blocked (128x4) scale layout.

    The logical ``[rows, cols]`` scale matrix is stored as 512-byte tiles of
    128 rows x 4 scale columns, tiles ordered ``row_block * ncb + col_block``
    with ``ncb = ceil_div(cols, 4)`` -- torchao ``to_blocked``, whole-matrix.
    Coordinates are ABSOLUTE. For columnwise scales pass transposed
    coordinates (feature index as ``row``, 32-row block index as ``col``).
    """
    return (
        ((row >> 7) * ncb + (scale_col >> 2)) * Int32(512)
        + (row & Int32(31)) * Int32(16)
        + ((row >> 5) & Int32(3)) * Int32(4)
        + (scale_col & Int32(3))
    )


@cute.jit
def _store_frag(dst: cute.Tensor, elem_offset: Int32, frag: cute.Tensor):
    """Store a register fragment as one contiguous run of ``dst`` elements.

    Callers guarantee the element offset is at least 16-byte aligned relative
    to the (validated, 32-byte-aligned) base, so the copy vectorizes.
    """
    cute.autovec_copy(
        frag,
        cute.make_tensor(
            (dst.iterator + elem_offset).align(16),
            cute.make_layout(cute.size(frag)),
        ),
    )


@cute.jit
def _quant_block_from_smem(
    sEpi: cute.Tensor,
    base_row: Int32,
    base_col: Int32,
    q_dst: cute.Tensor,
    q_offset: Int32,
    sf_dst: cute.Tensor,
    sf_idx: Int32,
    COLWISE: cutlass.Constexpr,
):
    """Quantize one 32-value MX block read from the BF16 staging tile
    (``sEpi[base_row, base_col + i]`` rowwise, ``sEpi[base_row + i, base_col]``
    columnwise):

    * amax: NaN-propagating |max| chain, so a NaN element invalidates the
      block exactly like the torchao reference.
    * scale: ``descale = amax / 448``; the public Float32 -> Float8E8M0FNU
      conversion rounds toward +inf (RCEIL) by construction. A non-finite
      amax (Inf would otherwise clamp to byte 254) is overridden to byte 255.
    * reciprocal, in the E8M0 byte domain: ``254 - byte`` reinterpreted as
      E8M0 and widened exactly to f32. Byte 0 (zero/tiny block) descales by
      2^127; byte 255 gives a NaN reciprocal so every element of an
      invalidated block quantizes to the E4M3 NaN code.
    * qdata: one f32 multiply per element, then the public saturating-RNE
      Float32 -> Float8E4M3FN conversion (byte-identical to torch's cast).

    The 32 qdata bytes are one contiguous ``q_dst`` run in both orientations,
    stored vectorized; the scale byte is stored individually at ``sf_idx``.
    """
    vals = []
    for i in cutlass.range_constexpr(_SF_VEC_SIZE):
        if cutlass.const_expr(COLWISE):
            v = sEpi[base_row + Int32(i), base_col]
        else:
            v = sEpi[base_row, base_col + Int32(i)]
        vals.append(Float32(v))

    amax = cute.arch.fmax(vals[0], vals[1], abs=True, nan=True)
    for i in cutlass.range_constexpr(2, _SF_VEC_SIZE):
        amax = cute.arch.fmax(amax, vals[i], abs=True, nan=True)

    scale_byte = Int32(
        cutlass.Float8E8M0FNU(amax / Float32(448.0)).bitcast(cutlass.Int8)
    ) & Int32(0xFF)
    amax_bits = Float32(amax).bitcast(Int32)
    if (amax_bits & Int32(0x7F800000)) == Int32(0x7F800000):
        scale_byte = Int32(255)

    recip_byte = (Int32(254) - scale_byte) & Int32(0xFF)
    recip = Float32(cutlass.Uint8(recip_byte).bitcast(cutlass.Float8E8M0FNU))

    qfrag = cute.make_rmem_tensor(cute.make_layout(_SF_VEC_SIZE), cutlass.Float8E4M3FN)
    for i in cutlass.range_constexpr(_SF_VEC_SIZE):
        qfrag[i] = cutlass.Float8E4M3FN(vals[i] * recip)
    _store_frag(q_dst, q_offset, qfrag)
    sf_dst[sf_idx] = cutlass.Uint8(scale_byte)


@cute.jit
def _epilogue_column_run(
    tTR_cAcc_s, num_acc: cutlass.Constexpr, even: cutlass.Constexpr
):
    """Trace-time proof that a thread's fragment is one contiguous column run.

    Returns the (static) first column. The column coordinates fold to Python
    ints at trace time; requiring one even-based, contiguous, increasing run
    also proves the fragment covers a single row (a two-row fragment would
    repeat columns), so this CHECKS the layout the epilogues need instead of
    assuming a physical thread-to-row mapping.
    """
    cols = []
    for v in cutlass.range_constexpr(num_acc):
        cols.append(tTR_cAcc_s[v][1])
    first = cols[0]
    if cutlass.const_expr(
        not all(isinstance(c, int) for c in cols)
        or tuple(cols) != tuple(range(first, first + num_acc))
        or (even and first % 2 != 0)
    ):
        raise ValueError(
            "the epilogue needs one contiguous, increasing"
            + (", even-based" if even else "")
            + f" column run per thread, but tTR_cAcc gave {cols}"
        )
    return first


# --- Epilogues: called once per subtile by all 128 epilogue threads with a
# CTA-uniform k_cnt; stores are never predicated (a tail tile's zeroed
# accumulator produces exactly the zero bytes the contract requires).


@cute.jit
def _wgrad_epilogue(
    tTR_rAcc,
    tTR_cAcc_s,
    epi_tidx,
    tile: TileCoords,
    sEpi,
    out,
    R: cutlass.Constexpr,
    N: cutlass.Constexpr,
):
    """Round the FP32 accumulator subtile to BF16 and store it (no quant).

    ``out`` is ``(mDw,)`` with ``mDw`` the ``(N, K, G)`` view of the
    contiguous ``[G, N, K]`` destination.
    """
    num_acc = cutlass.const_expr(cute.size(tTR_rAcc))
    frag_col = _epilogue_column_run(tTR_cAcc_s, num_acc, even=False)

    frag = cute.make_rmem_tensor(cute.make_layout(num_acc), cutlass.BFloat16)
    for v in cutlass.range_constexpr(num_acc):
        frag[v] = cutlass.BFloat16(tTR_rAcc[v])

    gDw = out[0]
    strides = gDw.stride
    if cutlass.const_expr(strides[1] != 1):
        raise ValueError(
            f"the wgrad epilogue stores a contiguous run along K, so dw's K "
            f"stride must be 1, got layout {gDw.layout}"
        )
    row = tile.row_base + tTR_cAcc_s[0][0]
    elem = (
        row * Int32(strides[0])
        + (tile.col_base + Int32(frag_col))
        + tile.expert * Int32(strides[2])
    )
    _store_frag(gDw, elem, frag)


@cute.jit
def _swiglu_fwd_epilogue(
    tTR_rAcc,
    tTR_cAcc_s,
    epi_tidx,
    tile: TileCoords,
    sEpi,
    out,
    R: cutlass.Constexpr,
    N: cutlass.Constexpr,
):
    """SwiGLU + BF16 pre-activation save + dual MXFP8 quantization.

    ``out`` = flat views ``(z [R*2F] bf16, h_row_q [R*F] e4m3,
    h_row_sf uint8, h_col_q [F*R] e4m3 in column-major storage order,
    h_col_sf uint8)``; ``N == 2F`` is the GEMM (and z) column count. Contract:
    the accumulator is rounded to BF16 first (that IS z), SwiGLU is evaluated
    once from the rounded values, h is rounded to BF16 once, and both
    quantizers consume the same staged BF16 h.
    """
    num_acc = cutlass.const_expr(cute.size(tTR_rAcc))  # 64
    half = cutlass.const_expr(num_acc // 2)  # 32 h columns per subtile
    F = cutlass.const_expr(N // 2)
    frag_col = _epilogue_column_run(tTR_cAcc_s, num_acc, even=True)

    mZ = out[0]
    mHrowQ = out[1]
    mHrowSF = out[2]
    mHcolQ = out[3]
    mHcolSF = out[4]

    # ---- stage 1: z store + h compute into the staging tile --------------
    lrow = tTR_cAcc_s[0][0]  # CTA-tile-local row of this thread's fragment
    row_g = tile.row_base + lrow
    zfrag = cute.make_rmem_tensor(cute.make_layout(num_acc), cutlass.BFloat16)
    for v in cutlass.range_constexpr(num_acc):
        zfrag[v] = cutlass.BFloat16(tTR_rAcc[v])
    _store_frag(mZ, row_g * Int32(N) + tile.col_base + Int32(frag_col), zfrag)

    for j in cutlass.range_constexpr(half):
        gate = Float32(zfrag[2 * j])
        up = Float32(zfrag[2 * j + 1])
        # sigmoid composed exactly like torch's float32 sigmoid (default-mode
        # exp plus a true divide); measured bit-identical to torch.sigmoid.
        sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(Float32(0.0) - gate))
        sEpi[lrow, Int32(j)] = cutlass.BFloat16((gate * sig) * up)

    cute.arch.barrier(
        barrier_id=_EPI_STAGE_BARRIER_ID, number_of_threads=_NUM_EPI_THREADS
    )

    # ---- stage 2: dual quantization off the staging tile -----------------
    # Global h column base of this subtile; frag_col is static, 64 per
    # subtile, so hbase is divisible by 32.
    hbase = (tile.col_base + Int32(frag_col)) >> 1
    ncb_row = cutlass.const_expr((F // _SF_VEC_SIZE + 3) // 4)
    ncb_col = cutlass.const_expr((R // _SF_VEC_SIZE + 3) // 4)

    # Rowwise 1x32: one block per thread (row = epi_tidx of the staging tile).
    q_row = tile.row_base + epi_tidx
    _quant_block_from_smem(
        sEpi,
        epi_tidx,
        Int32(0),
        mHrowQ,
        q_row * Int32(F) + hbase,
        mHrowSF,
        _blocked_scale_idx(
            q_row, (tile.col_base + Int32(frag_col)) >> 6, Int32(ncb_row)
        ),
        COLWISE=False,
    )

    # Columnwise 32x1: 32 columns x 4 row-blocks = one block per thread.
    col_l = epi_tidx & Int32(31)
    blk = epi_tidx >> 5
    col_g = hbase + col_l
    _quant_block_from_smem(
        sEpi,
        blk * Int32(32),
        col_l,
        mHcolQ,
        col_g * Int32(R) + tile.row_base + blk * Int32(32),
        mHcolSF,
        _blocked_scale_idx(col_g, (tile.row_base >> 5) + blk, Int32(ncb_col)),
        COLWISE=True,
    )

    # The next subtile reuses the staging tile.
    cute.arch.barrier(
        barrier_id=_EPI_STAGE_BARRIER_ID, number_of_threads=_NUM_EPI_THREADS
    )


@cute.jit
def _dswiglu_bwd_epilogue(
    tTR_rAcc,
    tTR_cAcc_s,
    epi_tidx,
    tile: TileCoords,
    sEpi,
    out,
    R: cutlass.Constexpr,
    N: cutlass.Constexpr,
):
    """dSwiGLU from the saved z + dual MXFP8 quantization of dz.

    ``out`` = ``(z [R*2F] bf16 INPUT, dz_row_q [R*2F] e4m3, dz_row_sf uint8,
    dz_col_q [2F*R] e4m3 column-major storage, dz_col_sf uint8)``; ``N == F``
    is the dgrad GEMM column count; dz has 2F element-interleaved columns.
    Contract: dh is rounded to BF16 first; gate/up come from the saved BF16 z;
    dgate/dup are each rounded to BF16 before interleaving; both quantizers
    consume the same staged BF16 dz. The z load is predicated on ``k_cnt`` --
    tail rows of z are read-forbidden and contribute zeros.
    """
    num_acc = cutlass.const_expr(cute.size(tTR_rAcc))  # 32
    two_f = cutlass.const_expr(2 * N)
    frag_col = _epilogue_column_run(tTR_cAcc_s, num_acc, even=False)

    mZ = out[0]
    mDzRowQ = out[1]
    mDzRowSF = out[2]
    mDzColQ = out[3]
    mDzColSF = out[4]

    # ---- stage 1: z load (predicated) + dSwiGLU into the staging tile ----
    lrow = tTR_cAcc_s[0][0]
    row_g = tile.row_base + lrow
    # dz columns covered by this subtile: [dzbase, dzbase + 64).
    dzbase = (tile.col_base + Int32(frag_col)) * Int32(2)

    zfrag = cute.make_rmem_tensor(cute.make_layout(2 * num_acc), cutlass.BFloat16)
    if tile.k_cnt > Int32(0):
        cute.autovec_copy(
            cute.make_tensor(
                (mZ.iterator + (row_g * Int32(two_f) + dzbase)).align(16),
                cute.make_layout(2 * num_acc),
            ),
            zfrag,
        )
    else:
        for i in cutlass.range_constexpr(2 * num_acc):
            zfrag[i] = cutlass.BFloat16(0.0)

    for j in cutlass.range_constexpr(num_acc):
        gate = Float32(zfrag[2 * j])
        up = Float32(zfrag[2 * j + 1])
        dh = Float32(cutlass.BFloat16(tTR_rAcc[j]))
        sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(Float32(0.0) - gate))
        silu = gate * sig
        dsilu = sig * (Float32(1.0) + gate * (Float32(1.0) - sig))
        sEpi[lrow, Int32(2 * j)] = cutlass.BFloat16((dh * up) * dsilu)
        sEpi[lrow, Int32(2 * j + 1)] = cutlass.BFloat16(dh * silu)

    cute.arch.barrier(
        barrier_id=_EPI_STAGE_BARRIER_ID, number_of_threads=_NUM_EPI_THREADS
    )

    # ---- stage 2: dual quantization off the staging tile -----------------
    ncb_row = cutlass.const_expr((two_f // _SF_VEC_SIZE + 3) // 4)
    ncb_col = cutlass.const_expr((R // _SF_VEC_SIZE + 3) // 4)

    # Rowwise 1x32: 128 rows x 2 blocks = two tasks per thread.
    q_row = tile.row_base + epi_tidx
    for blk in cutlass.range_constexpr(2):
        _quant_block_from_smem(
            sEpi,
            epi_tidx,
            Int32(blk * _SF_VEC_SIZE),
            mDzRowQ,
            q_row * Int32(two_f) + dzbase + Int32(blk * _SF_VEC_SIZE),
            mDzRowSF,
            _blocked_scale_idx(q_row, (dzbase >> 5) + Int32(blk), Int32(ncb_row)),
            COLWISE=False,
        )

    # Columnwise 32x1: 64 columns x 4 row-blocks = two tasks per thread.
    for k in cutlass.range_constexpr(2):
        task = epi_tidx + Int32(128 * k)
        col_l = task & Int32(63)
        blk = task >> 6
        col_g = dzbase + col_l
        _quant_block_from_smem(
            sEpi,
            blk * Int32(32),
            col_l,
            mDzColQ,
            col_g * Int32(R) + tile.row_base + blk * Int32(32),
            mDzColSF,
            _blocked_scale_idx(col_g, (tile.row_base >> 5) + blk, Int32(ncb_col)),
            COLWISE=True,
        )

    cute.arch.barrier(
        barrier_id=_EPI_STAGE_BARRIER_ID, number_of_threads=_NUM_EPI_THREADS
    )


# --- The shared kernel and launch builder.


@cute.kernel
def _grouped_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB: cute.Tensor,
    offs: cute.Tensor,
    out,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
    cfg: cutlass.Constexpr,
    storage_type: cutlass.Constexpr,
    EPILOGUE: cutlass.Constexpr,
):
    """One CTA computes one 128 x 128 output tile. Warps: 0 TMA, 1 MMA,
    4-7 epilogue, 2-3 idle."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, bidz = cute.arch.block_idx()

    num_groups = cutlass.const_expr(cute.size(offs, mode=[0]))
    num_k_tiles_full = cutlass.const_expr(cute.size(mA, mode=[1]) // _CTA_K)
    gemm_m = cutlass.const_expr(cute.size(mA, mode=[0]))
    gemm_n = cutlass.const_expr(cute.size(mB, mode=[0]))

    # ------------------------------------------------------------------
    # Tile coordinates. No scheduler: the grid IS the tile enumeration.
    # ------------------------------------------------------------------
    tile_m = Int32(bidx)
    tile_n = Int32(bidy)
    if cutlass.const_expr(not cfg.ragged_k):
        row_base = tile_m * _CTA_M
        # Unrolled G-way scan: the owning expert is the number of groups that
        # end at or before this tile's row base; a zero-token expert is never
        # selected since its end equals its start.
        expert = Int32(0)
        for g in cutlass.range_constexpr(num_groups - 1):
            expert += Int32(offs[g] <= row_base)
        is_active = Int32(offs[num_groups - 1] > row_base)
        k_base = Int32(0)
        k_cnt = is_active * Int32(num_k_tiles_full)
        l_b = expert
    else:
        expert = Int32(bidz)
        row_base = tile_m * _CTA_M
        prev = offs[cutlass.max(expert - Int32(1), Int32(0))] * Int32(expert > Int32(0))
        # Exact, not a ceil: every group boundary is a multiple of cta_tile_k.
        k_base = prev // _CTA_K
        # Clamp: nonmonotone offsets (undefined behavior per the contract, and
        # only device-checkable) would otherwise make k_cnt negative -- the
        # loops still run zero trips, but the epilogue would read TMEM the MMA
        # never wrote. Clamped, a malformed expert degrades to an all-zero
        # slice instead.
        k_cnt = cutlass.max((offs[expert] - prev) // _CTA_K, Int32(0))
        l_b = Int32(0)
    tile = TileCoords(
        expert=expert,
        row_base=row_base,
        col_base=tile_n * _CTA_N,
        k_cnt=k_cnt,
    )

    # ------------------------------------------------------------------
    # Shared memory and pipelines
    # ------------------------------------------------------------------
    smem = utils.SmemAllocator()
    storage = smem.allocate(storage_type)

    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
    sEpi = None
    if cutlass.const_expr(cfg.epi_smem_cols > 0):
        sEpi = storage.sEpi.get_tensor(
            cute.make_layout((_CTA_M, cfg.epi_smem_cols), stride=(cfg.epi_smem_cols, 1))
        )

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
    )

    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar.data_ptr(),
        num_stages=_NUM_AB_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        # The EXACT byte count of the four TMA copies of one stage: too small
        # and the MMA consumes a partially arrived stage.
        tx_count=_AB_STAGE_BYTES,
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar.data_ptr(),
        num_stages=_NUM_ACC_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, _NUM_EPI_THREADS
        ),
        cta_layout_vmnk=cluster_layout_vmnk,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=_TMEM_ALLOC_BARRIER_ID,
        num_threads=32 * 5,  # the MMA warp joins the four epilogue warps
    )
    epilogue_barrier = pipeline.NamedBarrier(
        barrier_id=_EPI_FINAL_BARRIER_ID,
        num_threads=_NUM_EPI_THREADS,
    )
    tmem_alloc = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=_FIRST_EPI_WARP,
        is_two_cta=False,
    )

    # ------------------------------------------------------------------
    # Tile the global tensors. One static descriptor per operand; the expert
    # is an L coordinate (ragged M) or a K-tile index base (ragged K).
    # ------------------------------------------------------------------
    gA = cute.local_tile(
        mA, cute.slice_(_MMA_TILER, (None, 0, None)), (None, None, None)
    )
    gB = cute.local_tile(
        mB, cute.slice_(_MMA_TILER, (0, None, None)), (None, None, None)
    )
    gSFA = cute.local_tile(
        mSFA, cute.slice_(_MMA_TILER, (None, 0, None)), (None, None, None)
    )
    gSFB = cute.local_tile(
        mSFB, cute.slice_(_MMA_TILER, (0, None, None)), (None, None, None)
    )

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = thr_mma.partition_B(gSFB)

    trivial_cta_layout = cute.make_layout(1)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        trivial_cta_layout,
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        trivial_cta_layout,
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        trivial_cta_layout,
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        trivial_cta_layout,
        cute.group_modes(sSFB, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    tAgA_slice = tAgA[(None, tile_m, None, 0)]
    tBgB_slice = tBgB[(None, tile_n, None, l_b)]
    tAgSFA_slice = tAgSFA[(None, tile_m, None, 0)]
    tBgSFB_slice = tBgSFB[(None, tile_n, None, l_b)]

    acc_shape = tiled_mma.partition_shape_C(_MMA_TILER[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, _NUM_ACC_STAGE))

    # ------------------------------------------------------------------
    # Warp 0: TMA producer
    # ------------------------------------------------------------------
    if warp_idx == _TMA_WARP_ID:
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, _NUM_AB_STAGE
        )
        for _ in cutlass.range(0, k_cnt, 1, unroll=1):
            ab_pipeline.producer_acquire(ab_producer_state)
            k_idx = k_base + ab_producer_state.count
            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
            cute.copy(
                tma_atom_a,
                tAgA_slice[(None, k_idx)],
                tAsA[(None, ab_producer_state.index)],
                tma_bar_ptr=bar,
            )
            cute.copy(
                tma_atom_b,
                tBgB_slice[(None, k_idx)],
                tBsB[(None, ab_producer_state.index)],
                tma_bar_ptr=bar,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA_slice[(None, k_idx)],
                tAsSFA[(None, ab_producer_state.index)],
                tma_bar_ptr=bar,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB_slice[(None, k_idx)],
                tBsSFB[(None, ab_producer_state.index)],
                tma_bar_ptr=bar,
            )
            ab_producer_state.advance()
        ab_pipeline.producer_tail(ab_producer_state)

    # ------------------------------------------------------------------
    # Warp 1: MMA
    # ------------------------------------------------------------------
    if warp_idx == _MMA_WARP_ID:
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # The MMA warp joins the allocation barrier but must never allocate.
        tmem_alloc.wait_for_alloc()
        acc_tmem_ptr = tmem_alloc.retrieve_ptr(Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
            dtype=sSFA.element_type,
        )
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                _MMA_TILER,
                _SF_VEC_SIZE,
                cute.slice_(sfa_smem_layout, (None, None, None, 0)),
            ),
        )
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sSFB.element_type,
        )
        tCtSFB = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                _MMA_TILER,
                _SF_VEC_SIZE,
                cute.slice_(sfb_smem_layout, (None, None, None, 0)),
            ),
        )
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_copy_and_partition(sSFA, tCtSFA)
        s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_copy_and_partition(sSFB, tCtSFB)

        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _NUM_AB_STAGE
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, _NUM_ACC_STAGE
        )
        tCtAcc = tCtAcc_base[(None, None, None, 0)]

        # Acquire and commit unconditionally, including when k_cnt == 0, so
        # the accumulator handoff barrier stays balanced on tail tiles.
        acc_pipeline.producer_acquire(acc_producer_state)
        for k_tile in cutlass.range(0, k_cnt, 1, unroll=1):
            ab_pipeline.consumer_wait(ab_consumer_state)
            stage_crd = (None, None, None, None, ab_consumer_state.index)
            cute.copy(s2t_sfa, tCsSFA_s2t[stage_crd], tCtSFA_s2t)
            cute.copy(s2t_sfb, tCsSFB_s2t[stage_crd], tCtSFB_s2t)
            # ACCUMULATE=False on the first K tile is what zeroes the
            # accumulator; there is no separate TMEM clear.
            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
            mma_crd = (None, None, None, ab_consumer_state.index)
            cute.gemm(
                tiled_mma,
                tCtAcc,
                [tCrA[mma_crd], tCtSFA],
                [tCrB[mma_crd], tCtSFB],
                tCtAcc,
            )
            ab_pipeline.consumer_release(ab_consumer_state)
            ab_consumer_state.advance()
        acc_pipeline.producer_commit(acc_producer_state)

    # ------------------------------------------------------------------
    # Warps 4-7: epilogue
    # ------------------------------------------------------------------
    if warp_idx >= _FIRST_EPI_WARP:
        tmem_alloc.allocate(_TMEM_TOTAL_COLS)
        tmem_alloc.wait_for_alloc()
        acc_tmem_ptr = tmem_alloc.retrieve_ptr(Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        epi_tidx = tidx - _FIRST_EPI_THREAD
        # TMEM -> register handoff. tTR_cAcc carries each register's (row, col)
        # coordinate in the CTA tile; every epilogue index is derived from it,
        # not from an assumed thread-to-row mapping. The d element type stays
        # Float32: an 8-bit d would steer get_tmem_load_op into layouts shaped
        # for a direct FP8 TMA store.
        epi_tile = (_CTA_M, cfg.epi_n_acc)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            _MMA_TILER, LayoutEnum.ROW_MAJOR, Float32, Float32, epi_tile, False
        )
        tAcc_mn = tCtAcc_base[((None, None), 0, 0, 0)]
        tAcc_epi = cute.flat_divide(tAcc_mn, epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        cAcc_epi = cute.flat_divide(
            cute.make_identity_tensor((_CTA_M, _CTA_N)), epi_tile
        )
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_cAcc[(None, None, None, 0, 0)].shape, Float32
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _NUM_ACC_STAGE
        )
        acc_pipeline.consumer_wait(acc_consumer_state)

        for s in cutlass.range_constexpr(_CTA_N // cfg.epi_n_acc):
            if k_cnt == Int32(0):
                # Tail tile or zero-token expert: nothing was accumulated, so
                # the fragment is zeroed here and the epilogue runs unchanged.
                for v in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    tTR_rAcc[v] = Float32(0.0)
            else:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, s)], tTR_rAcc)
            EPILOGUE(
                tTR_rAcc,
                tTR_cAcc[(None, None, None, 0, s)],
                epi_tidx,
                tile,
                sEpi,
                out,
                gemm_m,
                gemm_n,
            )

        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)
        tmem_alloc.relinquish_alloc_permit()
        epilogue_barrier.arrive_and_wait()
        tmem_alloc.free(acc_tmem_ptr)


@cute.jit
def _launch_grouped_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    sfa_flat: cute.Tensor,
    sfb_flat: cute.Tensor,
    offs: cute.Tensor,
    out,
    stream,
    cfg: cutlass.Constexpr,
    EPILOGUE: cutlass.Constexpr,
):
    """Build the four static TMA descriptors and launch. Grid is data-independent.

    This is a trace body: calling it directly retraces the kernel on every
    invocation. The public launchers ``cute.compile`` it once per shape key,
    passing ``cfg`` and ``EPILOGUE`` as trailing Constexpr args, then call the
    compiled executor with the runtime args only.
    """
    a_dtype = mA.element_type
    b_dtype = mB.element_type
    sf_dtype = cutlass.Float8E8M0FNU

    gemm_m = cutlass.const_expr(cute.size(mA, mode=[0]))
    gemm_k = cutlass.const_expr(cute.size(mA, mode=[1]))
    gemm_n = cutlass.const_expr(cute.size(mB, mode=[0]))
    l_a = cutlass.const_expr(cute.size(mA, mode=[2]))
    l_b = cutlass.const_expr(cute.size(mB, mode=[2]))
    num_groups = cutlass.const_expr(cute.size(offs, mode=[0]))

    if cutlass.const_expr(
        gemm_m % _CTA_M != 0 or gemm_n % _CTA_N != 0 or gemm_k % _CTA_K != 0
    ):
        raise ValueError(
            f"GEMM extents ({gemm_m}, {gemm_n}, {gemm_k}) must be multiples of "
            f"({_CTA_M}, {_CTA_N}, {_CTA_K})"
        )

    # Retile the flat blocked E8M0 buffers into the GEMM-domain SF layout. The
    # buffers travel flat by ABI and may arrive as raw uint8, so the pointer is
    # recast: the MMA rejects a scale operand that is not E8M0.
    mSFA = cute.make_tensor(
        cute.recast_ptr(sfa_flat.iterator, dtype=cutlass.Float8E8M0FNU),
        blockscaled_utils.tile_atom_to_shape_SF((gemm_m, gemm_k, l_a), _SF_VEC_SIZE),
    )
    mSFB = cute.make_tensor(
        cute.recast_ptr(sfb_flat.iterator, dtype=cutlass.Float8E8M0FNU),
        blockscaled_utils.tile_atom_to_shape_SF((gemm_n, gemm_k, l_b), _SF_VEC_SIZE),
    )

    # The one blockscaled tiled MMA, K-major on both operands, FP32 acc.
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        sf_dtype,
        _SF_VEC_SIZE,
        tcgen05.CtaGroup.ONE,
        (_CTA_M, _CTA_N),
    )
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
    )

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, _MMA_TILER, a_dtype, _NUM_AB_STAGE
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, _MMA_TILER, b_dtype, _NUM_AB_STAGE
    )
    sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, _MMA_TILER, _SF_VEC_SIZE, _NUM_AB_STAGE
    )
    sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, _MMA_TILER, _SF_VEC_SIZE, _NUM_AB_STAGE
    )

    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id),
        mA,
        cute.slice_(a_smem_layout, (None, None, None, 0)),
        _MMA_TILER,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id),
        mB,
        cute.slice_(b_smem_layout, (None, None, None, 0)),
        _MMA_TILER,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # The 512-byte scale atom is contiguous; TMA moves it as 8-byte elements.
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id),
        mSFA,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
        _MMA_TILER,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Uint64,
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), tiled_mma.thr_id),
        mSFB,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
        _MMA_TILER,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Uint64,
    )

    @cute.struct
    class SharedStorage:
        # The *_empty ranges are live storage: each Pipeline*.create consumes
        # 2 x num_stages barriers starting at the *_full pointer.
        ab_full_mbar: cute.struct.MemRange[cutlass.Int64, _NUM_AB_STAGE]
        ab_empty_mbar: cute.struct.MemRange[cutlass.Int64, _NUM_AB_STAGE]
        acc_full_mbar: cute.struct.MemRange[cutlass.Int64, _NUM_ACC_STAGE]
        acc_empty_mbar: cute.struct.MemRange[cutlass.Int64, _NUM_ACC_STAGE]
        tmem_holding_buf: cutlass.Int32
        # A zero-size struct field would be degenerate; keep a tiny slab.
        sEpi: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, max(_CTA_M * cfg.epi_smem_cols, 8)],
            128,
        ]
        sA: cute.struct.Align[
            cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout.outer)], 1024
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout.outer)], 1024
        ]
        sSFA: cute.struct.Align[
            cute.struct.MemRange[sf_dtype, cute.cosize(sfa_smem_layout)], 1024
        ]
        sSFB: cute.struct.Align[
            cute.struct.MemRange[sf_dtype, cute.cosize(sfb_smem_layout)], 1024
        ]

    smem_bytes = cutlass.const_expr(SharedStorage.size_in_bytes())
    if cutlass.const_expr(smem_bytes > _SMEM_CAPACITY_BYTES):
        raise ValueError(
            f"shared memory request {smem_bytes} B exceeds the sm_100 capacity "
            f"{_SMEM_CAPACITY_BYTES} B"
        )

    if cutlass.const_expr(not cfg.ragged_k):
        grid = (gemm_m // _CTA_M, gemm_n // _CTA_N, 1)
    else:
        grid = (gemm_m // _CTA_M, gemm_n // _CTA_N, num_groups)

    _grouped_gemm_kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_sfa,
        tma_tensor_sfa,
        tma_atom_sfb,
        tma_tensor_sfb,
        offs,
        out,
        a_smem_layout,
        b_smem_layout,
        sfa_smem_layout,
        sfb_smem_layout,
        cfg,
        SharedStorage,
        EPILOGUE,
    ).launch(
        grid=grid,
        block=(_THREADS, 1, 1),
        cluster=(1, 1, 1),
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _executor_slot(key: tuple) -> list:
    """One memo slot per (kernel, shape, device, dtype, DSL version) key; the
    trace needs real tensors, so the first caller compiles and fills it once."""
    return []


def _cache_key(kind: str, dims: tuple, tensors: tuple, device) -> tuple:
    return (
        kind,
        dims,
        tuple(str(t.dtype) for t in tensors),
        device.index,
        torch.cuda.get_device_capability(device),
        cutlass.__version__,
    )


def _common_launch_checks(name: str, device, tensors, groups: int):
    """Support and safety gates shared by the three launchers."""
    if any(_is_fake(t) for t in tensors):
        raise ValueError(
            f"{name} cannot run on fake/meta tensors; call the corresponding "
            "torchao::* op instead, whose register_fake handles tracing"
        )
    if groups < 1:
        raise ValueError(f"G must be at least 1, got {groups}")
    if device.type != "cuda":
        raise ValueError(f"{name} requires CUDA tensors, got device {device}")
    major, _minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise NotImplementedError(
            f"{name} requires an SM100-class GPU (compute capability 10.x), "
            f"got {torch.cuda.get_device_capability(device)}"
        )


def _stream_for(device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(int(torch.cuda.current_stream(device).cuda_stream))


def _check_sf_pointer_alignment(name: str, buf: torch.Tensor):
    # Scale buffers feed TMA descriptors (inputs) or vectorized stores; a
    # contiguous view with a storage offset can be 2-byte aligned, and only
    # the launcher promises the alignment.
    if buf.data_ptr() % 32 != 0:
        raise ValueError(
            f"{name} must be 32-byte aligned, but its data pointer is "
            f"{buf.data_ptr() % 32} bytes past an aligned address"
        )


_E4M3 = torch.float8_e4m3fn
_BF16 = torch.bfloat16


def launch_grouped_gemm_swiglu_fwd(
    x_q, x_sf, w13_t_q, w13_t_sf, offsets, z_bf16, h_row_q, h_row_sf, h_col_q, h_col_sf
):
    """FC1 grouped GEMM + SwiGLU + dual MXFP8 quantization, one kernel launch.

    Inputs are prequantized: ``x_q`` E4M3 ``[R, D]`` row-major with blocked
    ``x_sf``; ``w13_t_q`` E4M3 ``[G, D, 2F]`` stride ``(2F*D, 1, D)`` with
    per-expert blocked ``w13_t_sf`` (the 2F axis is element-interleaved
    gate/up). Destinations are caller-allocated: ``z_bf16 [R, F, 2]``,
    ``h_row_q [R, F]`` row-major + ``h_row_sf``, ``h_col_q [R, F]``
    column-major + ``h_col_sf`` (whole-matrix blocked for logical
    ``[F, R/32]``). Every destination byte is written, including the
    inactive-tail zeros.
    """
    if x_q.ndim != 2:
        raise ValueError(f"x_q must be 2D [R, D], got shape {tuple(x_q.shape)}")
    rows, model_dim = x_q.shape
    if w13_t_q.ndim != 3 or w13_t_q.shape[1] != model_dim or w13_t_q.shape[2] % 2:
        raise ValueError(
            f"w13_t_q must be [G, D, 2F] with D == {model_dim} and even 2F, "
            f"got shape {tuple(w13_t_q.shape)}"
        )
    groups, _, two_hidden = w13_t_q.shape
    hidden = two_hidden // 2
    device = x_q.device
    tensors = (
        x_q,
        x_sf,
        w13_t_q,
        w13_t_sf,
        offsets,
        z_bf16,
        h_row_q,
        h_row_sf,
        h_col_q,
        h_col_sf,
    )
    _common_launch_checks("launch_grouped_gemm_swiglu_fwd", device, tensors, groups)
    if rows == 0:
        raise ValueError(
            "R == 0 is handled by the op layer (empty destinations, no launch)"
        )

    validate_feature_dims(model_dim=model_dim, hidden_dim=hidden)
    validate_allocated_rows(rows)
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    validate_grouped_operand(
        x_q,
        name="x_q",
        shape=(rows, model_dim),
        stride=(model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        w13_t_q,
        name="w13_t_q",
        shape=(groups, model_dim, two_hidden),
        stride=(model_dim * two_hidden, 1, model_dim),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        x_sf,
        name="x_sf",
        logical_rows=rows,
        logical_cols=model_dim // _SF_VEC_SIZE,
        device=device,
    )
    validate_blocked_scales(
        w13_t_sf,
        name="w13_t_sf",
        logical_rows=two_hidden,
        logical_cols=model_dim // _SF_VEC_SIZE,
        device=device,
        groups=groups,
    )
    _check_sf_pointer_alignment("x_sf", x_sf)
    _check_sf_pointer_alignment("w13_t_sf", w13_t_sf)
    validate_destination(
        z_bf16,
        name="z_bf16",
        shape=(rows, hidden, 2),
        stride=(two_hidden, 2, 1),
        dtype=_BF16,
        device=device,
    )
    validate_destination(
        h_row_q,
        name="h_row_q",
        shape=(rows, hidden),
        stride=(hidden, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_destination(
        h_col_q,
        name="h_col_q",
        shape=(rows, hidden),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        h_row_sf,
        name="h_row_sf",
        logical_rows=rows,
        logical_cols=hidden // _SF_VEC_SIZE,
        device=device,
    )
    validate_blocked_scales(
        h_col_sf,
        name="h_col_sf",
        logical_rows=hidden,
        logical_cols=rows // _SF_VEC_SIZE,
        device=device,
    )
    _check_sf_pointer_alignment("h_row_sf", h_row_sf)
    _check_sf_pointer_alignment("h_col_sf", h_col_sf)
    # The epilogue computes flat element offsets in Int32.
    if rows * two_hidden >= 2**31:
        raise ValueError(
            f"R * 2F = {rows * two_hidden} does not fit the epilogue's int32 "
            "element indexing"
        )
    if two_hidden // _CTA_N > 65535:
        raise ValueError(f"2F = {two_hidden} exceeds the launch grid's Y limit")

    stream = _stream_for(device)
    args = (
        from_dlpack(activation_gemm_view(x_q), assumed_align=16),
        from_dlpack(weight_gemm_view(w13_t_q), assumed_align=16),
        from_dlpack(x_sf.view(torch.uint8).view(-1), assumed_align=16),
        from_dlpack(w13_t_sf.view(torch.uint8).view(-1), assumed_align=16),
        from_dlpack(offsets, assumed_align=4),
        (
            from_dlpack(z_bf16.view(rows, two_hidden).view(-1), assumed_align=16),
            from_dlpack(h_row_q.view(-1), assumed_align=16),
            from_dlpack(h_row_sf.view(torch.uint8).view(-1), assumed_align=16),
            from_dlpack(h_col_q.t().reshape(-1), assumed_align=16),
            from_dlpack(h_col_sf.view(torch.uint8).view(-1), assumed_align=16),
        ),
        stream,
    )
    key = _cache_key(
        "swiglu_fwd", (rows, model_dim, hidden, groups), tensors[:5], device
    )
    slot = _executor_slot(key)
    if not slot:
        slot.append(
            cute.compile(
                _launch_grouped_gemm, *args, _SWIGLU_FWD_CONFIG, _swiglu_fwd_epilogue
            )
        )
    slot[0](*args)


def launch_grouped_gemm_dswiglu_bwd(
    do_q,
    do_sf,
    w2_dgrad_q,
    w2_dgrad_sf,
    z_bf16,
    offsets,
    dz_row_q,
    dz_row_sf,
    dz_col_q,
    dz_col_sf,
):
    """FC2 dgrad grouped GEMM + dSwiGLU + dual MXFP8 quantization, one launch.

    ``do_q`` E4M3 ``[R, D]`` row-major + blocked ``do_sf``; ``w2_dgrad_q``
    E4M3 ``[G, D, F]`` stride ``(D*F, 1, D)`` + per-expert blocked
    ``w2_dgrad_sf``; ``z_bf16`` is the exact ``[R, F, 2]`` tensor the forward
    kernel wrote (tail rows are never read). Destinations: ``dz_row_q
    [R, 2F]`` row-major + ``dz_row_sf``, ``dz_col_q [R, 2F]`` column-major +
    ``dz_col_sf`` (whole-matrix blocked for logical ``[2F, R/32]``), gate/up
    gradients element-interleaved.
    """
    if do_q.ndim != 2:
        raise ValueError(f"do_q must be 2D [R, D], got shape {tuple(do_q.shape)}")
    rows, model_dim = do_q.shape
    if w2_dgrad_q.ndim != 3 or w2_dgrad_q.shape[1] != model_dim:
        raise ValueError(
            f"w2_dgrad_q must be [G, D, F] with D == {model_dim}, got shape "
            f"{tuple(w2_dgrad_q.shape)}"
        )
    groups, _, hidden = w2_dgrad_q.shape
    two_hidden = 2 * hidden
    device = do_q.device
    tensors = (
        do_q,
        do_sf,
        w2_dgrad_q,
        w2_dgrad_sf,
        z_bf16,
        offsets,
        dz_row_q,
        dz_row_sf,
        dz_col_q,
        dz_col_sf,
    )
    _common_launch_checks("launch_grouped_gemm_dswiglu_bwd", device, tensors, groups)
    if rows == 0:
        raise ValueError(
            "R == 0 is handled by the op layer (empty destinations, no launch)"
        )

    validate_feature_dims(model_dim=model_dim, hidden_dim=hidden)
    validate_allocated_rows(rows)
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    validate_grouped_operand(
        do_q,
        name="do_q",
        shape=(rows, model_dim),
        stride=(model_dim, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        w2_dgrad_q,
        name="w2_dgrad_q",
        shape=(groups, model_dim, hidden),
        stride=(model_dim * hidden, 1, model_dim),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        z_bf16,
        name="z_bf16",
        shape=(rows, hidden, 2),
        stride=(two_hidden, 2, 1),
        dtype=_BF16,
        device=device,
    )
    validate_blocked_scales(
        do_sf,
        name="do_sf",
        logical_rows=rows,
        logical_cols=model_dim // _SF_VEC_SIZE,
        device=device,
    )
    validate_blocked_scales(
        w2_dgrad_sf,
        name="w2_dgrad_sf",
        logical_rows=hidden,
        logical_cols=model_dim // _SF_VEC_SIZE,
        device=device,
        groups=groups,
    )
    _check_sf_pointer_alignment("do_sf", do_sf)
    _check_sf_pointer_alignment("w2_dgrad_sf", w2_dgrad_sf)
    validate_destination(
        dz_row_q,
        name="dz_row_q",
        shape=(rows, two_hidden),
        stride=(two_hidden, 1),
        dtype=_E4M3,
        device=device,
    )
    validate_destination(
        dz_col_q,
        name="dz_col_q",
        shape=(rows, two_hidden),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        dz_row_sf,
        name="dz_row_sf",
        logical_rows=rows,
        logical_cols=two_hidden // _SF_VEC_SIZE,
        device=device,
    )
    validate_blocked_scales(
        dz_col_sf,
        name="dz_col_sf",
        logical_rows=two_hidden,
        logical_cols=rows // _SF_VEC_SIZE,
        device=device,
    )
    _check_sf_pointer_alignment("dz_row_sf", dz_row_sf)
    _check_sf_pointer_alignment("dz_col_sf", dz_col_sf)
    if rows * two_hidden >= 2**31:
        raise ValueError(
            f"R * 2F = {rows * two_hidden} does not fit the epilogue's int32 "
            "element indexing"
        )
    if hidden // _CTA_N > 65535:
        raise ValueError(f"F = {hidden} exceeds the launch grid's Y limit")

    stream = _stream_for(device)
    args = (
        from_dlpack(activation_gemm_view(do_q), assumed_align=16),
        from_dlpack(weight_gemm_view(w2_dgrad_q), assumed_align=16),
        from_dlpack(do_sf.view(torch.uint8).view(-1), assumed_align=16),
        from_dlpack(w2_dgrad_sf.view(torch.uint8).view(-1), assumed_align=16),
        from_dlpack(offsets, assumed_align=4),
        (
            from_dlpack(z_bf16.view(rows, two_hidden).view(-1), assumed_align=16),
            from_dlpack(dz_row_q.view(-1), assumed_align=16),
            from_dlpack(dz_row_sf.view(torch.uint8).view(-1), assumed_align=16),
            from_dlpack(dz_col_q.t().reshape(-1), assumed_align=16),
            from_dlpack(dz_col_sf.view(torch.uint8).view(-1), assumed_align=16),
        ),
        stream,
    )
    key = _cache_key(
        "dswiglu_bwd", (rows, model_dim, hidden, groups), tensors[:6], device
    )
    slot = _executor_slot(key)
    if not slot:
        slot.append(
            cute.compile(
                _launch_grouped_gemm, *args, _DSWIGLU_BWD_CONFIG, _dswiglu_bwd_epilogue
            )
        )
    slot[0](*args)


def launch_grouped_gemm_wgrad(dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets, dw):
    """Grouped MXFP8 wgrad into a caller-allocated BF16 ``[G, N, K]``, one launch.

    Inputs are the columnwise-quantized outputs of the forward/backward
    kernels: ``dy_col_q`` E4M3 logical ``[R, N]`` stride ``(1, R)`` with
    ``dy_col_sf`` whole-matrix blocked for logical ``[N, R/32]``, and
    ``x_col_q`` / ``x_col_sf`` likewise for ``[R, K]``. Every element of
    ``dw`` is written, including the all-zero slice of a zero-token expert.
    """
    if dy_col_q.ndim != 2 or x_col_q.ndim != 2:
        raise ValueError(
            "dy_col_q and x_col_q must be 2D logical [R, N] and [R, K], got "
            f"{tuple(dy_col_q.shape)} and {tuple(x_col_q.shape)}"
        )
    rows, out_features = dy_col_q.shape
    x_rows, in_features = x_col_q.shape
    if x_rows != rows:
        raise ValueError(
            f"dy_col_q and x_col_q must share the row dim: {rows} vs {x_rows}"
        )
    groups = offsets.numel()
    device = dy_col_q.device
    tensors = (dy_col_q, dy_col_sf, x_col_q, x_col_sf, offsets, dw)
    _common_launch_checks("launch_grouped_gemm_wgrad", device, tensors, groups)

    validate_allocated_rows(rows)
    for name, value in (("dy_col_q's N", out_features), ("x_col_q's K", in_features)):
        if value <= 0 or value % 128 != 0:
            raise ValueError(f"{name} must be a positive multiple of 128, got {value}")
    validate_group_offsets(offsets, num_groups=groups, allocated_rows=rows)
    validate_grouped_operand(
        dy_col_q,
        name="dy_col_q",
        shape=(rows, out_features),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_grouped_operand(
        x_col_q,
        name="x_col_q",
        shape=(rows, in_features),
        stride=(1, rows),
        dtype=_E4M3,
        device=device,
    )
    validate_blocked_scales(
        dy_col_sf,
        name="dy_col_sf",
        logical_rows=out_features,
        logical_cols=rows // _SF_VEC_SIZE,
        device=device,
    )
    validate_blocked_scales(
        x_col_sf,
        name="x_col_sf",
        logical_rows=in_features,
        logical_cols=rows // _SF_VEC_SIZE,
        device=device,
    )
    _check_sf_pointer_alignment("dy_col_sf", dy_col_sf)
    _check_sf_pointer_alignment("x_col_sf", x_col_sf)
    validate_destination(
        dw,
        name="dw_bf16",
        shape=(groups, out_features, in_features),
        stride=(out_features * in_features, in_features, 1),
        dtype=_BF16,
        device=device,
    )
    if groups * out_features * in_features >= 2**31:
        raise ValueError(
            f"dw_bf16 has {groups * out_features * in_features} elements, which "
            "does not fit the epilogue's int32 element index"
        )
    if in_features // _CTA_N > 65535:
        raise ValueError(f"K = {in_features} exceeds the launch grid's Y limit")
    if groups > 65535:
        raise ValueError(f"G = {groups} exceeds the launch grid's Z limit")

    if rows == 0:
        # Every expert has zero rows: every slice is the zero matrix. The
        # destination is NOT empty here, and the contraction is.
        dw.zero_()
        return

    stream = _stream_for(device)
    args = (
        # The free transpose: logical [R, N] stride (1, R) IS a K-contiguous
        # [N, R], so the ragged axis becomes the contraction.
        from_dlpack(activation_gemm_view(dy_col_q.t()), assumed_align=16),
        from_dlpack(activation_gemm_view(x_col_q.t()), assumed_align=16),
        from_dlpack(dy_col_sf.view(torch.uint8).view(-1), assumed_align=16),
        from_dlpack(x_col_sf.view(torch.uint8).view(-1), assumed_align=16),
        from_dlpack(offsets, assumed_align=4),
        (from_dlpack(dw.permute(1, 2, 0), assumed_align=16),),
        stream,
    )
    key = _cache_key(
        "wgrad", (rows, out_features, in_features, groups), tensors[:5], device
    )
    slot = _executor_slot(key)
    if not slot:
        slot.append(
            cute.compile(_launch_grouped_gemm, *args, _WGRAD_CONFIG, _wgrad_epilogue)
        )
    slot[0](*args)
