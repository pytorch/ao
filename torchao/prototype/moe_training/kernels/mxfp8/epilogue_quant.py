# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Rowwise 1x32 and columnwise 32x1 quantizing epilogue for the MXFP8 grouped-MLP kernels.

Direction-agnostic and GEMM-agnostic: everything here consumes a run of packed
``bf16x2`` words held **one output row per thread**, which is exactly what the
tcgen05 TMEM->register copy delivers (``Ld32x32bOp`` has ``ThrID 32:1`` and a
destination TV layout ``(32, EPI_N_ACC):(EPI_N_ACC, 1)``, so thread ``t`` owns
row ``t`` of the epilogue subtile and all of its columns, contiguously). Two
consequences are load-bearing and are asserted by the geometry below:

* the rowwise 1x32 amax runs along N inside a single thread -- no cross-lane op;
* the columnwise 32x1 amax runs along M inside a single warp, and a 32-row MX
  block is never split across warps or CTAs, because the CTA tile is 128 rows,
  ``32 | 128``, and every expert boundary is 128-aligned.

WORD CONTRACT (every entry point below assumes it): word ``j`` of ``words``
holds output column ``col_base + 2*j`` in its **low** bf16 half and column
``col_base + 2*j + 1`` in its **high** half, and 16 words are exactly one
32-value scale block. Callers must have rounded to BF16 already -- both because
the kernel contract defines correctness at that boundary and because
:func:`float_to_e8m0` is exact only for a BF16-valued amax.

The columnwise destination has stride ``(1, R)``, i.e. it is physically a
row-major ``[cols, R]`` buffer, so the epilogue must transpose. It does that
through ``sPad``, a shared-memory staging tile of **BF16** values (not quantized
bytes): the reader's 16-byte load has to come out as four consecutive rows of
one column, which ``mul_cvt_2x`` + ``prmt_even``/``prmt_odd`` produce from two
words x two columns; staging bytes instead would force a stride-2 byte gather.

Both columnwise scale orientations use whole-matrix ``to_blocked`` coordinates
(feature index as the blocked row, row-block index as the blocked column). That
is this family's choice and it deliberately differs from torchao's per-group
``triton_mx_block_rearrange_2d_K_groups``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.utils import SmemAllocator

from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_epilogue import (
    SCALE_BLOCK,
    abs_max_nan_bf16x2,
    blocked_scale_idx,
    e8m0_reciprocal_bf16,
    float_to_e8m0,
    fold_amax,
    max_nan_bf16x2,
    mul_cvt_2x,
    prmt_even,
    prmt_odd,
)

__all__ = [
    "WORDS_PER_SCALE_BLOCK",
    "COLWISE_PAIRS",
    "COLWISE_ROWS_PER_WARP",
    "COLWISE_PAIR_PITCH",
    "COLWISE_WORDS_PER_WARP",
    "EPILOGUE_ROWS_PER_WARP",
    "spad_words",
    "spad_bytes",
    "alloc_spad",
    "rowwise_quant_block",
    "rowwise_quant_store",
    "rowwise_scale_flush",
    "colwise_quant_store",
]

# 16 bf16x2 words == 32 values == one E8M0 scale block.
WORDS_PER_SCALE_BLOCK = SCALE_BLOCK // 2
# One columnwise staging chunk is 32 output columns == 16 column pairs.
COLWISE_PAIRS = WORDS_PER_SCALE_BLOCK
# Rows one epilogue warp owns; equal to the 32x1 block height, which is what
# keeps the whole columnwise round trip inside a warp.
COLWISE_ROWS_PER_WARP = SCALE_BLOCK
EPILOGUE_ROWS_PER_WARP = COLWISE_ROWS_PER_WARP
# Word pitch between two column-pair slabs. The +4 pad is the entire swizzle:
# the reader's 128-bit shared loads are serviced in four phases of eight lanes,
# and phase 0 covers pairs 0..7 at word `36*pair + 4*chunk`, i.e. banks
# `4*pair + 4*chunk (mod 32)` -- eight distinct 4-word groups, all 32 banks, no
# conflict. At pitch 32 every one of those eight lanes lands on the same bank
# group and the load degenerates to an 8-way conflict.
COLWISE_PAIR_PITCH = COLWISE_ROWS_PER_WARP + 4
COLWISE_WORDS_PER_WARP = COLWISE_PAIRS * COLWISE_PAIR_PITCH
# Every sPad access is a 16-byte vector; 144 * pair + 64 * tq + 16 * chunk and
# the 2304-byte per-warp stride are all multiples of 16, so this holds for the
# allocation too.
_VEC_BYTES = 16
_VEC_WORDS = _VEC_BYTES // 4


def spad_words(num_epilogue_warps: int = 4) -> int:
    """Int32 word count of the columnwise staging tile.

    Sized by the warp count rather than by a config object so that this module
    stays importable, and testable, without the GEMM core. Callers holding a
    ``GroupedGemmConfig`` pass ``len(config.epilogue_warp_ids)``.
    """
    return num_epilogue_warps * COLWISE_WORDS_PER_WARP


def spad_bytes(num_epilogue_warps: int = 4) -> int:
    """Shared-memory bytes the columnwise transpose costs (9216 B at 4 warps).

    That is 0.27 of one 128x128x128 E4M3 AB pipeline stage, i.e. it does not
    change the achievable stage count. Feed it to
    ``config.smem_bytes(epilogue_smem_bytes=...)``.
    """
    return 4 * spad_words(num_epilogue_warps)  # 4 bytes per Int32 word


@cute.jit
def alloc_spad(
    allocator: SmemAllocator, NUM_EPILOGUE_WARPS: cutlass.Constexpr = 4
) -> cute.Tensor:
    """Allocate the columnwise staging tile as a flat Int32 shared tensor."""
    return allocator.allocate_tensor(
        Int32,
        cute.make_layout(spad_words(NUM_EPILOGUE_WARPS)),
        byte_alignment=_VEC_BYTES,
    )


@cute.jit
def _store_vec_i32(dst: cute.Tensor, word_offset: Int32, src: cute.Tensor):
    """Vectorized store of a register word run into a flat byte destination.

    `dst` is any 1-byte-element flat view; the run is reinterpreted as Int32, so
    `word_offset` counts 4-byte words from the start of the buffer.
    """
    cute.autovec_copy(
        src,
        cute.make_tensor(
            (cute.recast_ptr(dst.iterator, dtype=Int32) + word_offset).align(
                _VEC_BYTES
            ),
            cute.make_layout(cute.size(src)),
        ),
    )


@cute.jit
def _load_spad_quad(spad: cute.Tensor, word_offset: Int32) -> cute.Tensor:
    """One 16-byte shared load: four consecutive rows of one column pair."""
    quad = cute.make_rmem_tensor((_VEC_WORDS,), Int32)
    cute.autovec_copy(
        cute.make_tensor(
            (spad.iterator + word_offset).align(_VEC_BYTES),
            cute.make_layout(_VEC_WORDS),
        ),
        quad,
    )
    return quad


@cute.jit
def rowwise_quant_block(words: cute.Tensor, WORD_BASE: cutlass.Constexpr = 0):
    """Quantize one 1x32 rowwise block held entirely in one thread's registers.

    Returns ``(qwords, scale_byte)``: eight Int32 words holding the 32 E4M3
    bytes in column order, and the E8M0 exponent byte for the block.

    The amax reduction is intra-thread by construction (see the module
    docstring), so there is no shuffle and no partial amax carried between
    epilogue subtiles.
    """
    amax_packed = words[WORD_BASE]
    for w in cutlass.range_constexpr(1, WORDS_PER_SCALE_BLOCK):
        amax_packed = abs_max_nan_bf16x2(amax_packed, words[WORD_BASE + w])
    # fold_amax masks the junk sign bits xorsign leaves behind, then folds the
    # two bf16 lanes with the NaN-propagating max.
    scale_byte = float_to_e8m0(fold_amax(amax_packed) << Int32(16))
    inv = e8m0_reciprocal_bf16(scale_byte)
    inv_packed = inv | (inv << Int32(16))

    qwords = cute.make_rmem_tensor((WORDS_PER_SCALE_BLOCK // 2,), Int32)
    for q in cutlass.range_constexpr(WORDS_PER_SCALE_BLOCK // 2):
        # mul_cvt_2x emits bytes [w0.lo, w0.hi, w1.lo, w1.hi], i.e. four
        # consecutive columns in increasing address order.
        qwords[q] = mul_cvt_2x(
            words[WORD_BASE + 2 * q], words[WORD_BASE + 2 * q + 1], inv_packed
        )
    return qwords, scale_byte


@cute.jit
def rowwise_quant_store(
    words: cute.Tensor,
    qdata: cute.Tensor,
    row: Int32,
    col: Int32,
    row_stride: Int32,
    WORD_BASE: cutlass.Constexpr = 0,
) -> Int32:
    """Quantize one 1x32 block and store its 32 E4M3 bytes; return the scale byte.

    `qdata` is a flat 1-byte-element view of a row-major destination whose row
    pitch is `row_stride`. `row_stride` is a multiple of 128 and `col` a
    multiple of 32 under the kernel contract, so the destination byte offset is
    32-byte aligned and the 32 bytes leave as two STG.128.

    The scale byte is returned rather than stored: consecutive subtiles produce
    consecutive blocked-scale columns, so the caller buffers them and flushes
    once per CTA tile through :func:`rowwise_scale_flush`.
    """
    qwords, scale_byte = rowwise_quant_block(words, WORD_BASE=WORD_BASE)
    _store_vec_i32(qdata, (row * row_stride + col) >> Int32(2), qwords)
    return scale_byte


@cute.jit
def rowwise_scale_flush(
    scales: cute.Tensor,
    row: Int32,
    scale_col_base: Int32,
    scale_bytes: cute.Tensor,
    num_scale_col_blocks: Int32,
    NUM_BYTES: cutlass.Constexpr,
):
    """Write `NUM_BYTES` consecutive rowwise blocked-scale bytes for one row.

    Scale columns that share ``scale_col >> 2`` are contiguous bytes in the
    tcgen05 blocked layout (they differ only in the low two bits of the flat
    index), so a run of 4 starting at a 4-aligned `scale_col_base` is one 4-byte
    store. `scale_col_base` is `tile_n * NUM_BYTES` in both kernels, hence
    aligned to whichever width is selected here.

    `scales` must be a flat uint8 view of the blocked buffer.
    """
    if cutlass.const_expr(NUM_BYTES % 4 == 0):
        _flush_scale_run(
            scales, row, scale_col_base, scale_bytes, num_scale_col_blocks, 4, NUM_BYTES
        )
    elif cutlass.const_expr(NUM_BYTES % 2 == 0):
        _flush_scale_run(
            scales, row, scale_col_base, scale_bytes, num_scale_col_blocks, 2, NUM_BYTES
        )
    else:
        for i in cutlass.range_constexpr(NUM_BYTES):
            idx = blocked_scale_idx(row, scale_col_base + i, num_scale_col_blocks)
            scales[idx] = cutlass.Uint8(scale_bytes[i])


@cute.jit
def _flush_scale_run(
    scales: cute.Tensor,
    row: Int32,
    scale_col_base: Int32,
    scale_bytes: cute.Tensor,
    num_scale_col_blocks: Int32,
    WIDTH: cutlass.Constexpr,
    NUM_BYTES: cutlass.Constexpr,
):
    """Emit `NUM_BYTES` scale bytes as `NUM_BYTES // WIDTH` packed stores.

    A packed store addresses `scales` in units of WIDTH bytes, so the byte index
    it lands on is `(idx // WIDTH) * WIDTH`. That equals `idx` only when the
    index is WIDTH-aligned; otherwise the store both corrupts a neighbouring
    block's scale byte and leaves the intended one unwritten.

    In the blocked layout every term of `blocked_scale_idx` except `scale_col & 3`
    is a multiple of 4, so `idx % WIDTH == scale_col % WIDTH`. The requirement is
    therefore exactly `scale_col_base % WIDTH == 0`, a property of the caller's
    argument rather than of the data. Today's callers pass `2 * tile_n` (WIDTH 2)
    and `8 * tile_n` (WIDTH 4), whose multipliers are multiples of WIDTH, so
    alignment holds structurally for any tile index.

    The `assert_` below only fires in an assertions-enabled build
    (`CUTE_DSL_ENABLE_ASSERTIONS=1`); it is a debugging aid, not the guarantee.
    The guarantee is the caller contract above, which matters because the
    design's generalization ("buffer min(4, CTA_N/64) bytes") would put a
    computed expression here.
    """
    packed_ty = cutlass.Uint32 if cutlass.const_expr(WIDTH == 4) else cutlass.Uint16
    packed_ptr = cute.recast_ptr(scales.iterator, dtype=packed_ty)
    for run in cutlass.range_constexpr(NUM_BYTES // WIDTH):
        acc = Int32(0)
        for i in cutlass.range_constexpr(WIDTH):
            acc = acc | ((scale_bytes[run * WIDTH + i] & Int32(0xFF)) << Int32(8 * i))
        idx = blocked_scale_idx(row, scale_col_base + run * WIDTH, num_scale_col_blocks)
        cute.testing.assert_(
            idx % WIDTH == 0,
            "packed scale store is not WIDTH-aligned: scale_col_base must be a "
            "multiple of the store width",
        )
        dst = cute.make_tensor(packed_ptr + (idx // WIDTH), cute.make_layout(1))
        dst[0] = packed_ty(acc)


@cute.jit
def colwise_quant_store(
    words: cute.Tensor,
    spad: cute.Tensor,
    qdata: cute.Tensor,
    scales: cute.Tensor,
    tidx: Int32,
    row_base: Int32,
    col_base: Int32,
    num_rows: Int32,
    num_scale_col_blocks: Int32,
    WORD_BASE: cutlass.Constexpr = 0,
):
    """Transpose-quantize one 32-column chunk into the ``(1, R)`` destination.

    Every epilogue thread contributes its own row's 32 columns through `words`
    and then reads back a *different* slice -- 16 rows of one column pair --
    which is the transpose. Writer and reader sets are the same 32 threads, so
    both hazards are covered by ``sync_warp`` rather than a CTA barrier.

    `qdata` is a flat 1-byte-element view of the ``(1, num_rows)``-strided
    destination, i.e. physically row-major ``[cols, num_rows]``. `scales` is a
    flat uint8 view of the whole-matrix blocked columnwise scale buffer, indexed
    with transposed coordinates (feature, row-block).

    `tidx` is the EPILOGUE-LOCAL thread index in
    ``[0, 32 * num_epilogue_warps)``: a kernel whose epilogue runs on warps 4-7
    of 256 threads passes ``thread_idx - 128``. The epilogue's first thread must
    be 32-aligned so that epilogue warp `w` is one physical warp -- otherwise
    ``sync_warp`` and the butterfly shuffle no longer cover the writer/reader
    set and the transpose silently mixes rows from two warps.

    `row_base` must be a multiple of 128 and `col_base` a multiple of 32.
    """
    lane = tidx % Int32(COLWISE_ROWS_PER_WARP)
    warp = tidx // Int32(COLWISE_ROWS_PER_WARP)
    warp_base = warp * Int32(COLWISE_WORDS_PER_WARP)

    # Writer: for a fixed pair the 32 lanes hit 32 consecutive words, so every
    # store is conflict-free without a swizzle.
    for p in cutlass.range_constexpr(COLWISE_PAIRS):
        spad[warp_base + Int32(p * COLWISE_PAIR_PITCH) + lane] = words[WORD_BASE + p]
    cute.arch.sync_warp()

    # Reader: thread (pair, tq) owns column pair `pair` and the 16 rows
    # [16*tq, 16*tq+16) of this warp's 32-row slab. `half` numbers those 16-row
    # halves across the whole CTA tile, so `half` and `warp` are the same index
    # at two granularities and the row-block below needs no extra arithmetic.
    pair = tidx % Int32(COLWISE_PAIRS)
    half = tidx // Int32(COLWISE_PAIRS)
    tq = half % Int32(2)
    feature = col_base + Int32(2) * pair
    pair_base = (
        warp_base
        + pair * Int32(COLWISE_PAIR_PITCH)
        + tq * Int32(COLWISE_ROWS_PER_WARP // 2)
    )

    quads = []
    amax_packed = Int32(0)
    for c in cutlass.range_constexpr(COLWISE_ROWS_PER_WARP // 2 // _VEC_WORDS):
        quad = _load_spad_quad(spad, pair_base + Int32(c * _VEC_WORDS))
        quads.append(quad)
        for t in cutlass.range_constexpr(_VEC_WORDS):
            amax_packed = abs_max_nan_bf16x2(amax_packed, quad[t])
    cute.arch.sync_warp()

    # The two threads holding the halves of one 32-row block differ exactly in
    # bit 4 of tidx, which is a lane bit, so one butterfly at distance 16
    # completes the amax. Masking first drops the junk signs xorsign leaves.
    amax_packed = amax_packed & Int32(0x7FFF7FFF)
    amax_packed = max_nan_bf16x2(
        amax_packed, cute.arch.shuffle_sync_bfly(amax_packed, COLWISE_PAIRS)
    )

    # Low half is column `2*pair`'s amax, high half is `2*pair+1`'s: two
    # independent scales out of one reduction. Both are BF16-valued widened to
    # f32, which is what makes float_to_e8m0 exact.
    lo_byte = float_to_e8m0((amax_packed & Int32(0xFFFF)) << Int32(16))
    hi_byte = float_to_e8m0((amax_packed >> Int32(16)) << Int32(16))
    inv_packed = e8m0_reciprocal_bf16(lo_byte) | (
        e8m0_reciprocal_bf16(hi_byte) << Int32(16)
    )

    # One of the two threads per (pair, row-block) owns the scale bytes. They
    # are not adjacent in the blocked layout (they differ in the feature index,
    # hence by 16 bytes), so this stays two 1-byte stores.
    if tq == Int32(0):
        row_block = (row_base // Int32(SCALE_BLOCK)) + warp
        scales[blocked_scale_idx(feature, row_block, num_scale_col_blocks)] = (
            cutlass.Uint8(lo_byte)
        )
        scales[
            blocked_scale_idx(feature + Int32(1), row_block, num_scale_col_blocks)
        ] = cutlass.Uint8(hi_byte)

    # Quantize and transpose: mul_cvt_2x turns two rows x two columns into
    # bytes [r0c0, r0c1, r1c0, r1c1], and the byte permutes split that into one
    # word per column holding four consecutive rows. This is where prmt_even /
    # prmt_odd belong -- not in any gate/up de-interleave, where the pair is
    # already in two separate FP32 registers.
    num_quads = cutlass.const_expr(COLWISE_ROWS_PER_WARP // 2 // _VEC_WORDS)
    col_lo = cute.make_rmem_tensor((num_quads,), Int32)
    col_hi = cute.make_rmem_tensor((num_quads,), Int32)
    for c in cutlass.range_constexpr(num_quads):
        quad = quads[c]
        pack01 = mul_cvt_2x(quad[0], quad[1], inv_packed)
        pack23 = mul_cvt_2x(quad[2], quad[3], inv_packed)
        col_lo[c] = prmt_even(pack01, pack23)
        col_hi[c] = prmt_odd(pack01, pack23)

    # col_lo/col_hi are 16 consecutive rows of one column: 16 contiguous bytes
    # of the [cols, num_rows] buffer. num_rows and row_base are multiples of 128
    # and the row offset is a multiple of 16, so both stores are STG.128.
    row_offset = row_base + half * Int32(COLWISE_ROWS_PER_WARP // 2)
    _store_vec_i32(qdata, (feature * num_rows + row_offset) >> Int32(2), col_lo)
    _store_vec_i32(
        qdata, ((feature + Int32(1)) * num_rows + row_offset) >> Int32(2), col_hi
    )
