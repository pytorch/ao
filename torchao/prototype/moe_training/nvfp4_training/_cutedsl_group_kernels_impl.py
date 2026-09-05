# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped RHT + NVFP4 kernels for SM100, ported from TransformerEngine.

Structural port of ``nvte_group_hadamard_transform_cast_fusion_graph_safe``
(TE ``graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu``): CLC
dynamic persistent scheduler, 16-warp specialization, and a 128-token epilogue
tile that is group-aligned by construction (group offsets are 128-aligned, so a
tile never straddles two experts). Numeric primitives are reused from
``_cutedsl_kernels_impl`` rather than ported from TE, so the outputs match
torchao's existing NVFP4 oracle, not TE bit-for-bit.

Axis naming follows torch, not TE: ``A`` is ``(tokens, hidden)`` row-major, so
``A.t()`` is TE's column-major ``(M=hidden, N=packed_tokens)`` with no copy. The
UMMA contracts 16 consecutive *tokens* against the 16x16 Hadamard.

Two kernels share that mainloop and differ only in their epilogues:
  - ``_Tcgen05GroupRowColFused``: quantizes col=RHT(A.t()) and row=A to NVFP4.
  - ``_Tcgen05GroupRhtAmax``: reduces col=max|RHT(A.t())|, row=max|A|, per group.

Columnwise scale factors are per-group swizzled ``[hidden, tokens_g]`` blocks
concatenated in one flat allocation. The group-local 64-token tile axis must
therefore restart at every group boundary; treating the allocation as one
globally swizzled ``[hidden, packed_tokens]`` buffer gives later groups the
wrong layout.
"""

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import blackwell_helpers as sm100_utils
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ._cutedsl_kernels_impl import (
    DEFAULT_SIGN_VECTOR,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    FP32_MAX,
    HADAMARD_DIM,
    TILE_BLOCKS,
    _abs_f32,
    _atom_max_f32_nonneg,
    _bf16hi_to_f32,
    _bf16lo_to_f32,
    _div_rn_f32,
    _get_rht_buffer,
    _get_sr_rng_buffer,
    _max_f32,
    _min_f32,
    _quant16,
    _round_rht_amax,
    philox4_all,
    philox_prep,
)

# --- tile shapes (TE :262-271). M = hidden, N = tokens, K = 16 (the RHT block) ---
M_TILE = 128  # hidden rows per tile
N_TILE = 16  # UMMA N = one RHT block
K = HADAMARD_DIM  # UMMA K = 16 contracted tokens
EPI_UNROLL = 8  # 128 tokens / 16 -> UMMAs per accumulator stage
TOKEN_TILE = N_TILE * EPI_UNROLL  # 128 tokens per mainloop/epilogue tile
MMA_TILER = (M_TILE, N_TILE, K)
K_TILE_MAX = 8  # token tiles per scheduler work item (TE :271)

ACC_STAGES = 4  # 512 TMEM columns / (8 * 16)
CLC_STAGES = 1
CLC_RESPONSE_I32 = 4  # 16-byte cluster-launch-control response

# Mainloop stage count from the SM100 shared-memory budget (TE :1257-1264). The
# A tile dominates; the reserve covers sB, every pipeline's mbarriers, the CLC
# response, and the TMEM holding buffer. The epilogues reduce in registers and
# atomic straight to global, so there is no per-group amax staging in SMEM.
_SMEM_CAPACITY = 232448
_A_TILE_BYTES = M_TILE * TOKEN_TILE * 2
_SMEM_RESERVE = 2048
MAINLOOP_STAGES = (_SMEM_CAPACITY - _SMEM_RESERVE) // _A_TILE_BYTES

# --- warp specialization (TE :395-416). 16 warps / 512 threads ---
MMA_WARP = 0
TMA_WARP = 1
SCHED_WARP = 2
IDLE_WARP = 3
COL_WARP_BEGIN = 4
COL_WARP_END = 8
ROW_WARP_BEGIN = 8
ROW_WARP_END = 16
N_WARPS = 16
TPB = 32 * N_WARPS
COL_THREADS = 32 * (COL_WARP_END - COL_WARP_BEGIN)  # 128
ROW_THREADS = 32 * (ROW_WARP_END - ROW_WARP_BEGIN)  # 256

# warpgroup_reg_alloc is warpgroup-granular, so all four warps of WG0 (incl. the
# idle warp) must agree on the dealloc. 128*32 + 128*192 + 256*136 = 63488 <= 65536.
REG_DEALLOC = 32
REG_COL = 192
REG_ROW = 136

TMEM_ALLOC_BAR = 1
TMEM_DEALLOC_BAR = 2

# Row epilogue thread map: 256 threads cover a (128 hidden, 128 token) tile as
# 16 hidden x 1 token per thread per pass, 4 passes.
ROW_HB = M_TILE // 16  # 8 hidden blocks of 16
ROW_TOK_PER_PASS = ROW_THREADS // ROW_HB  # 32 tokens per pass
ROW_PASSES = TOKEN_TILE // ROW_TOK_PER_PASS  # 4


# Group lookup is a binary search unrolled to a constexpr depth -- chosen to keep the
# epilogues branch-free -- so it resolves exactly 2**GROUP_SEARCH_STEPS groups and the
# group count is capped there. Raising the cap means raising the depth with it: at
# E > 2**GROUP_SEARCH_STEPS the search exits with hi - lo > 1 and returns a group index
# off by one, which is a silently wrong amax rather than a failure. Keep the two in
# sync; ``test_cutedsl_group_rht_amax_rejects_too_many_groups`` pins the boundary.
#
# The 64 is ours, not inherited: TE's grouped NVFP4 kernels also cap at 64, but for
# unrelated reasons -- the pointer-list kernels (kMaxTensorsPerKernel) are bounded by
# the 4 KB kernel-argument limit, and the graph-safe ones by a shared-memory scratch
# array. TE's graph-safe kernels, which are the ones this design mirrors (packed input
# plus device-side offsets, no pointer arrays), use an unbounded `while` search and have
# no depth limit at all. Nothing here forces 64; it is comfortably above the local
# expert counts these models train at (671B at EP=64 gives 4).
MAX_GROUPS = 64
GROUP_SEARCH_STEPS = 6
assert MAX_GROUPS <= 2**GROUP_SEARCH_STEPS, (
    f"MAX_GROUPS={MAX_GROUPS} exceeds what a depth-{GROUP_SEARCH_STEPS} search resolves "
    f"({2**GROUP_SEARCH_STEPS})"
)


def _group_idx(token, offsets_t, num_groups):
    """Group containing ``token``, from cumulative row-end offsets.

    Branch-free port of triton ``_get_group_idx_binary``: the halvings are
    unrolled to a constexpr depth, so there is no dynamic control flow inside
    the epilogues. Offsets alone determine membership, which is correct for
    both SAME_BOTH_DIMS and VARYING_FIRST_DIM. The result is CTA-uniform
    because a 128-token tile never straddles a 128-aligned group boundary.
    """
    lo = cutlass.Int32(0)
    hi = num_groups
    for _ in range(GROUP_SEARCH_STEPS):
        mid = lo + (hi - lo) // cutlass.Int32(2)
        probe = cutlass.select_(mid > cutlass.Int32(0), mid - cutlass.Int32(1), 0)
        ge = token >= offsets_t[probe]
        active = (hi - lo) > cutlass.Int32(1)
        lo = cutlass.select_(active, cutlass.select_(ge, mid, lo), lo)
        hi = cutlass.select_(active, cutlass.select_(ge, hi, mid), hi)
    return lo


@cute.jit
def _flush_group_max(run_max, amax_t, g, lane):
    """Reduce a warp's running max and commit it to ``amax_t[g]``.

    The epilogues carry ``run_max`` across every tile that shares a group, so
    this runs once per group per work item rather than once per tile. A zero
    flush -- an empty work item, or one whose first tile already crossed -- is a
    no-op against the pre-zeroed buffer.

    Needs ``@cute.jit``: the lane predicate is a dynamic branch, which only
    lowers inside an AST-preprocessed function.
    """
    for offset in range(5):
        run_max = _max_f32(
            run_max, cute.arch.shuffle_sync_bfly(run_max, 1 << (4 - offset))
        )
    if lane == cutlass.Int32(0):
        _atom_max_f32_nonneg(amax_t.iterator + g, run_max)


def _group_at_work_item(tile_n_base, offsets_t, num_groups):
    """Group of a work item's first tile, with that group's end offset.

    A work item is K_TILE_MAX consecutive token tiles, so one search covers all
    of them unless a tile crosses out of the returned group; the epilogues
    re-search on that crossing rather than stepping ``g``, which keeps the
    result identical to a per-tile lookup even when a group is empty. Both
    searches go through ``_group_idx``, so both inherit its ``MAX_GROUPS`` /
    ``GROUP_SEARCH_STEPS`` depth cap -- raising one without the other returns a
    silently wrong group index rather than failing.
    """
    g = _group_idx(tile_n_base * cutlass.Int32(TOKEN_TILE), offsets_t, num_groups)
    return g, offsets_t[g]


def _global_scale(amax):
    """NVFP4 two-level scale scalars from a global amax (TE :779-785).

    Returns ``(encode, decode, encode / fp4_max)``; a zero amax yields identity
    scales so the block scales stay finite.
    """
    is_zero = amax == cutlass.Float32(0.0)
    safe = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), amax))
    c = _min_f32(
        _div_rn_f32(cutlass.Float32(FP8_E4M3_MAX * FP4_E2M1_MAX), safe),
        cutlass.Float32(FP32_MAX),
    )
    c = cutlass.Float32(
        cutlass.select_(c == cutlass.Float32(0.0), cutlass.Float32(1.0), c)
    )
    enc = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), c))
    dec = _div_rn_f32(cutlass.Float32(1.0), enc)
    return enc, dec, enc * cutlass.Float32(1.0 / FP4_E2M1_MAX)


class _GroupRhtMainloop:
    """Shared mainloop plumbing for the two grouped RHT kernels.

    Both kernels stream the same A tiles through the same UMMA and scheduler and
    differ only in their epilogues, so the MMA/TMA/scheduler setup is built once
    here and the kernel bodies stay independent.
    """

    def _setup(self, mA: cute.Tensor, mB: cute.Tensor, hidden, tokens):
        mma_op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            MMA_TILER,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,  # A: contiguous along hidden
            OperandMajorMode.K,  # B: H^T stored (N=16, K=16) row-major
        )
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))

        # A tile: 128 hidden x 128 tokens -> 8 k-blocks of 16 tokens.
        a_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        a_shape = tiled_mma.partition_shape_A(
            cute.dice((M_TILE, N_TILE, TOKEN_TILE), (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            a_atom, cute.append(a_shape, MAINLOOP_STAGES), order=(1, 2, 3)
        )
        # Same bytes, plain (hidden, token, stage) grouping for the row warps.
        # This is the DSL equivalent of TE's as_position_independent_swizzle_tensor.
        a_clean_layout = cute.tile_to_shape(
            a_atom, (M_TILE, TOKEN_TILE, MAINLOOP_STAGES), order=(0, 1, 2)
        )

        b_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.BFloat16
        )
        b_shape = tiled_mma.partition_shape_B(cute.dice(MMA_TILER, (None, 1, 1)))
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            b_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            (M_TILE, N_TILE, TOKEN_TILE),
            tiled_mma,
            (1, 1, 1, 1),
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
            MMA_TILER,
            tiled_mma,
            (1, 1, 1, 1),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        # TMEM accumulator: 4 stages x 8 sub-tiles x 16 columns = 512 columns.
        acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(cute.append(acc_shape, EPI_UNROLL), ACC_STAGES)
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        # The Hadamard rides its own one-shot barrier (TE :546-553), so the
        # per-tile mainloop transaction covers the A tile only.
        num_tma_load_bytes = M_TILE * TOKEN_TILE * 2
        num_b_load_bytes = N_TILE * K * 2

        tiles_in_m = hidden // cutlass.Int32(M_TILE)
        tiles_in_n = tokens // cutlass.Int32(TOKEN_TILE)
        # One work item = up to K_TILE_MAX consecutive token tiles at a fixed
        # hidden slab, reproducing TE's tile_n_base = q * K_TILE_MAX (TE :301-306).
        tiles_in_n_outer = (
            tiles_in_n + cutlass.Int32(K_TILE_MAX - 1)
        ) // cutlass.Int32(K_TILE_MAX)
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (tiles_in_m, tiles_in_n_outer, cutlass.Int32(1)),
            (1, 1, 1),
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        return (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            tile_sched_params,
            grid,
        )


class _Tcgen05GroupRowColFused(_GroupRhtMainloop):
    """Fused grouped RHT columnwise + raw rowwise NVFP4 quantization.

    One TMA-loaded A tile feeds two consumers: the UMMA (which applies the 16x16
    RHT along the token axis, accumulating in TMEM for the columnwise path) and
    the row warp group (which reads the same SMEM bytes for the rowwise path).
    """

    def __init__(self, sr: bool = False, fast_math: bool = False):
        self.sr = sr
        self.fast_math = fast_math

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (hidden, tokens, 1) bf16, hidden contiguous
        mB: cute.Tensor,  # (16, 16, 1) bf16 = H^T
        mColFP4: cute.Tensor,  # (hidden, tokens//8) u32
        mColSF: cute.Tensor,  # flat u32 concatenation of per-group swizzled scales
        mRowFP4: cute.Tensor,  # (tokens, hidden//16) u64: the row code pair per store
        mRowSF: cute.Tensor,  # (tokens//128, hidden//64, 32, 16) e4m3
        row_amax_t: cute.Tensor,  # (num_tensors,) f32
        col_amax_t: cute.Tensor,  # (num_tensors,) f32
        sr_rng_t: cute.Tensor,  # (8,) i32 Philox state
        offsets_t: cute.Tensor,  # (num_tensors,) i32 cumulative row-end offsets
        logical_len_t: cute.Tensor,  # (1,) i32 valid padded token count
        hidden: cutlass.Int32,
        tokens: cutlass.Int32,
        num_tensors: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.c_layout = utils.LayoutEnum.from_tensor(mColFP4)
        (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            tile_sched_params,
            grid,
        ) = self._setup(mA, mB, hidden, tokens)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mColFP4,
            mColSF,
            mRowFP4,
            mRowSF,
            row_amax_t,
            col_amax_t,
            sr_rng_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            hidden,
            num_tensors,
            tile_sched_params,
        ).launch(grid=grid, block=(TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mColFP4: cute.Tensor,
        mColSF: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        col_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_b_load_bytes: cutlass.Constexpr,
        tiles_in_n: cutlass.Int32,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, MAINLOOP_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
            clc_mbar: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, CLC_STAGES * 4]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mainloop: one TMA producer, two heterogeneous consumer groups on the
        # same stage -- the UMMA (released via tcgen05.commit) and the 256 row
        # threads (released via a plain mbarrier arrive). TE hand-rolled this as
        # CustomizedPipelineTmaUmmaAsync; the DSL exposes it directly.
        ab_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, ROW_THREADS
            ),
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # One arrival per col warp: consumer_release runs under elect_one.
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, COL_WARP_END - COL_WARP_BEGIN
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Every active role consumes exactly one CLC stage per work item. The
        # idle warp must stay out of this count or the pipeline deadlocks.
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar.data_ptr(),
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                TPB - 32,  # 480
            ),
            tx_count=16,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 + COL_THREADS,  # mma + col
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=COL_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        if warp_idx == SCHED_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # A lives in SMEM once; the MMA sees the swizzled UMMA grouping and the
        # row warps see a plain (hidden, token, stage) view of the same bytes.
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
            cute.slice_((M_TILE, N_TILE, TOKEN_TILE), (None, 0, None)),
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

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        work_tile = tile_sched.initial_work_tile_info()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, CLC_STAGES
        )

        # Graph-safe work bound: the scheduler is sized from host-known capacity,
        # but how much of that capacity holds real tokens is only known on device
        # (TE derives the same bound from offsets[num_tensors] at :224). Tiles at
        # or past this point are never loaded or stored.
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)
        # tile_n * tiles_in_h + tile_m gives each tile a stable identity from its
        # coordinates alone. This kernel's CLC scheduler is persistent, so which tile a
        # CTA visits next is not fixed; deriving the SR Philox counter from coordinates
        # rather than from a running per-thread counter is what keeps the stream a pure
        # function of position and makes the same rng_state reproduce the same codes.
        tiles_in_h = hidden // cutlass.Int32(M_TILE)

        # ==================== TMA warp (mainloop producer) ====================
        if warp_idx == TMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)

            # One-shot Hadamard load: 16x16 bf16, never re-armed (TE :546-553).
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, num_b_load_bytes
                )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0)],
                tBsB[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, tile_m, tile_n_base + k_tile, 0)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    ab_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==================== Scheduler warp ====================
        if warp_idx == SCHED_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, CLC_STAGES
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                tile_sched.advance_to_next_work(
                    clc_pipeline.producer_get_barrier(clc_producer_state)
                )
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # ==================== Idle warp ====================
        # Must dealloc with the rest of warpgroup 0 and must NOT touch the CLC
        # pipeline (it is excluded from the 480 consumer arrivals).
        if warp_idx == IDLE_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            tmem.wait_for_alloc()
            tCtAcc_base = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    for i in cutlass.range_constexpr(EPI_UNROLL):
                        # ScaleOut.Zero throughout: every UMMA fully overwrites
                        # its 128x16 accumulator over the K=16 Hadamard block.
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        acc = tCtAcc_base[
                            (None, None, None, i, acc_producer_state.index)
                        ]
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrA[(None, None, i, ab_consumer_state.index)],
                            tCrB[(None, None, 0, 0)],
                            acc,
                        )
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                    ab_pipeline.consumer_release(
                        ab_consumer_state, pipeline.PipelineOp.TCGen05Mma
                    )
                    ab_consumer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== Columnwise epilogue (TMEM -> NVFP4) ====================
        if warp_idx >= COL_WARP_BEGIN and warp_idx < COL_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_COL)
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

            # TE loads this thread's full 8x16 accumulator fragment before
            # releasing the producer. Preserve the MMA's physical TMEM strides
            # while presenting the eight N subtiles as one 128-column tile.
            bulk_tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout(
                    (M_TILE, (N_TILE, EPI_UNROLL), ACC_STAGES),
                    stride=(65536, (1, N_TILE), M_TILE),
                ),
            )
            bulk_copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition.x64),
                cutlass.Float32,
            )
            bulk_tiled_copy_t2r = tcgen05.make_tmem_copy(
                bulk_copy_atom_t2r, bulk_tCtAcc[(None, None, 0)]
            )
            bulk_thr_copy_t2r = bulk_tiled_copy_t2r.get_slice(tidx)
            bulk_tTR_tAcc = bulk_thr_copy_t2r.partition_S(bulk_tCtAcc)
            bulk_tTR_rAcc = cute.make_rmem_tensor(((64, 1), 1, 2), cutlass.Float32)

            # This thread owns one hidden row and, per sub-tile, the 16 tokens
            # of one NVFP4 block -> 8 blocks (128 tokens) per epilogue tile.
            h_local = tidx - COL_WARP_BEGIN * cutlass.Int32(32)
            rCol = cute.make_rmem_tensor((2 * EPI_UNROLL,), cutlass.Uint32)
            rColSF = cute.make_rmem_tensor((EPI_UNROLL,), cutlass.Float8E4M3FN)

            col_state = None
            if cutlass.const_expr(self.sr):
                col_state = philox_prep(
                    cutlass.Uint32(sr_rng_t[0]),
                    cutlass.Uint32(sr_rng_t[1]),
                    cutlass.Uint32(sr_rng_t[2]),
                )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    _, dec, enc_over_fp4max = _global_scale(col_amax_t[g])
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        tile_n = tile_n_base + k_tile
                        token = tile_n * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                            _, dec, enc_over_fp4max = _global_scale(col_amax_t[g])
                        h_global = tile_m * cutlass.Int32(M_TILE) + h_local
                        tile_id = tile_n * tiles_in_h + tile_m

                        acc_pipeline.consumer_wait(acc_consumer_state)
                        cute.copy(
                            bulk_tiled_copy_t2r,
                            bulk_tTR_tAcc[(None, None, None, acc_consumer_state.index)],
                            bulk_tTR_rAcc,
                        )
                        bulk_vals = bulk_tTR_rAcc.load().reshape((16, 8))
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        for u in cutlass.range_constexpr(EPI_UNROLL):
                            vals = bulk_vals[(None, u)]
                            col_rb = None
                            if cutlass.const_expr(self.sr):
                                # One draw per 16-element block, indexed by its position in
                                # the columnwise (hidden, tokens) tile.
                                col_rb = philox4_all(
                                    col_state,
                                    tile_id * cutlass.Int32(TILE_BLOCKS)
                                    + h_local * cutlass.Int32(TOKEN_TILE // 16)
                                    + cutlass.Int32(u),
                                )
                            w0, w1, sf = _quant16(
                                vals,
                                enc_over_fp4max,
                                dec,
                                self.sr,
                                col_rb,
                                rht_acc=True,
                                fast_math=self.fast_math,
                            )
                            rCol[u * 2] = w0
                            rCol[u * 2 + 1] = w1
                            rColSF[u] = sf

                        gCol = cute.local_tile(
                            mColFP4, (M_TILE, TOKEN_TILE // 8), (tile_m, tile_n)
                        )
                        cute.autovec_copy(rCol, gCol[(h_local, None)])
                        _store_grouped_col_sf_u32(
                            mColSF,
                            rColSF,
                            h_global,
                            tile_n * cutlass.Int32(TOKEN_TILE // 16),
                            g,
                            offsets_t,
                            hidden,
                        )

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== Rowwise epilogue (SMEM -> NVFP4) ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < ROW_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_ROW)
            r_local = tidx - ROW_WARP_BEGIN * cutlass.Int32(32)
            hb = r_local % cutlass.Int32(ROW_HB)  # 16-hidden block
            t0 = r_local // cutlass.Int32(ROW_HB)  # token within a pass

            blk = cute.make_rmem_tensor((16,), cutlass.Float32)
            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            rPair = cute.make_rmem_tensor((2,), cutlass.Uint32)
            row_state = None
            if cutlass.const_expr(self.sr):
                row_state = philox_prep(
                    cutlass.Uint32(sr_rng_t[4]),
                    cutlass.Uint32(sr_rng_t[5]),
                    cutlass.Uint32(sr_rng_t[6]),
                )
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[g])
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        token = (tile_n_base + k_tile) * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                            _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[g])

                        ab_pipeline.consumer_wait(ab_consumer_state)
                        stage = ab_consumer_state.index
                        tile_n = tile_n_base + k_tile
                        gRow = cute.local_tile(
                            mRowFP4, (TOKEN_TILE, M_TILE // 16), (tile_n, tile_m)
                        )
                        if cutlass.const_expr(self.sr):
                            tile_id = tile_n * tiles_in_h + tile_m
                        for p in cutlass.range_constexpr(ROW_PASSES):
                            tok = p * cutlass.Int32(ROW_TOK_PER_PASS) + t0
                            cute.autovec_copy(
                                cute.local_tile(
                                    sA_clean[(None, tok, stage)], (16,), (hb,)
                                ),
                                rBlk,
                            )
                            rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                            for j in cutlass.range_constexpr(8):
                                blk[2 * j] = _bf16lo_to_f32(rWords[j])
                                blk[2 * j + 1] = _bf16hi_to_f32(rWords[j])
                            row_rb = None
                            if cutlass.const_expr(self.sr):
                                # One draw per 16-element block, indexed by its position in
                                # the rowwise (tokens, hidden) tile.
                                row_rb = philox4_all(
                                    row_state,
                                    tile_id * cutlass.Int32(TILE_BLOCKS)
                                    + tok * cutlass.Int32(M_TILE // 16)
                                    + hb,
                                )
                            w0, w1, sf = _quant16(
                                blk,
                                r_enc_over_fp4max,
                                r_dec,
                                self.sr,
                                row_rb,
                                fast_math=self.fast_math,
                            )
                            # One 64-bit store, not two 32-bit ones. A warp covers 4
                            # tokens x ROW_HB hidden blocks, so consecutive lanes differ in
                            # hb: as two u32 stores each wrote 4B at an 8B lane stride,
                            # spanning 64B to fill 32B, and every sector came back half
                            # wasted (8.00 sectors/instruction against an ideal of 4.00).
                            # Storing the pair makes the lanes contiguous and lands at the
                            # ideal, which is what TE's STG.E.64 already does here.
                            rPair[0] = w0
                            rPair[1] = w1
                            pair64 = cute.recast_tensor(rPair, cutlass.Uint64)
                            gRow[(tok, hb)] = pair64[0]
                            _store_sf_byte(
                                mRowSF,
                                sf,
                                tile_n * cutlass.Int32(TOKEN_TILE) + tok,
                                tile_m * cutlass.Int32(ROW_HB) + hb,
                            )
                        ab_pipeline.consumer_release(
                            ab_consumer_state, pipeline.PipelineOp.AsyncThread
                        )
                        ab_consumer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()


# maxsize=None and no defaults, for the reasons spelled out over ``_compile_fused_kernel``:
# an entry is a compiled kernel a CUDA-graph capture may depend on, so the cache must never
# evict, and the key is the literal (args, kwargs) shape, so every caller passes all three
# positionally or the pre-capture warm-up warms keys nothing will look up.
@functools.lru_cache(maxsize=None)
def _compile_group_fused_kernel(device_idx: int, sr: bool, fast_math: bool):
    """Compile the grouped fused kernel with symbolic shapes (cached per device+flags).

    ``sym_int`` divisibilities let one compiled kernel serve any
    ``hidden % 128``, ``tokens % 128``. Exact and fast arithmetic are separate
    cache entries; sign vector, amaxes, offsets, and RNG remain runtime buffers.
    """
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)

    fake_a = make_fake_tensor(
        cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)
    )
    fake_b = make_fake_tensor(
        cutlass.BFloat16, (HADAMARD_DIM, HADAMARD_DIM, 1), stride=(HADAMARD_DIM, 1, 1)
    )
    # The columnwise epilogue stores a thread's 16 contiguous u32 with one
    # autovec_copy, which widens only as far as it can prove alignment: the u32
    # default is 4B, so it lowers to sixteen scalar STG. The allocation is
    # torch.empty (256B) and tokens % TOKEN_TILE == 0 makes the row stride a
    # multiple of TOKEN_TILE // 8 u32 = 64B, so every row start is 16B aligned.
    fake_col_fp4 = make_fake_tensor(
        cutlass.Uint32,
        (h_sym, cute.sym_int(divisibility=TOKEN_TILE // 8)),
        stride=(cute.sym_int(divisibility=TOKEN_TILE // 8), 1),
        assumed_align=16,
    )
    fake_col_sf = make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,))
    # u64: the row epilogue stores the two code words of a 16-hidden block together.
    # The allocation is torch.empty (256B) and hidden % 128 == 0 makes the row stride
    # hidden/2 bytes, a multiple of 64B, so every row start is 16B aligned.
    fake_row_fp4 = make_fake_tensor(
        cutlass.Uint64,
        (t_sym, cute.sym_int(divisibility=M_TILE // 16)),
        stride=(free(), 1),
        assumed_align=16,
    )
    fake_row_sf = make_fake_tensor(
        cutlass.Float8E4M3FN, (free(), free(), 32, 16), stride=(free(), 512, 16, 1)
    )
    fake_amax = make_fake_tensor(cutlass.Float32, (free(),), stride=(1,))
    fake_i32_1 = make_fake_tensor(cutlass.Int32, (1,), stride=(1,))
    # (8,) Philox state: [col_seed_lo/hi, col_off_lo/hi, row_seed_lo/hi, row_off_lo/hi].
    fake_sr_rng = make_fake_tensor(cutlass.Int32, (8,), stride=(1,))
    fake_offsets = make_fake_tensor(cutlass.Int32, (free(),), stride=(1,))

    return cute.compile(
        _Tcgen05GroupRowColFused(sr=sr, fast_math=fast_math),
        fake_a,
        fake_b,
        fake_col_fp4,
        fake_col_sf,
        fake_row_fp4,
        fake_row_sf,
        fake_amax,
        fake_amax,
        fake_sr_rng,
        fake_offsets,
        fake_i32_1,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_rht_quantize_row_col_impl(
    A: torch.Tensor,
    offsets: torch.Tensor,
    row_global_amax: torch.Tensor,
    col_global_amax: torch.Tensor,
    num_tensors: int,
    sign_vector=DEFAULT_SIGN_VECTOR,
    logical_packed_length: Optional[torch.Tensor] = None,
    stochastic_rounding: bool = False,
    sr_rng: Optional[torch.Tensor] = None,
    use_fast_math: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped fused RHT columnwise + raw rowwise NVFP4 quantization.

    ``A`` is ``(tokens, hidden)`` bfloat16 row-major with ``tokens % 128 == 0``
    and ``hidden % 128 == 0``. Returns ``(col_fp4, col_sf, row_fp4, row_sf)``
    with swizzled scale factors; the wrapper returns views matching torchao's
    ``(qa, sfa, qd, sfd)`` contract. Columnwise scale storage is a flat
    concatenation of independently swizzled group buffers.
    """
    tokens, hidden = A.shape
    dev = A.device
    A = A.detach()

    col_fp4 = torch.empty((hidden, tokens // 8), dtype=torch.uint32, device=dev)
    row_fp4 = torch.empty((tokens, hidden // 8), dtype=torch.uint32, device=dev)
    col_sf = torch.empty(
        (hidden // 128, tokens // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )
    row_sf = torch.empty(
        (tokens // 128, hidden // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )

    rht_nk = _get_rht_buffer(tuple(sign_vector), dev.index)
    sr_rng_t = _get_sr_rng_buffer(dev.index)
    if stochastic_rounding:
        # [col_seed, col_offset, row_seed, row_offset] int64 -> the eight little-endian
        # 32-bit halves Philox keys and counters are built from. One 32-byte D2D copy,
        # so it stays graph-capturable and does no host RNG.
        sr_rng_t.copy_(sr_rng[:4].view(torch.int32))
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # The CuteDSL entry point requires byte_offset==0, and offsets[-1:] is a
    # nonzero-offset view for every multi-group launch. Cloning is device-side
    # and stays capturable.
    logical_packed_length = logical_packed_length.clone()

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    compiled = _compile_group_fused_kernel(
        dev.index, bool(stochastic_rounding), bool(use_fast_math)
    )
    compiled(
        A.t().unsqueeze(-1),
        rht_nk,
        col_fp4,
        col_sf.view(torch.uint32).flatten(),
        row_fp4.view(torch.uint64),
        row_sf,
        row_global_amax,
        col_global_amax,
        sr_rng_t,
        offsets,
        logical_packed_length,
        int(hidden),
        int(tokens),
        int(num_tensors),
        stream,
    )
    return col_fp4.view(torch.uint8), col_sf, row_fp4.view(torch.uint8), row_sf


class _Tcgen05GroupRhtAmax(_GroupRhtMainloop):
    """Per-group post-RHT columnwise amax and raw rowwise amax.

    Identical mainloop to the fused kernel; both epilogues reduce to a max-abs
    instead of quantizing. A 128-token tile lies entirely inside one group, so
    the group index is CTA-uniform and each warp contributes one atomic per
    tile. The running max is per tile, not per work item, because consecutive
    tiles can belong to different groups.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        col_amax_t: cute.Tensor,  # (num_tensors,) f32, pre-zeroed
        row_amax_t: cute.Tensor,  # (num_tensors,) f32, pre-zeroed
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        hidden: cutlass.Int32,
        tokens: cutlass.Int32,
        num_tensors: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            tile_sched_params,
            grid,
        ) = self._setup(mA, mB, hidden, tokens)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            col_amax_t,
            row_amax_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            num_tensors,
            tile_sched_params,
        ).launch(grid=grid, block=(TPB, 1, 1), stream=stream)

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
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_b_load_bytes: cutlass.Constexpr,
        tiles_in_n: cutlass.Int32,
        num_tensors: cutlass.Int32,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cutlass.Int32(32)

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, MAINLOOP_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
            clc_mbar: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, CLC_STAGES * 4]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, ROW_THREADS
            ),
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, COL_WARP_END - COL_WARP_BEGIN
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar.data_ptr(),
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TPB - 32),
            tx_count=16,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + COL_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=COL_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        if warp_idx == SCHED_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

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
            cute.slice_((M_TILE, N_TILE, TOKEN_TILE), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        work_tile = tile_sched.initial_work_tile_info()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, CLC_STAGES
        )
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)

        # ==================== TMA warp ====================
        if warp_idx == TMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, num_b_load_bytes
                )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0)],
                tBsB[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, tile_m, tile_n_base + k_tile, 0)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    ab_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==================== Scheduler warp ====================
        if warp_idx == SCHED_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, CLC_STAGES
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                tile_sched.advance_to_next_work(
                    clc_pipeline.producer_get_barrier(clc_producer_state)
                )
                clc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # ==================== Idle warp ====================
        if warp_idx == IDLE_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            tmem.wait_for_alloc()
            tCtAcc_base = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    for i in cutlass.range_constexpr(EPI_UNROLL):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        acc = tCtAcc_base[
                            (None, None, None, i, acc_producer_state.index)
                        ]
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrA[(None, None, i, ab_consumer_state.index)],
                            tCrB[(None, None, 0, 0)],
                            acc,
                        )
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                    ab_pipeline.consumer_release(
                        ab_consumer_state, pipeline.PipelineOp.TCGen05Mma
                    )
                    ab_consumer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== Columnwise amax (post-RHT, from TMEM) ====================
        if warp_idx >= COL_WARP_BEGIN and warp_idx < COL_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_COL)
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
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    run_max = cutlass.Float32(0.0)
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        token = (tile_n_base + k_tile) * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            # Crossing out of the cached group is the only point the
                            # running max has to reach memory: everything before it
                            # belongs to `g`, everything after to the next group.
                            _flush_group_max(run_max, col_amax_t, g, lane)
                            run_max = cutlass.Float32(0.0)
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                        acc_pipeline.consumer_wait(acc_consumer_state)
                        tile_max = cutlass.Float32(0.0)
                        for u in cutlass.range_constexpr(EPI_UNROLL):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[
                                    (
                                        None,
                                        None,
                                        None,
                                        0,
                                        0,
                                        u,
                                        acc_consumer_state.index,
                                    )
                                ],
                                tTR_rAcc,
                            )
                            vals = tTR_rAcc.load().reshape((16,))
                            for i in cutlass.range_constexpr(16):
                                tile_max = _max_f32(tile_max, _abs_f32(vals[i]))
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        run_max = _max_f32(run_max, _round_rht_amax(tile_max))
                    _flush_group_max(run_max, col_amax_t, g, lane)

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== Rowwise amax (raw A, from SMEM) ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < ROW_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_ROW)
            r_local = tidx - ROW_WARP_BEGIN * cutlass.Int32(32)
            hb = r_local % cutlass.Int32(ROW_HB)
            t0 = r_local // cutlass.Int32(ROW_HB)

            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    run_max = cutlass.Float32(0.0)
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        token = (tile_n_base + k_tile) * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            _flush_group_max(run_max, row_amax_t, g, lane)
                            run_max = cutlass.Float32(0.0)
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                        ab_pipeline.consumer_wait(ab_consumer_state)
                        stage = ab_consumer_state.index
                        tile_max = cutlass.Float32(0.0)
                        for p in cutlass.range_constexpr(ROW_PASSES):
                            tok = p * cutlass.Int32(ROW_TOK_PER_PASS) + t0
                            cute.autovec_copy(
                                cute.local_tile(
                                    sA_clean[(None, tok, stage)], (16,), (hb,)
                                ),
                                rBlk,
                            )
                            rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                            for j in cutlass.range_constexpr(8):
                                tile_max = _max_f32(
                                    tile_max, _abs_f32(_bf16lo_to_f32(rWords[j]))
                                )
                                tile_max = _max_f32(
                                    tile_max, _abs_f32(_bf16hi_to_f32(rWords[j]))
                                )
                        ab_pipeline.consumer_release(
                            ab_consumer_state, pipeline.PipelineOp.AsyncThread
                        )
                        ab_consumer_state.advance()

                        run_max = _max_f32(run_max, tile_max)
                    _flush_group_max(run_max, row_amax_t, g, lane)

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()


@functools.lru_cache(maxsize=None)
def _compile_group_amax_kernel(device_idx: int):
    """Compile the grouped RHT amax kernel with symbolic shapes."""
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)

    k = _Tcgen05GroupRhtAmax()
    # The TMEM->register op is selected from the row/col-majorness of the
    # columnwise output; the amax kernel has none, so a contiguous stand-in of
    # the same shape picks the same enum.
    dummy = torch.empty(
        (M_TILE, N_TILE), dtype=torch.int32, device=torch.device("cuda", device_idx)
    )
    k.c_layout = utils.LayoutEnum.from_tensor(from_dlpack(dummy))

    return cute.compile(
        k,
        make_fake_tensor(cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)),
        make_fake_tensor(
            cutlass.BFloat16,
            (HADAMARD_DIM, HADAMARD_DIM, 1),
            stride=(HADAMARD_DIM, 1, 1),
        ),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_rht_amax_impl(
    A: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    sign_vector=DEFAULT_SIGN_VECTOR,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``max|RHT(A.t())|`` and ``max|A|``.

    ``A`` is ``(tokens, hidden)`` bfloat16 row-major. Returns
    ``(col_amax, row_amax)``, each ``(num_tensors,)`` float32. The buffers start
    at zero because the epilogues accumulate with atomic max.
    """
    tokens, hidden = A.shape
    dev = A.device
    A = A.detach()

    col_amax = torch.zeros((num_tensors,), dtype=torch.float32, device=dev)
    row_amax = torch.zeros((num_tensors,), dtype=torch.float32, device=dev)
    rht_nk = _get_rht_buffer(tuple(sign_vector), dev.index)
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # See the fused kernel: the entry point requires byte_offset==0.
    logical_packed_length = logical_packed_length.clone()

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_amax_kernel(dev.index)(
        A.t().unsqueeze(-1),
        rht_nk,
        col_amax,
        row_amax,
        offsets,
        logical_packed_length,
        int(hidden),
        int(tokens),
        int(num_tensors),
        stream,
    )
    return col_amax, row_amax


def _work_tile_coord(work_tile):
    """(hidden tile, first token tile) for a scheduler work item."""
    coord = work_tile.tile_idx
    return coord[0], coord[1] * cutlass.Int32(K_TILE_MAX)


def _k_tile_count(tile_n_base, tiles_in_n):
    """Token tiles in this work item, clamped to the problem (TE :566)."""
    rem = tiles_in_n - tile_n_base
    return cutlass.select_(
        rem < cutlass.Int32(K_TILE_MAX), rem, cutlass.Int32(K_TILE_MAX)
    )


def _valid_tile_count(tile_n_base, n_all, tiles_in_n_valid):
    """Of this work item's ``n_all`` tiles, how many precede the logical bound."""
    rem = tiles_in_n_valid - tile_n_base
    rem = cutlass.select_(rem > cutlass.Int32(0), rem, cutlass.Int32(0))
    return cutlass.select_(rem < n_all, rem, n_all)


def _store_grouped_col_sf_u32(mSF_u32, rSF, r, c_base, g, offsets_t, hidden):
    """Store 8 columnwise scale bytes in group-local swizzled tiles.

    Each group owns a separately swizzled ``(hidden, group_tokens // 16)``
    scale buffer. ``mSF_u32`` is their flat concatenation, so the address is
    the span of preceding groups plus this hidden block's group-local
    64-token tile. The byte order within a tile matches the standard NVFP4
    swizzle used by ``_store_sf_byte``.
    """
    prev = cutlass.select_(g > cutlass.Int32(0), g - cutlass.Int32(1), 0)
    group_start = cutlass.select_(
        g > cutlass.Int32(0), offsets_t[prev], cutlass.Int32(0)
    )
    group_len = offsets_t[g] - group_start
    r_blk = r // cutlass.Int32(128)
    r_lane = r % cutlass.Int32(32)
    r_grp = (r % cutlass.Int32(128)) // cutlass.Int32(32)
    # A group has ``hidden * group_len / 64`` u32 scale words.  The preceding
    # concatenated groups cover the same expression with ``group_start``.
    # The product is taken in 64 bits: at DeepSeek-V3 671B (hidden 7168) it passes
    # 2^31 once ``group_start`` reaches 299,593 rows, and an Int32 multiply wraps
    # negative there, so the store lands far below the buffer. The quotient is at
    # most ``hidden * tokens / 64``, which is comfortably Int32, so only the
    # multiply needs widening and the index arithmetic below stays 32-bit.
    prefix_words = cutlass.Int32(
        cutlass.Int64(hidden) * cutlass.Int64(group_start) // cutlass.Int64(64)
    )
    words_per_hidden_block = group_len * cutlass.Int32(2)
    c_local = c_base - group_start // cutlass.Int32(16)
    # Plain Python loops: this helper is not AST-preprocessed, so the trace
    # unrolls them the same way cutlass.range_constexpr would inside a kernel.
    for half in range(2):
        packed = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        for i in range(4):
            packed[i] = rSF[half * 4 + i]
        word = cute.recast_tensor(packed, cutlass.Uint32)[0]
        word_col = c_local // cutlass.Int32(4) + cutlass.Int32(half)
        mSF_u32[
            prefix_words
            + r_blk * words_per_hidden_block
            + word_col * cutlass.Int32(128)
            + r_lane * cutlass.Int32(4)
            + r_grp
        ] = word


def _store_sf_byte(mSF, sf, r, c):
    """Scatter one swizzled scale-factor byte.

    The cutlass NVFP4 layout maps logical ``SF[r, c]`` to
    ``storage[r//128, c//4, r%32, (r%128//32)*4 + c%4]`` over a
    ``(R//128, C//4, 32, 16)`` buffer.
    """
    mSF[
        (
            r // cutlass.Int32(128),
            c // cutlass.Int32(4),
            r % cutlass.Int32(32),
            ((r % cutlass.Int32(128)) // cutlass.Int32(32)) * cutlass.Int32(4)
            + c % cutlass.Int32(4),
        )
    ] = sf
