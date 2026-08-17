# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared MXFP8 grouped blockscaled GEMM core: descriptors, mainloop, TMEM, T2R.

Everything between "torch tensors" and "an FP32 accumulator tile in registers".
The three kernels in this family plug different epilogues into this one mainloop
by passing a module-level function as the ``EPILOGUE`` Constexpr; see
``grouped_gemm_config.EPILOGUE_PROTOCOL_DOC``.

Three structural decisions, all descending from the per-expert row counts being
multiples of 128:

*No per-group tensormaps, anywhere.* Every operand is one host-built static TMA
descriptor over the whole tensor. Per-expert selection is an integer coordinate:
an L coordinate for the 3-D weight operands, and a K-tile index base for the
wgrad kernel's ragged contraction. The latter is exact because the blockscaled
scale-factor layout's ``Rest_K`` stride is a constant 512 bytes for *every* MN
row-block, so advancing the K tile index advances every row-block's byte address
by the same amount. That was verified on this wheel (probe V1) and it is what
deletes the tensormap workspace, the descriptor-init kernel, the descriptor
fences, and the padded-offset prefix sum that the reference kernels carry.

*No tile scheduler.* With the ragged axis tile-aligned, kernels A/B enumerate all
of ``[0, R/128)`` M tiles and kernel C's ``(N/128, K/128, G)`` grid is fully
static; only C's K-loop trip count is data-dependent. So there is no
``max_active_clusters`` query, no persistent loop, no offsets prefix scan, and no
work-tile shared-memory pipeline.

*The inactive tail needs no special code path.* A tile whose row base is at or
past the active row count runs with ``k_cnt == 0``: no TMA loads are issued (so
no inactive row is ever read), the accumulator fragment is zeroed in registers,
and the *unmodified* epilogue emits the zeros the contract requires. ``k_cnt`` is
CTA-uniform, so no barrier arrival count can desynchronize.

Expert lookup for the ragged-M kernels is an unrolled G-way scan over the
device-side offsets -- no host synchronization and no ``.item()``.
"""

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
from cutlass.utils import LayoutEnum

from torchao.prototype.moe_training.kernels.mxfp8.grouped_gemm_config import (
    SMEM_CAPACITY_BYTES,
    TMEM_TOTAL_COLS,
    GroupedGemmConfig,
    RaggedAxis,
)
from torchao.prototype.moe_training.kernels.mxfp8.grouped_mlp_epilogue import (
    validate_group_offsets_device,
)

__all__ = [
    "TileCoords",
    "activation_gemm_view",
    "weight_gemm_view",
    "make_operand_views",
    "make_tiled_mma",
    "make_sf_gemm_tensor",
    "t2r_partition",
    "dump_accumulator_epilogue",
    "grouped_gemm_kernel",
    "launch_grouped_gemm",
]


# ---------------------------------------------------------------------------
# Host-side operand views. Pure torch, no copies: every one of these is a
# restride of the caller's storage into the (MN, K, L) GEMM domain the core
# wants, with K contiguous.
# ---------------------------------------------------------------------------


def activation_gemm_view(t: torch.Tensor) -> torch.Tensor:
    """``[MN, K]`` K-contiguous -> ``(MN, K, 1)`` with a defined batch stride.

    Covers ``x_q``/``do_q`` directly, and both wgrad operands via their free
    transpose: a logical ``[R, N]`` with stride ``(1, R)`` *is* a K-contiguous
    ``[N, R]``, so pass ``t.t()``.
    """
    mn, k = t.shape
    if t.stride() != (k, 1):
        raise ValueError(
            f"GEMM operand must be K-contiguous with stride {(k, 1)}, got {t.stride()}"
        )
    return torch.as_strided(t, (mn, k, 1), (k, 1, mn * k))


def weight_gemm_view(w: torch.Tensor) -> torch.Tensor:
    """``[G, K, N]`` stride ``(K*N, 1, K)`` -> ``(N, K, G)`` stride ``(K, 1, K*N)``.

    That is the prequantized weight layout both kernels A and B receive, and the
    permute is exactly the K-major B operand the MMA wants, so the expert becomes
    an L coordinate and no descriptor is ever rebuilt.
    """
    g, k, n = w.shape
    if w.stride() != (k * n, 1, k):
        raise ValueError(
            f"grouped weight must have stride {(k * n, 1, k)}, got {w.stride()}"
        )
    return w.permute(2, 1, 0)


def make_operand_views(a: torch.Tensor, b: torch.Tensor):
    """``(mA, mB)`` in the GEMM domain, choosing the view from ``b``'s rank.

    A 3-D ``b`` is a grouped weight (expert becomes the L coordinate); a 2-D one
    is the wgrad case, where both operands are ungrouped and the expert is a
    K-tile index base instead.
    """
    return activation_gemm_view(a), (
        weight_gemm_view(b) if b.ndim == 3 else activation_gemm_view(b)
    )


@dataclass
class TileCoords:
    """The per-CTA tile description handed to the epilogue. All fields CTA-uniform.

    ``k_cnt == 0`` marks both cases where the mainloop is skipped: an inactive
    tail tile (ragged M) and a zero-token expert (ragged K). The epilogue must
    not branch on it -- the accumulator is already zero.
    """

    tile_m: Int32
    tile_n: Int32
    expert: Int32
    row_base: Int32
    col_base: Int32
    k_cnt: Int32


def make_tiled_mma(cfg: GroupedGemmConfig, a_dtype, b_dtype, sf_dtype):
    """The one blockscaled tiled MMA, K-major on both operands.

    No operand in this family is MN-major, so the 8-bit MN-major N-step and
    transpose-swizzle caveats never apply. ``MmaMXF8F6F4Op`` hard-wires FP32
    accumulation and instruction K=32, so a 128-element K tile is four MMA-K
    instructions and exactly one scale-factor atom.
    """
    return sm100_utils.make_blockscaled_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        sf_dtype,
        cfg.sf_vec_size,
        tcgen05.CtaGroup.ONE,
        (cfg.cta_tile_m, cfg.cta_tile_n),
    )


def make_sf_gemm_tensor(
    flat_sf: cute.Tensor, mn: int, k: int, l: int, sf_vec_size: int
):
    """Retile a flat blocked E8M0 buffer into the GEMM-domain scale-factor layout.

    The buffer is carried flat by ABI and may arrive as either uint8 or
    float8_e8m0fnu, so the pointer is recast unconditionally -- the MMA rejects a
    scale operand whose element type is not E8M0.

    ``tile_atom_to_shape_SF`` builds kernel IR, so this must be called inside a
    trace even though the shapes are static and every evaluation folds.
    """
    return cute.make_tensor(
        cute.recast_ptr(flat_sf.iterator, dtype=cutlass.Float8E8M0FNU),
        blockscaled_utils.tile_atom_to_shape_SF((mn, k, l), sf_vec_size),
    )


def t2r_partition(tidx, tAcc_base: cute.Tensor, cfg: GroupedGemmConfig):
    """Accumulator -> register handoff. See ``config.T2R_PARTITION_DOC``.

    ``elem_ty_d`` is Float32 even though the real outputs are E4M3: passing an
    8-bit d type steers ``get_tmem_load_op`` into the tmem_dp=16 layouts, which
    are shaped for a direct FP8 TMA store, not for a dual-quantization epilogue.

    ``tTR_cAcc`` carries each register's ``(row, col)`` in the CTA tile. Deriving
    every index from it, rather than from a raw register number, is what makes
    the gate/up de-interleave and the scale addressing correct by construction
    whatever the copy atom's internal value order is.
    """
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        cfg.cta_tile_shape_mnk,
        LayoutEnum.ROW_MAJOR,
        Float32,
        Float32,
        cfg.epi_tile,
        False,
    )
    # (MMA, MMA_M, MMA_N, ACC_STAGE) -> (CTA_M, CTA_N); one accumulator stage.
    tAcc_mn = tAcc_base[((None, None), 0, 0, 0)]
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
    tAcc_epi = cute.flat_divide(tAcc_mn, cfg.epi_tile)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
    cAcc_epi = cute.flat_divide(
        cute.make_identity_tensor((cfg.cta_tile_m, cfg.cta_tile_n)), cfg.epi_tile
    )
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N), values are (row, col) in the CTA tile
    tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
    # (T2R, T2R_M, T2R_N)
    tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[(None, None, None, 0, 0)].shape, Float32)
    return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_cAcc


def _s2t_copy_and_partition(sSF: cute.Tensor, tSF: cute.Tensor):
    """SMEM -> TMEM scale-factor copy, issued once per K tile from the MMA warp.

    ``Cp4x32x128bOp`` carries the warpx4 broadcast qualifier: issue it as a plain
    ``cute.copy`` and never wrap it in ``elect_one()``, which deadlocks because
    the compiler already inserts the election.
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


@cute.jit
def dump_accumulator_epilogue(
    tTR_rAcc,
    tTR_cAcc_s,
    tiled_copy_t2r,
    epi_tidx,
    subtile_idx: cutlass.Constexpr,
    tile: TileCoords,
    epi_smem,
    out,
    cfg: cutlass.Constexpr,
):
    """The M2a gate epilogue: write the raw FP32 accumulator to ``out[0]``.

    ``out[0]`` is ``[M, N, L]`` FP32, where L is 1 for a ragged-M kernel and the
    expert count for a ragged-K one. It exists to prove the blockscaled MMA and
    the scale-factor addressing before any real epilogue does; it is deliberately
    a scalar store loop, not a vectorized one.
    """
    gD = out[0]
    if cutlass.const_expr(cfg.ragged_axis is RaggedAxis.K):
        out_l = tile.expert
    else:
        out_l = Int32(0)
    for v in cutlass.range_constexpr(cute.size(tTR_rAcc)):
        crd = tTR_cAcc_s[v]
        gD[(tile.row_base + crd[0], tile.col_base + crd[1], out_l)] = tTR_rAcc[v]


@cute.kernel
def grouped_gemm_kernel(
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
    EPI_SMEM_BYTES: cutlass.Constexpr,
    VALIDATE_OFFSETS: cutlass.Constexpr,
):
    """One CTA computes one 128 x cta_tile_n output tile. Warps: 0 TMA, 1 MMA,
    4-7 epilogue, 2-3 idle."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, bidz = cute.arch.block_idx()

    num_groups = cutlass.const_expr(cute.size(offs, mode=[0]))
    num_k_tiles_full = cutlass.const_expr(cute.size(mA, mode=[1]) // cfg.cta_tile_k)

    # ------------------------------------------------------------------
    # Device-side precondition on the offset VALUES, which the host cannot
    # check without synchronizing. One block, one warp, one lane.
    # ------------------------------------------------------------------
    if cutlass.const_expr(VALIDATE_OFFSETS):
        if bidx == 0 and bidy == 0 and bidz == 0:
            if warp_idx == 0:
                with cute.arch.elect_one():
                    validate_group_offsets_device(
                        offs,
                        Int32(cute.size(mA, mode=[0]))
                        if cutlass.const_expr(cfg.ragged_axis is RaggedAxis.M)
                        else Int32(cute.size(mA, mode=[1])),
                    )

    # ------------------------------------------------------------------
    # Tile coordinates. No scheduler: the grid IS the tile enumeration.
    # ------------------------------------------------------------------
    tile_m = Int32(bidx)
    tile_n = Int32(bidy)
    if cutlass.const_expr(cfg.ragged_axis is RaggedAxis.M):
        row_base = tile_m * cfg.cta_tile_m
        # Unrolled G-way scan: the owning expert is the number of groups that end
        # at or before this tile's row base. A zero-token expert is never
        # selected, since its end equals its start.
        expert = Int32(0)
        for g in cutlass.range_constexpr(num_groups - 1):
            expert += Int32(offs[g] <= row_base)
        # Branch-free tail predicate: rows at or past offs[-1] belong to no
        # expert, so the whole mainloop is skipped for them.
        is_active = Int32(offs[num_groups - 1] > row_base)
        k_base = Int32(0)
        k_cnt = is_active * Int32(num_k_tiles_full)
        l_a = Int32(0)
        l_b = expert
    else:
        expert = Int32(bidz)
        row_base = tile_m * cfg.cta_tile_m
        # offs[expert - 1], clamped for expert 0; offsets are nonnegative so the
        # multiply is a legal select.
        prev = offs[cutlass.max(expert - Int32(1), Int32(0))] * Int32(expert > Int32(0))
        # Exact, not a ceil: every group boundary is a multiple of cta_tile_k.
        k_base = prev // cfg.cta_tile_k
        k_cnt = (offs[expert] - prev) // cfg.cta_tile_k
        l_a = Int32(0)
        l_b = Int32(0)
    tile = TileCoords(
        tile_m=tile_m,
        tile_n=tile_n,
        expert=expert,
        row_base=row_base,
        col_base=tile_n * cfg.cta_tile_n,
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
    epi_smem = None
    if cutlass.const_expr(EPI_SMEM_BYTES > 0):
        epi_smem = storage.sEpi.data_ptr()

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cfg.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
    )

    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar.data_ptr(),
        num_stages=cfg.num_ab_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        # Must be the EXACT byte count of all four TMA copies in one stage: too
        # small and the MMA consumes a partially arrived stage.
        tx_count=cutlass.const_expr(cfg.ab_stage_bytes),
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    # One accumulator stage, so this is a single mbarrier pair; there is no
    # inter-tile pipelining to overlap with.
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar.data_ptr(),
        num_stages=cfg.num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, cfg.num_epilogue_threads
        ),
        cta_layout_vmnk=cluster_layout_vmnk,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=cfg.tmem_alloc_barrier_id,
        num_threads=32 * (1 + len(cfg.epilogue_warp_ids)),
    )
    epilogue_barrier = pipeline.NamedBarrier(
        barrier_id=cfg.epilogue_sync_barrier_id,
        num_threads=cfg.num_epilogue_threads,
    )
    tmem_alloc = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=cfg.epilogue_warp_ids[0],
        is_two_cta=False,
    )

    # ------------------------------------------------------------------
    # Tile the global tensors. One static descriptor per operand; the expert is
    # an L coordinate (ragged M) or a K-tile index base (ragged K).
    # ------------------------------------------------------------------
    mma_tiler = cfg.mma_tiler_mnk
    gA = cute.local_tile(
        mA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
    )
    gB = cute.local_tile(
        mB, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
    )
    gSFA = cute.local_tile(
        mSFA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
    )
    gSFB = cute.local_tile(
        mSFB, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
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
    # Strip the stride-0 sf_vec_size sub-mode: the 512-byte scale atom is
    # contiguous and TMA moves it as 8-byte elements.
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

    tAgA_slice = tAgA[(None, tile_m, None, l_a)]
    tBgB_slice = tBgB[(None, tile_n, None, l_b)]
    tAgSFA_slice = tAgSFA[(None, tile_m, None, l_a)]
    tBgSFB_slice = tBgSFB[(None, tile_n, None, l_b)]

    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, cfg.num_acc_stage))

    # ------------------------------------------------------------------
    # Warp 0: TMA producer
    # ------------------------------------------------------------------
    if warp_idx == cfg.tma_warp_id:
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, cfg.num_ab_stage
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
    if warp_idx == cfg.mma_warp_id:
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
                mma_tiler,
                cfg.sf_vec_size,
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
                mma_tiler,
                cfg.sf_vec_size,
                cute.slice_(sfb_smem_layout, (None, None, None, 0)),
            ),
        )
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_copy_and_partition(sSFA, tCtSFA)
        s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_copy_and_partition(sSFB, tCtSFB)

        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, cfg.num_ab_stage
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, cfg.num_acc_stage
        )
        tCtAcc = tCtAcc_base[(None, None, None, 0)]

        # Acquire and commit unconditionally, including when k_cnt == 0, so the
        # accumulator handoff barrier stays balanced on tail tiles.
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
    if warp_idx >= cfg.epilogue_warp_ids[0]:
        # A power-of-two multiple of 32 columns is required; shared memory
        # already pins us to one CTA per SM, so taking the whole array is free.
        tmem_alloc.allocate(TMEM_TOTAL_COLS)
        tmem_alloc.wait_for_alloc()
        acc_tmem_ptr = tmem_alloc.retrieve_ptr(Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        epi_tidx = tidx - cfg.first_epilogue_thread
        tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_cAcc = t2r_partition(
            epi_tidx, tCtAcc_base, cfg
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, cfg.num_acc_stage
        )
        acc_pipeline.consumer_wait(acc_consumer_state)

        for s in cutlass.range_constexpr(cfg.num_epi_subtiles):
            if k_cnt == Int32(0):
                # Tail tile or zero-token expert: nothing was accumulated, so the
                # fragment is zeroed here and the epilogue runs unchanged.
                for v in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    tTR_rAcc[v] = Float32(0.0)
            else:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, s)], tTR_rAcc)
            EPILOGUE(
                tTR_rAcc,
                tTR_cAcc[(None, None, None, 0, s)],
                tiled_copy_t2r,
                epi_tidx,
                s,
                tile,
                epi_smem,
                out,
                cfg,
            )

        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)
        tmem_alloc.relinquish_alloc_permit()
        epilogue_barrier.arrive_and_wait()
        tmem_alloc.free(acc_tmem_ptr)


@cute.jit
def launch_grouped_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    sfa_flat: cute.Tensor,
    sfb_flat: cute.Tensor,
    offs: cute.Tensor,
    out,
    stream,
    cfg: cutlass.Constexpr,
    EPILOGUE: cutlass.Constexpr,
    EPI_SMEM_BYTES: cutlass.Constexpr = 0,
    VALIDATE_OFFSETS: cutlass.Constexpr = True,
):
    """Build the four static TMA descriptors and launch. Grid is data-independent.

    ``mA`` is the GEMM-domain ``(M, K, L)`` operand and ``mB`` the ``(N, K, L)``
    one, both K-major. ``sfa_flat`` / ``sfb_flat`` are the flat blocked E8M0
    buffers; they are retiled here, not by the caller. ``out`` is whatever tuple
    of destinations the epilogue expects.

    This is a trace body, not a launcher. Calling it directly retraces the whole
    kernel on every invocation -- 130 ms measured at these shapes. A public
    launcher must wrap it in its own ``@cute.jit`` entry point taking only the
    dynamic tensors, ``cute.compile`` that once behind a ``functools.cache``, and
    call the compiled executor (35 us). The Constexpr arguments must not be
    passed again to that executor; hand it the dynamic arguments only, or it
    raises "cannot be converted to pointer".

    ``VALIDATE_OFFSETS`` emits the device-side precondition check on the offset
    *values*, which the host cannot see without synchronizing. Note what it does
    and does not buy: ``cute.testing.assert_`` is compiled out entirely unless
    ``CUTE_DSL_ENABLE_ASSERTIONS=1`` is set in the environment, and when it does
    fire it traps the kernel and leaves the CUDA context unusable
    (``unspecified launch failure``) rather than raising cleanly. It is a
    debugging aid, not a guardrail; the host validators are the guardrail.
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

    if cutlass.const_expr(gemm_m % cfg.cta_tile_m != 0):
        raise ValueError(f"GEMM M {gemm_m} must be a multiple of {cfg.cta_tile_m}")
    if cutlass.const_expr(gemm_n % cfg.cta_tile_n != 0):
        raise ValueError(f"GEMM N {gemm_n} must be a multiple of {cfg.cta_tile_n}")
    if cutlass.const_expr(gemm_k % cfg.cta_tile_k != 0):
        raise ValueError(f"GEMM K {gemm_k} must be a multiple of {cfg.cta_tile_k}")

    mSFA = make_sf_gemm_tensor(sfa_flat, gemm_m, gemm_k, l_a, cfg.sf_vec_size)
    mSFB = make_sf_gemm_tensor(sfb_flat, gemm_n, gemm_k, l_b, cfg.sf_vec_size)

    tiled_mma = make_tiled_mma(cfg, a_dtype, b_dtype, sf_dtype)
    mma_tiler = cfg.mma_tiler_mnk
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cfg.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
    )

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler, a_dtype, cfg.num_ab_stage
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler, b_dtype, cfg.num_ab_stage
    )
    sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler, cfg.sf_vec_size, cfg.num_ab_stage
    )
    sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler, cfg.sf_vec_size, cfg.num_ab_stage
    )

    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        sm100_utils.cluster_shape_to_tma_atom_A(cfg.cluster_shape_mn, tiled_mma.thr_id),
        mA,
        cute.slice_(a_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        sm100_utils.cluster_shape_to_tma_atom_B(cfg.cluster_shape_mn, tiled_mma.thr_id),
        mB,
        cute.slice_(b_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # The 512-byte scale atom is contiguous; TMA must move it as 8-byte elements.
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        sm100_utils.cluster_shape_to_tma_atom_A(cfg.cluster_shape_mn, tiled_mma.thr_id),
        mSFA,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Uint64,
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        sm100_utils.cluster_shape_to_tma_atom_SFB(
            cfg.cluster_shape_mn, tiled_mma.thr_id
        ),
        mSFB,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Uint64,
    )

    # Round UP: flooring would hand back fewer bytes than the epilogue asked for
    # and its last store would land in sA.
    epi_words = cutlass.const_expr(max((EPI_SMEM_BYTES + 3) // 4, 1))

    @cute.struct
    class SharedStorage:
        ab_full_mbar: cute.struct.MemRange[cutlass.Int64, cfg.num_ab_stage]
        ab_empty_mbar: cute.struct.MemRange[cutlass.Int64, cfg.num_ab_stage]
        acc_full_mbar: cute.struct.MemRange[cutlass.Int64, cfg.num_acc_stage]
        acc_empty_mbar: cute.struct.MemRange[cutlass.Int64, cfg.num_acc_stage]
        tmem_holding_buf: cutlass.Int32
        sEpi: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, epi_words], 128]
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
    if cutlass.const_expr(smem_bytes > SMEM_CAPACITY_BYTES):
        # The struct's 1024-byte operand alignment costs a little more than the
        # config's arithmetic, so check the real number rather than the estimate.
        raise ValueError(
            f"shared memory request {smem_bytes} B exceeds the sm_100 capacity "
            f"{SMEM_CAPACITY_BYTES} B: lower num_ab_stage (currently "
            f"{cfg.num_ab_stage}) or the epilogue's {EPI_SMEM_BYTES} B request"
        )

    if cutlass.const_expr(cfg.ragged_axis is RaggedAxis.M):
        grid = (gemm_m // cfg.cta_tile_m, gemm_n // cfg.cta_tile_n, 1)
    else:
        grid = (gemm_m // cfg.cta_tile_m, gemm_n // cfg.cta_tile_n, num_groups)

    grouped_gemm_kernel(
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
        EPI_SMEM_BYTES,
        VALIDATE_OFFSETS,
    ).launch(
        grid=grid,
        block=(cfg.threads, 1, 1),
        cluster=(*cfg.cluster_shape_mn, 1),
        smem=smem_bytes,
        stream=stream,
    )
