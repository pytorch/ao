# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Optional, Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    compute_amax,
    compute_scale_from_amax,
    validate_group_sizes,
)


def _make_tile_smem_layouts(tile_m: int, tile_k: int):
    """Create shared memory layouts for input and output tiles.

    Both layouts use row-major format (K is fastest-changing dimension).

    Args:
        tile_m: Tile size in M dimension
        tile_k: Tile size in K dimension

    Returns:
        Tuple of (smem_layout_in, smem_layout_out), both for shared memory
    """
    import cutlass.cute as cute

    smem_layout_in = cute.make_layout(
        (tile_m, tile_k),
        stride=(tile_k, 1),
    )
    smem_layout_out = cute.make_layout(
        (tile_m, tile_k),
        stride=(tile_k, 1),
    )
    return smem_layout_in, smem_layout_out


# Config format:
# (compute_warps, tile_m, tile_k, k_tiles_per_cta)
_CUTEDSL_CONFIGS = {
    "bf16_default": (4, 128, 32, 4),
    "fallback": (6, 128, 32, 2),
}


def _select_cutedsl_config(
    input_dtype: torch.dtype,
    scaling_mode: str,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """Select kernel configuration based on input dtype.

    Args:
        input_dtype: Input dtype
        scaling_mode: Scaling mode ("floor" or "rceil")

    Returns:
        Tuple of (config_name, (compute_warps, tile_m, tile_k, k_tiles_per_cta))
    """
    if input_dtype == torch.bfloat16:
        config_name = "bf16_default"
    else:
        config_name = "fallback"
    return config_name, _CUTEDSL_CONFIGS[config_name]


@functools.cache
def _compile_mxfp8_quantize_2d_cutedsl(
    input_dtype_name: str,
    scaling_mode: str,
    compute_warps: int,
    tile_m: int,
    tile_k: int,
    requested_stage_count: int,
    k_tiles_per_cta: int,
    is_full_k_tiles: bool,
    blocked_scale_output: bool,
    has_offs: bool = False,
):
    """Compile the 2D MXFP8 quantization kernel using CuTeDSL.

    Uses warp-specialized TMA kernel with:
    - Warp 0: Producer (issues TMA global→shared and shared→global)
    - Warps 1..compute_warps: Consumers (quantize in registers)

    Args:
        input_dtype_name: Input dtype ("torch.float32" or "torch.bfloat16")
        scaling_mode: Scaling mode ("floor" or "rceil")
        compute_warps: Number of compute warps
        tile_m: Tile size in M dimension
        tile_k: Tile size in K dimension
        requested_stage_count: Requested pipeline stages (capped by k_tiles_per_cta)
        k_tiles_per_cta: Number of K tiles per CTA
        is_full_k_tiles: Whether K dimension is perfectly tiled
        blocked_scale_output: Whether to output scales in blocked layout for tcgen05

    Returns:
        Compiled CuTeDSL kernel callable
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    if input_dtype_name == "torch.float32":
        INPUT_CUTLASS_DTYPE = cutlass.Float32
    elif input_dtype_name == "torch.bfloat16":
        INPUT_CUTLASS_DTYPE = cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported input dtype for CuTeDSL quantize_2d: {input_dtype_name}"
        )

    # Warp-specialized TMA kernel:
    # - warp 0: producer (issues TMA G2S and S2G)
    # - warps [1..compute_warps]: consumers (quantize)
    # Note: we intentionally keep store on warp 0 (no dedicated store
    # warp).  A split load-warp/store-warp design was tested and
    # mostly regressed throughput, so this layout is the tuned
    # default.
    COMPUTE_WARPS = compute_warps
    TILE_M = tile_m
    TILE_K = tile_k
    K_TILES_PER_CTA = k_tiles_per_cta
    IS_FULL_K_TILES_VALUE = is_full_k_tiles
    BLOCKED_SCALE_OUTPUT_VALUE = blocked_scale_output

    THREADS_PER_BLOCK = (1 + COMPUTE_WARPS) * 32
    assert COMPUTE_WARPS >= 1
    assert TILE_M > 0 and TILE_K > 0
    assert TILE_K % 32 == 0

    SCALE_DIM_K_VALUE = 32
    K_BLOCKS_PER_TILE = TILE_K // SCALE_DIM_K_VALUE
    assert K_BLOCKS_PER_TILE > 0
    assert requested_stage_count >= 1
    # B200 sweeps on our representative shapes showed no benefit
    # beyond 2 stages. We keep stage setup generic so future tuning can
    # revisit this, but the current tuned contract is 1 or 2 stages.
    assert requested_stage_count <= 2
    assert K_TILES_PER_CTA >= 1
    STAGE_COUNT_VALUE = min(requested_stage_count, K_TILES_PER_CTA)

    input_elem_bytes = 4 if input_dtype_name == "torch.float32" else 2
    # SMEM_STORE_VEC is used both as an element count (make_layout) and as a byte
    # count (.align(16)); these coincide only because the output is 1-byte
    # Float8E4M3FN. The stores and the vectorized loads also index SMEM by raw
    # iterator arithmetic, which assumes the contiguous row-/col-major layouts
    # built in _make_tile_smem_layouts. Revisit both if the output dtype changes
    # or a swizzled SMEM layout is introduced.
    SMEM_STORE_VEC = 16
    assert SCALE_DIM_K_VALUE % SMEM_STORE_VEC == 0
    TILE_COPY_BYTES = TILE_M * TILE_K * input_elem_bytes
    M_THREADS = COMPUTE_WARPS * 32
    M_ITERS_PER_LANE = ceil_div(TILE_M, M_THREADS)

    @cute.struct
    class SharedStorage:
        tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, STAGE_COUNT_VALUE]
        in_smem: cute.struct.Align[
            cute.struct.MemRange[
                INPUT_CUTLASS_DTYPE, STAGE_COUNT_VALUE * TILE_M * TILE_K
            ],
            128,
        ]
        out_smem: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float8E4M3FN, STAGE_COUNT_VALUE * TILE_M * TILE_K
            ],
            128,
        ]

    class Mxfp8Quantize2dKernel:
        @cute.jit
        def _load_block_full_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
        ):
            """Load a full 32-element quantization block from shared memory to registers.

            Loads all elements without bounds checking.

            Args:
                sIN_tile: Input tile in shared memory (TILE_M, TILE_K)
                m_rel: Row index within tile
                k_base: Starting K index for this block within tile

            Returns:
                vals_block: 32 input elements in register memory
            """
            vals_block = cute.make_rmem_tensor((SCALE_DIM_K_VALUE,), cutlass.Float32)
            raw = cute.make_rmem_tensor((SCALE_DIM_K_VALUE,), INPUT_CUTLASS_DTYPE)
            cute.autovec_copy(
                cute.make_tensor(
                    (sIN_tile.iterator + (m_rel * TILE_K + k_base)).align(16),
                    cute.make_layout(SCALE_DIM_K_VALUE),
                ),
                raw,
            )
            for i in range(SCALE_DIM_K_VALUE):
                vals_block[i] = cutlass.Float32(raw[i])
            return vals_block

        @cute.jit
        def _store_scales_reg_to_gmem_vec(
            self,
            scales_tensor: cute.Tensor,
            m: cutlass.Int64,
            k_block_base: cutlass.Int64,
            scale_buffer: cute.Tensor,
            num_scales: cutlass.Int32,
            BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
        ):
            """Store scales from registers to global memory using vectorized writes when possible.

            Uses uint32 vectorized writes for 4 scales in blocked layout.

            Args:
                scales_tensor: Output scales in global memory
                m: Global M coordinate
                k_block_base: Starting K block index
                scale_buffer: Buffer of scales in register memory (uint8)
                num_scales: Number of scales to store
                BLOCKED_SCALE_OUTPUT: Whether using blocked layout (enables vectorization)

            Storage locations:
                Input: scale_buffer (registers)
                Output: scales_tensor (global memory)
            """
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
                # Blocked layout with 4 contiguous scales - write as uint32
                if num_scales == 4:
                    # Pack 4 uint8 scales into uint32 and write
                    scales_tensor_u32 = cute.recast_tensor(
                        scales_tensor, cutlass.Uint32
                    )
                    scale_buffer_u32 = cute.recast_tensor(scale_buffer, cutlass.Uint32)
                    scales_tensor_u32[m, k_block_base // cutlass.Int64(4)] = (
                        scale_buffer_u32[0]
                    )
                else:
                    # Fallback for non-4 cases (e.g., tail tiles)
                    for i in range(num_scales):
                        k_block = k_block_base + i
                        scales_tensor[m, k_block] = scale_buffer[i]
            else:
                # Row-major layout - scalar stores
                for i in range(num_scales):
                    k_block = k_block_base + i
                    scales_tensor[m, k_block] = scale_buffer[i]

        @cute.jit
        def _quantize_then_store_reg_to_smem(
            self,
            vals_group: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            sout_base: cutlass.Int32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            """Quantize SMEM_STORE_VEC elements to FP8 and store them as one vector.

            Applies inverse scale and converts to FP8 with saturation.
            The whole group is one vector op so no per-chunk register staging is
            needed, and the shared store is a single SMEM_STORE_VEC-byte access.

            Args:
                vals_group: SMEM_STORE_VEC input elements in register memory
                inv_scale: Inverse scale in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K)
                m_rel: Row index within tile
                sout_base: Starting K index for this group within tile
                USE_RCEIL: Scale calculation mode (kept for the shared call signature)

            Storage locations:
                Inputs: vals_group, inv_scale (registers)
                Output: sOUT_tile (shared memory)
            """
            q_vec = vals_group.load() * inv_scale
            q_fp8 = cute.make_rmem_tensor((SMEM_STORE_VEC,), cutlass.Float8E4M3FN)
            q_fp8.store(q_vec.to(cutlass.Float8E4M3FN))
            cute.autovec_copy(
                q_fp8,
                cute.make_tensor(
                    (
                        sOUT_tile.iterator
                        + cute.crd2idx((m_rel, sout_base), sOUT_tile.layout)
                    ).align(16),
                    cute.make_layout(SMEM_STORE_VEC),
                ),
            )

        @cute.jit
        def _quantize_block_then_store_reg_to_smem_full(
            self,
            vals_block: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            """Quantize and store a full 32-element block by processing 8 chunks of 4 elements.

            Args:
                vals_block: 32 input elements in register memory
                inv_scale: Inverse scale in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K)
                m_rel: Row index within tile
                k_base: Starting K index for this block within tile
                USE_RCEIL: Whether using RCEIL mode or FLOOR mode

            Storage locations:
                Inputs: vals_block, inv_scale (registers)
                Output: sOUT_tile (shared memory)
            """
            for g in range(SCALE_DIM_K_VALUE // SMEM_STORE_VEC):
                local_base = g * SMEM_STORE_VEC
                vals_group = cute.make_rmem_tensor((SMEM_STORE_VEC,), cutlass.Float32)
                for i in range(SMEM_STORE_VEC):
                    vals_group[i] = vals_block[local_base + i]
                self._quantize_then_store_reg_to_smem(
                    vals_group,
                    inv_scale,
                    sOUT_tile,
                    m_rel,
                    k_base + local_base,
                    USE_RCEIL,
                )

        @cute.jit
        def _issue_tma_load(
            self,
            tma_atom_in: cute.CopyAtom,
            gIN_tile: cute.Tensor,
            sIN_tile: cute.Tensor,
            tma_mbar_ptr: cutlass.Int64,
            warp_idx: cutlass.Int32,
        ):
            """Issue TMA load from global memory to shared memory (producer warp only).

            Only warp 0 executes the TMA load and updates the barrier.

            Args:
                tma_atom_in: TMA copy atom for G2S
                gIN_tile: Input tile in global memory (TILE_M, TILE_K)
                sIN_tile: Input tile in shared memory (TILE_M, TILE_K)
                tma_mbar_ptr: TMA barrier pointer
                warp_idx: Warp index

            Storage locations:
                Source: gIN_tile (global memory)
                Destination: sIN_tile (shared memory)
            """
            if warp_idx == 0:
                cta_layout = cute.make_layout((1,))
                sIN_for_tma_partition = cute.group_modes(sIN_tile, 0, 2)
                gIN_for_tma_partition = cute.group_modes(gIN_tile, 0, 2)
                tINs, tINg = cpasync.tma_partition(
                    tma_atom_in,
                    0,
                    cta_layout,
                    sIN_for_tma_partition,
                    gIN_for_tma_partition,
                )
                tINg_stage0 = tINg[None]
                tINs_stage0 = tINs[None]
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        tma_mbar_ptr, TILE_COPY_BYTES
                    )
                cute.copy(
                    tma_atom_in,
                    tINg_stage0,
                    tINs_stage0,
                    tma_bar_ptr=tma_mbar_ptr,
                )

        @cute.jit
        def _issue_tma_store(
            self,
            tma_atom_out: cute.CopyAtom,
            gOUT_tile: cute.Tensor,
            sOUT_tile: cute.Tensor,
            warp_idx: cutlass.Int32,
        ):
            """Issue TMA store from shared memory to global memory (producer warp only).

            Synchronizes threads before store. Only warp 0 executes the TMA store.

            Args:
                tma_atom_out: TMA copy atom for S2G
                gOUT_tile: Output tile in global memory (TILE_M, TILE_K)
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K)
                warp_idx: Warp index

            Storage locations:
                Source: sOUT_tile (shared memory)
                Destination: gOUT_tile (global memory)
            """
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            cute.arch.sync_threads()
            if warp_idx == 0:
                cta_layout = cute.make_layout((1,))
                sOUT_for_tma_partition = cute.group_modes(sOUT_tile, 0, 2)
                gOUT_for_tma_partition = cute.group_modes(gOUT_tile, 0, 2)
                tOUTs, tOUTg = cpasync.tma_partition(
                    tma_atom_out,
                    0,
                    cta_layout,
                    sOUT_for_tma_partition,
                    gOUT_for_tma_partition,
                )
                tOUTs_stage0 = tOUTs[None]
                tOUTg_stage0 = tOUTg[None]
                cute.copy(
                    tma_atom_out,
                    tOUTs_stage0,
                    tOUTg_stage0,
                )

        @cute.kernel
        def kernel(
            self,
            inp_mk: cute.Tensor,
            tma_atom_in: cute.CopyAtom,
            tma_tensor_in: cute.Tensor,
            out_mk: cute.Tensor,
            tma_atom_out: cute.CopyAtom,
            tma_tensor_out: cute.Tensor,
            scales_out_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            blocked_scale_layout: cute.Layout,
            offs: Optional[cute.Tensor],
            SCALE_DIM_K: cutlass.Constexpr[int],
            USE_RCEIL: cutlass.Constexpr[bool],
            IS_FULL_K_TILES: cutlass.Constexpr[bool],
            STAGE_COUNT: cutlass.Constexpr[int],
        ):
            """Main MXFP8 quantization kernel with warp specialization and TMA pipeline.

            Warp roles:
            - Warp 0: Producer (TMA loads/stores)
            - Warps 1..compute_warps: Consumers (quantize in registers)

            Pipeline stages:
            - Stage 0: Load tile to shared memory, quantize, store to global
            - Stage 1 (if enabled): Prefetch next tile while processing current

            Args:
                inp_mk: Input tensor in global memory (M, K)
                tma_atom_in: TMA copy atom for G2S
                tma_tensor_in: TMA tensor view for input
                out_mk: Output tensor in global memory (M, K)
                tma_atom_out: TMA copy atom for S2G
                tma_tensor_out: TMA tensor view for output
                scales_out_u8: Output scales tensor in global memory (M, K//32) or blocked layout
                M: M dimension size
                K: K dimension size
                k_blocks: Number of 32-element blocks in K
                m_cta_tiles: Number of tiles in M dimension
                k_cta_tiles: Number of tile groups in K dimension
                blocked_scale_layout: Layout for blocked scale output
                offs: Tensor of group end offsets for validation
                SCALE_DIM_K: Block size (32)
                USE_RCEIL: Whether using RCEIL mode
                IS_FULL_K_TILES: Whether K is perfectly tiled
                STAGE_COUNT: Number of pipeline stages

            Storage locations:
                Inputs: inp_mk (global memory)
                Outputs: out_mk, scales_out_u8 (global memory)
                Intermediate: shared memory for tiles, registers for computation
            """
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
            bidx, bidy, _ = cute.arch.block_idx()

            # Validate group sizes are multiples of 128 if offs is provided
            if cutlass.const_expr(offs is not None):
                if tidx == 0:
                    validate_group_sizes(offs)

            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            # The tuned contract keeps STAGE_COUNT <= 2.
            tma_mbar_ptr0 = storage.tma_mbar_ptr.data_ptr()
            tma_mbar_ptr1 = tma_mbar_ptr0
            if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                tma_mbar_ptr1 = tma_mbar_ptr0 + 1

            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(TILE_M, TILE_K)
            staged_layout_in = cute.make_layout(
                (STAGE_COUNT_VALUE, TILE_M, TILE_K),
                stride=(TILE_M * TILE_K, TILE_K, 1),
            )
            staged_layout_out = cute.make_layout(
                (STAGE_COUNT_VALUE, TILE_M, TILE_K),
                stride=(TILE_M * TILE_K, TILE_K, 1),
            )
            sIN_staged = storage.in_smem.get_tensor(staged_layout_in)
            sOUT_staged = storage.out_smem.get_tensor(staged_layout_out)
            stage_elems = TILE_M * TILE_K
            sIN_tile0 = cute.make_tensor(
                sIN_staged.iterator + 0 * stage_elems, smem_layout_in
            )
            sOUT_tile0 = cute.make_tensor(
                sOUT_staged.iterator + 0 * stage_elems, smem_layout_out
            )
            sIN_tile1 = sIN_tile0
            sOUT_tile1 = sOUT_tile0
            if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                sIN_tile1 = cute.make_tensor(
                    sIN_staged.iterator + 1 * stage_elems, smem_layout_in
                )
                sOUT_tile1 = cute.make_tensor(
                    sOUT_staged.iterator + 1 * stage_elems, smem_layout_out
                )

            if tidx == 0:
                cpasync.prefetch_descriptor(tma_atom_in)
                cpasync.prefetch_descriptor(tma_atom_out)
                cute.arch.mbarrier_init(tma_mbar_ptr0, 1)
                if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                    cute.arch.mbarrier_init(tma_mbar_ptr1, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            k_tile_group_idx = cutlass.Int64(bidx)
            m_tile = cutlass.Int64(bidy)
            m0 = m_tile * TILE_M
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                scales_tensor = cute.make_tensor(
                    scales_out_u8.iterator,
                    blocked_scale_layout,
                )
            else:
                scales_tensor = scales_out_u8
            for tile_step in cutlass.range_constexpr(K_TILES_PER_CTA):
                k_tile_eff = k_tile_group_idx * K_TILES_PER_CTA + tile_step

                stage_idx = tile_step % STAGE_COUNT

                sIN_tile = sIN_tile0
                sOUT_tile = sOUT_tile0
                tma_mbar_ptr = tma_mbar_ptr0
                if cutlass.const_expr(STAGE_COUNT > 1):
                    tma_mbar_ptr = tma_mbar_ptr0 + stage_idx
                if cutlass.const_expr(STAGE_COUNT > 1):
                    if stage_idx == 1:
                        sIN_tile = sIN_tile1
                        sOUT_tile = sOUT_tile1

                tma_phase = (tile_step // STAGE_COUNT) % 2

                if cutlass.const_expr(
                    tile_step == 0 or not (STAGE_COUNT > 1 and K_TILES_PER_CTA > 1)
                ):
                    gIN_tile = cute.local_tile(
                        tma_tensor_in, (TILE_M, TILE_K), (m_tile, k_tile_eff)
                    )
                    self._issue_tma_load(
                        tma_atom_in,
                        gIN_tile,
                        sIN_tile,
                        tma_mbar_ptr,
                        warp_idx,
                    )

                if cutlass.const_expr(STAGE_COUNT > 1 and K_TILES_PER_CTA > 1):
                    if cutlass.const_expr(tile_step + 1 < K_TILES_PER_CTA):
                        k_tile_next = k_tile_group_idx * K_TILES_PER_CTA + tile_step + 1
                        next_stage_idx = (tile_step + 1) % STAGE_COUNT
                        sIN_tile_next = sIN_tile0
                        tma_mbar_ptr_next = tma_mbar_ptr0
                        if cutlass.const_expr(STAGE_COUNT > 1):
                            tma_mbar_ptr_next = tma_mbar_ptr0 + next_stage_idx
                        if cutlass.const_expr(STAGE_COUNT > 1):
                            if next_stage_idx == 1:
                                sIN_tile_next = sIN_tile1

                        gIN_tile_next = cute.local_tile(
                            tma_tensor_in, (TILE_M, TILE_K), (m_tile, k_tile_next)
                        )
                        self._issue_tma_load(
                            tma_atom_in,
                            gIN_tile_next,
                            sIN_tile_next,
                            tma_mbar_ptr_next,
                            warp_idx,
                        )

                if warp_idx >= 1 and warp_idx <= compute_warps:
                    cute.arch.mbarrier_wait(tma_mbar_ptr, tma_phase)
                    lane = tidx % 32
                    m_lane = (warp_idx - 1) * 32 + lane

                    for mm in cutlass.range_constexpr(M_ITERS_PER_LANE):
                        m_rel = m_lane + mm * M_THREADS
                        m = m0 + m_rel
                        if cutlass.const_expr(IS_FULL_K_TILES):
                            if m_rel < TILE_M:
                                # Buffer scales for vectorized store
                                scale_buffer = cute.make_rmem_tensor(
                                    (K_BLOCKS_PER_TILE,), cutlass.Uint8
                                )

                                for kb in cutlass.range_constexpr(K_BLOCKS_PER_TILE):
                                    k_base = kb * SCALE_DIM_K_VALUE
                                    vals_block = self._load_block_full_smem_to_reg(
                                        sIN_tile,
                                        m_rel,
                                        k_base,
                                    )

                                    amax = compute_amax(vals_block)

                                    scale_biased, inv_scale = compute_scale_from_amax(
                                        amax, USE_RCEIL
                                    )
                                    scale_buffer[kb] = scale_biased

                                    self._quantize_block_then_store_reg_to_smem_full(
                                        vals_block,
                                        inv_scale,
                                        sOUT_tile,
                                        m_rel,
                                        k_base,
                                        USE_RCEIL,
                                    )

                                # Vectorized scale store
                                k_block_base = k_tile_eff * K_BLOCKS_PER_TILE
                                self._store_scales_reg_to_gmem_vec(
                                    scales_tensor,
                                    m,
                                    k_block_base,
                                    scale_buffer,
                                    cutlass.Int32(K_BLOCKS_PER_TILE),
                                    BLOCKED_SCALE_OUTPUT_VALUE,
                                )
                        else:
                            m_in_bounds = m < M
                            if m_rel < TILE_M and m_in_bounds:
                                # Buffer scales for vectorized store
                                scale_buffer = cute.make_rmem_tensor(
                                    (K_BLOCKS_PER_TILE,), cutlass.Uint8
                                )
                                num_valid_scales = cutlass.Int32(0)

                                for kb in cutlass.range_constexpr(K_BLOCKS_PER_TILE):
                                    k_block = k_tile_eff * K_BLOCKS_PER_TILE + kb
                                    if k_block < k_blocks:
                                        k_base = kb * SCALE_DIM_K_VALUE
                                        vals_block = self._load_block_full_smem_to_reg(
                                            sIN_tile,
                                            m_rel,
                                            k_base,
                                        )

                                        amax = compute_amax(vals_block)

                                        scale_biased, inv_scale = (
                                            compute_scale_from_amax(amax, USE_RCEIL)
                                        )
                                        scale_buffer[num_valid_scales] = scale_biased
                                        num_valid_scales = num_valid_scales + 1

                                        self._quantize_block_then_store_reg_to_smem_full(
                                            vals_block,
                                            inv_scale,
                                            sOUT_tile,
                                            m_rel,
                                            k_base,
                                            USE_RCEIL,
                                        )

                                # Vectorized scale store
                                if num_valid_scales > 0:
                                    k_block_base = k_tile_eff * K_BLOCKS_PER_TILE
                                    self._store_scales_reg_to_gmem_vec(
                                        scales_tensor,
                                        m,
                                        k_block_base,
                                        scale_buffer,
                                        num_valid_scales,
                                        BLOCKED_SCALE_OUTPUT_VALUE,
                                    )

                gOUT_tile = cute.local_tile(
                    tma_tensor_out, (TILE_M, TILE_K), (m_tile, k_tile_eff)
                )
                self._issue_tma_store(
                    tma_atom_out,
                    gOUT_tile,
                    sOUT_tile,
                    warp_idx,
                )

        @cute.jit
        def __call__(
            self,
            inp_mk: cute.Tensor,
            out_mk: cute.Tensor,
            scales_out_u8: cute.Tensor,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            stream: cuda.CUstream,
            offs: Optional[cute.Tensor],
        ):
            """Kernel launcher that sets up TMA descriptors and blocked scale layout.

            Args:
                inp_mk: Input tensor in global memory (M, K)
                out_mk: Output quantized data tensor in global memory (M, K)
                scales_out_u8: Output scales tensor in global memory (M, K//32) or blocked layout
                M: M dimension size
                K: K dimension size
                k_blocks: Number of 32-element blocks in K
                m_cta_tiles: Number of tiles in M dimension
                k_cta_tiles: Number of tile groups in K dimension
                stream: CUDA stream
                offs: Tensor of group end offsets for validation (group sizes must be multiples of 128)

            Storage locations:
                All tensors in global memory
            """
            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(TILE_M, TILE_K)
            # Use tcgen05.CtaGroup.ONE for the optimised single-CTA Blackwell (SM 10.x) TMA load path.
            g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
                g2s_op,
                inp_mk,
                smem_layout_in,
                (TILE_M, TILE_K),
            )
            tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_mk,
                smem_layout_out,
                (TILE_M, TILE_K),
            )

            blocked_scale_layout = cute.make_layout((1,))
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                padded_scale_cols = cute.round_up(k_blocks, 4)
                m_block_tiles = cute.ceil_div(M, 128)
                k_block_tiles = padded_scale_cols // cutlass.Int64(4)
                blocked_scale_layout = cute.make_layout(
                    ((32, 4, m_block_tiles), (4, k_block_tiles)),
                    stride=(
                        (16, 4, cutlass.Int64(128) * padded_scale_cols),
                        (1, cutlass.Int64(512)),
                    ),
                )

            self.kernel(
                inp_mk,
                tma_atom_in,
                tma_tensor_in,
                out_mk,
                tma_atom_out,
                tma_tensor_out,
                scales_out_u8,
                M,
                K,
                k_blocks,
                m_cta_tiles,
                k_cta_tiles,
                blocked_scale_layout,
                offs,
                SCALE_DIM_K=SCALE_DIM_K_VALUE,
                USE_RCEIL=(scaling_mode == "rceil"),
                IS_FULL_K_TILES=IS_FULL_K_TILES_VALUE,
                STAGE_COUNT=STAGE_COUNT_VALUE,
            ).launch(
                grid=(k_cta_tiles, m_cta_tiles, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    kernel = Mxfp8Quantize2dKernel()

    m = cute.sym_int(divisibility=32)
    k = cute.sym_int(divisibility=32)
    kb = cute.sym_int()
    inp_stride0 = cute.sym_int()
    inp_stride1 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()
    scale_stride0 = cute.sym_int()
    scale_stride1 = cute.sym_int()

    fake_inp = make_fake_tensor(
        INPUT_CUTLASS_DTYPE,
        (m, k),
        stride=(inp_stride0, inp_stride1),
    )
    fake_out = make_fake_tensor(
        cutlass.Float8E4M3FN,
        (m, k),
        stride=(out_stride0, out_stride1),
    )
    if blocked_scale_output:
        scale_flat = cute.sym_int()
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (scale_flat,),
            stride=(scale_stride0,),
        )
    else:
        fake_scales = make_fake_tensor(
            cutlass.Uint8,
            (m, kb),
            stride=(scale_stride0, scale_stride1),
        )
    fake_stream = make_fake_stream()

    if has_offs:
        offs_stride = cute.sym_int()
        fake_offs = make_fake_tensor(
            cutlass.Int32,
            (cute.sym_int(),),
            stride=(offs_stride,),
        )
    else:
        fake_offs = None

    compile_options = (
        "--enable-tvm-ffi"
        if fake_offs is None
        else "--enable-tvm-ffi --enable-assertions"
    )
    return cute.compile(
        kernel,
        inp_mk=fake_inp,
        out_mk=fake_out,
        scales_out_u8=fake_scales,
        M=0,
        K=0,
        k_blocks=0,
        m_cta_tiles=1,
        k_cta_tiles=1,
        stream=fake_stream,
        offs=fake_offs,
        options=compile_options,
    )


def mxfp8_quantize_cutedsl_2d_1x32(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
    offs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to MXFP8 format using CuTe DSL kernel.

    Quantizes along the K dimension - each row has K//32 scales, one per block of 32 K elements.

    Args:
        x: Input tensor of shape (M, K)
        block_size: Block size for quantization along K (only 32 supported)
        scaling_mode: Scaling mode ("floor" or "rceil")
        stage_count: Number of pipeline stages (1 or 2)
        blocked_scale_output: Whether to output scales in blocked layout
        offs: Optional tensor of group end offsets for validation (must have group sizes as multiples of 128)

    Returns:
        q_data: Quantized data in row-major layout with shape (M, K)
        scales: Scales tensor with shape (M, K//32) or blocked layout
    """
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    M, K = x.shape
    assert K % 128 == 0, "K must be divisible by 128"
    assert M % 128 == 0, "M must be divisible by 128"

    if offs is not None:
        assert offs.is_cuda, "offs tensor must be CUDA"
        assert offs.dtype == torch.int32, "offs must be int32 tensor"
        assert offs.dim() == 1, "offs must be 1D tensor"

    _, config = _select_cutedsl_config(x.dtype, scaling_mode)
    compute_warps, tile_m, tile_k, k_tiles_per_cta = config
    # B200 sweeps over representative shapes showed no
    # measurable benefit above 2 stages. We keep this configurable for
    # benchmarking, and the effective stage count remains capped by
    # k_tiles_per_cta below.
    assert stage_count >= 1, "stage_count must be >= 1"
    assert stage_count <= 2, "stage_count must be <= 2"
    is_full_k_tiles = K % (tile_k * k_tiles_per_cta) == 0
    is_sm_10x = torch.cuda.get_device_capability()[0] == 10
    if blocked_scale_output and not is_sm_10x:
        raise NotImplementedError(
            "blocked_scale_output is only supported on SM 10.x GPUs "
            "because it produces the tcgen05 blocked scale layout"
        )

    # Output in row-major layout: stride (K, 1).
    q_data = torch.empty_strided(
        (M, K),
        (K, 1),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    k_blocks = K // block_size
    padded_scale_rows = ceil_div(M, 128) * 128
    padded_scale_cols = ceil_div(k_blocks, 4) * 4
    if blocked_scale_output:
        scales_u8 = torch.empty(
            (padded_scale_rows * padded_scale_cols,),
            device=x.device,
            dtype=torch.uint8,
        )
    else:
        scales_u8 = torch.empty(
            (M, k_blocks),
            device=x.device,
            dtype=torch.uint8,
        )

    compiled = _compile_mxfp8_quantize_2d_cutedsl(
        str(x.dtype),
        scaling_mode,
        compute_warps,
        tile_m,
        tile_k,
        stage_count,
        k_tiles_per_cta,
        is_full_k_tiles,
        blocked_scale_output,
        offs is not None,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    m_cta_tiles = ceil_div(M, tile_m)
    k_cta_tiles = ceil_div(K, tile_k * k_tiles_per_cta)

    compiled(
        x,
        q_data,
        scales_u8,
        int(M),
        int(K),
        int(k_blocks),
        int(m_cta_tiles),
        int(k_cta_tiles),
        stream,
        offs,
    )
    scales = scales_u8.view(torch.float8_e8m0fnu)
    scales = (
        scales.view(padded_scale_rows, padded_scale_cols)
        if blocked_scale_output
        else scales_u8.view(M, k_blocks)
    )
    return q_data, scales
