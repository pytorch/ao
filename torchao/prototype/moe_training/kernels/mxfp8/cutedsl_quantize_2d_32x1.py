# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Tuple

import torch

from torchao.utils import ceil_div

from .cute_utils import (
    F8_MAX,
    compute_amax,
    compute_scale_from_amax,
    load_vals_chunk_full,
    load_vals_chunk_tail,
)


def _make_tile_smem_layouts(cute, tile_m: int, tile_k: int):
    """Create shared memory layouts for input and output tiles.

    Input uses row-major format. Output uses column-major format.

    Args:
        cute: CuTe module
        tile_m: Tile size in M dimension
        tile_k: Tile size in K dimension

    Returns:
        Tuple of (smem_layout_in, smem_layout_out) for shared memory
    """
    # Input SMEM: Row-major layout
    smem_layout_in = cute.make_layout(
        (tile_m, tile_k),
        stride=(tile_k, 1),
    )
    # Output SMEM: Column-major layout
    smem_layout_out = cute.make_layout(
        (tile_m, tile_k),
        stride=(1, tile_m),
    )
    return smem_layout_in, smem_layout_out


# Config format:
# (compute_warps, tile_m, tile_k, m_tiles_per_cta)
# NOTE: Swapped tile_m/tile_k and changed k_tiles_per_cta to m_tiles_per_cta for 32x1 scaling
_CUTEDSL_CONFIGS = {
    "bf16_default": (4, 32, 128, 4),
    "fallback": (6, 32, 128, 2),
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
        Tuple of (config_name, (compute_warps, tile_m, tile_k, m_tiles_per_cta))
    """
    if input_dtype == torch.bfloat16:
        config_name = "bf16_default"
    else:
        config_name = "fallback"
    return config_name, _CUTEDSL_CONFIGS[config_name]


@functools.cache
def _compile_mxfp8_quantize_2d_cutedsl_32x1(
    input_dtype_name: str,
    scaling_mode: str,
    compute_warps: int,
    tile_m: int,
    tile_k: int,
    requested_stage_count: int,
    m_tiles_per_cta: int,
    is_full_m_tiles: bool,
    blocked_scale_output: bool,
):
    """Compile the 2D MXFP8 quantization kernel using CuTeDSL for 32x1 scaling.

    Uses warp-specialized TMA kernel with:
    - Warp 0: Producer (issues TMA global→shared and shared→global)
    - Warps 1..compute_warps: Consumers (quantize in registers)

    Args:
        input_dtype_name: Input dtype ("torch.float32" or "torch.bfloat16")
        scaling_mode: Scaling mode ("floor" or "rceil")
        compute_warps: Number of compute warps
        tile_m: Tile size in M dimension (32 for 32x1 scaling)
        tile_k: Tile size in K dimension (128 for 32x1 scaling)
        requested_stage_count: Requested pipeline stages (capped by m_tiles_per_cta)
        m_tiles_per_cta: Number of M tiles per CTA
        is_full_m_tiles: Whether M dimension is perfectly tiled
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
            f"Unsupported input dtype for CuTeDSL quantize_2d_32x1: {input_dtype_name}"
        )

    # Warp-specialized TMA kernel:
    # - warp 0: producer (issues TMA G2S and S2G)
    # - warps [1..compute_warps]: consumers (quantize)
    COMPUTE_WARPS = compute_warps
    TILE_M = tile_m
    TILE_K = tile_k
    M_TILES_PER_CTA = m_tiles_per_cta
    IS_FULL_M_TILES_VALUE = is_full_m_tiles
    BLOCKED_SCALE_OUTPUT_VALUE = blocked_scale_output

    THREADS_PER_BLOCK = (1 + COMPUTE_WARPS) * 32
    assert COMPUTE_WARPS >= 1
    assert TILE_M > 0 and TILE_K > 0
    assert TILE_M % 32 == 0  # Changed from TILE_K % 32 == 0 for 32x1 scaling

    SCALE_DIM_M_VALUE = 32  # Changed from SCALE_DIM_K_VALUE for 32x1 scaling
    M_BLOCKS_PER_TILE = TILE_M // SCALE_DIM_M_VALUE
    assert M_BLOCKS_PER_TILE > 0
    assert requested_stage_count >= 1
    assert requested_stage_count <= 2
    assert M_TILES_PER_CTA >= 1
    STAGE_COUNT_VALUE = min(requested_stage_count, M_TILES_PER_CTA)

    input_elem_bytes = 4 if input_dtype_name == "torch.float32" else 2
    TILE_COPY_BYTES = TILE_M * TILE_K * input_elem_bytes
    K_THREADS = COMPUTE_WARPS * 32  # Changed from M_THREADS for 32x1 scaling
    K_ITERS_PER_LANE = ceil_div(TILE_K, K_THREADS)

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

    class Mxfp8Quantize2dKernel32x1:
        @cute.jit
        def _load_block_full_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            k_rel: cutlass.Int32,
            m_base: cutlass.Int32,
        ):
            """Load a full 32-element quantization block from shared memory to registers.

            Loads all elements without bounds checking. For 32x1 scaling, we load 32 elements
            along the M dimension for a given K position.

            Args:
                sIN_tile: Input tile in shared memory (TILE_M, TILE_K)
                k_rel: Column index within tile
                m_base: Starting M index for this block within tile

            Returns:
                vals_block: 32 input elements in register memory
            """
            vals_block = cute.make_rmem_tensor((SCALE_DIM_M_VALUE,), cutlass.Float32)
            for i in range(SCALE_DIM_M_VALUE):
                vals_block[i] = cutlass.Float32(sIN_tile[m_base + i, k_rel])
            return vals_block

        @cute.jit
        def _load_block_tail_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            m0: cutlass.Int64,
            k_rel: cutlass.Int32,
            m_base: cutlass.Int32,
            M: cutlass.Int64,
        ):
            """Load a 32-element quantization block from shared memory to registers with bounds checking.

            Out-of-bounds elements are set to 0.0. For 32x1 scaling, we check M dimension bounds.

            Args:
                sIN_tile: Input tile in shared memory (TILE_M, TILE_K)
                m0: Global M offset for this tile
                k_rel: Column index within tile
                m_base: Starting M index for this block within tile
                M: Total M dimension size for bounds checking

            Returns:
                vals_block: 32 input elements in register memory (out-of-bounds set to 0.0)
            """
            vals_block = cute.make_rmem_tensor((SCALE_DIM_M_VALUE,), cutlass.Float32)
            for i in range(SCALE_DIM_M_VALUE):
                m = m0 + m_base + i
                if m < M:
                    vals_block[i] = cutlass.Float32(sIN_tile[m_base + i, k_rel])
                else:
                    vals_block[i] = cutlass.Float32(0.0)
            return vals_block

        @cute.jit
        def _store_scales_reg_to_gmem_vec(
            self,
            scales_tensor: cute.Tensor,
            k: cutlass.Int64,
            m_block_base: cutlass.Int64,
            scale_buffer: cute.Tensor,
            num_scales: cutlass.Int32,
            BLOCKED_SCALE_OUTPUT: cutlass.Constexpr[bool],
        ):
            """Store scales from registers to global memory using vectorized writes when possible.

            Uses uint32 vectorized writes for 4 scales in blocked layout.
            For 32x1 scaling, we store scales along the M dimension for each K.

            Args:
                scales_tensor: Output scales in global memory
                k: Global K coordinate
                m_block_base: Starting M block index
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
                    scales_tensor_u32[k, m_block_base // cutlass.Int64(4)] = (
                        scale_buffer_u32[0]
                    )
                else:
                    # Fallback for non-4 cases (e.g., tail tiles)
                    for i in range(num_scales):
                        m_block = m_block_base + i
                        scales_tensor[k, m_block] = scale_buffer[i]
            else:
                # Row-major layout - scalar stores
                for i in range(num_scales):
                    m_block = m_block_base + i
                    scales_tensor[k, m_block] = scale_buffer[i]

        @cute.jit
        def _store_q_fp8_reg_to_smem(
            self,
            q_fp8_vals4: cute.Tensor,
            sOUT_tile: cute.Tensor,
            m_base: cutlass.Int32,
            k_rel: cutlass.Int32,
            sout_stride: cutlass.Int32,
        ):
            """Store 4 FP8 values from registers to shared memory.

            For 32x1 scaling with column-major SMEM, we store 4 values along the M dimension.

            Args:
                q_fp8_vals4: 4 FP8 values in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K) with column-major layout
                m_base: Starting M index for this chunk within tile
                k_rel: Column index within tile
                sout_stride: Stride for storing values

            Storage locations:
                Input: q_fp8_vals4 (registers)
                Output: sOUT_tile (shared memory, column-major)
            """
            for i in range(4):
                sOUT_tile[m_base + i, k_rel] = q_fp8_vals4[i]

        @cute.jit
        def _quantize_then_store_reg_to_smem(
            self,
            vals_chunk: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m_base: cutlass.Int32,
            k_rel: cutlass.Int32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            """Quantize 4 input elements to FP8 and store to shared memory.

            Applies inverse scale, optional clamping (FLOOR mode), and converts to FP8.

            Args:
                vals_chunk: 4 input elements in register memory
                inv_scale: Inverse scale in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K)
                m_base: Starting M index for this chunk within tile
                k_rel: Column index within tile
                USE_RCEIL: Whether using RCEIL mode (no clamping) or FLOOR mode (clamp to ±448)

            Storage locations:
                Inputs: vals_chunk, inv_scale (registers)
                Output: sOUT_tile (shared memory)
            """
            q_vals4_vec = vals_chunk.load() * inv_scale
            if not cutlass.const_expr(USE_RCEIL):
                q_vals4_vec = cute.where(q_vals4_vec > F8_MAX, F8_MAX, q_vals4_vec)
                q_vals4_vec = cute.where(q_vals4_vec < -F8_MAX, -F8_MAX, q_vals4_vec)
            q_fp8_vec4 = q_vals4_vec.to(cutlass.Float8E4M3FN)
            q_fp8_vals4 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
            q_fp8_vals4.store(q_fp8_vec4)
            self._store_q_fp8_reg_to_smem(q_fp8_vals4, sOUT_tile, m_base, k_rel, 1)

        @cute.jit
        def _quantize_block_then_store_reg_to_smem_full(
            self,
            vals_block: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            k_rel: cutlass.Int32,
            m_base: cutlass.Int32,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            """Quantize and store a full 32-element block by processing 8 chunks of 4 elements.

            For 32x1 scaling, process 32 elements along M dimension.

            Args:
                vals_block: 32 input elements in register memory
                inv_scale: Inverse scale in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K)
                k_rel: Column index within tile
                m_base: Starting M index for this block within tile
                USE_RCEIL: Whether using RCEIL mode or FLOOR mode

            Storage locations:
                Inputs: vals_block, inv_scale (registers)
                Output: sOUT_tile (shared memory)
            """
            chunk_vec = 4
            num_chunks = SCALE_DIM_M_VALUE // chunk_vec
            for c in range(num_chunks):
                local_base = c * chunk_vec
                sout_m_base = m_base + local_base
                vals_chunk = load_vals_chunk_full(vals_block, local_base)
                self._quantize_then_store_reg_to_smem(
                    vals_chunk, inv_scale, sOUT_tile, sout_m_base, k_rel, USE_RCEIL
                )

        @cute.jit
        def _quantize_block_then_store_reg_to_smem_tail(
            self,
            vals_block: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m0: cutlass.Int64,
            k_rel: cutlass.Int32,
            m_base: cutlass.Int32,
            M: cutlass.Int64,
            USE_RCEIL: cutlass.Constexpr[bool],
        ):
            """Quantize and store a 32-element block with bounds checking by processing 8 chunks.

            Out-of-bounds elements are handled in the chunk loading stage.

            Args:
                vals_block: 32 input elements in register memory
                inv_scale: Inverse scale in register memory
                sOUT_tile: Output tile in shared memory (TILE_M, TILE_K)
                m0: Global M offset for this tile
                k_rel: Column index within tile
                m_base: Starting M index for this block within tile
                M: Total M dimension size for bounds checking
                USE_RCEIL: Whether using RCEIL mode or FLOOR mode

            Storage locations:
                Inputs: vals_block, inv_scale (registers)
                Output: sOUT_tile (shared memory)
            """
            chunk_vec = 4
            num_chunks = SCALE_DIM_M_VALUE // chunk_vec
            for c in range(num_chunks):
                local_base = c * chunk_vec
                sout_m_base = m_base + local_base
                vals_chunk = load_vals_chunk_tail(
                    vals_block, m0, sout_m_base, local_base, M
                )
                self._quantize_then_store_reg_to_smem(
                    vals_chunk, inv_scale, sOUT_tile, sout_m_base, k_rel, USE_RCEIL
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
                sIN_for_tma_partition = cute.group_modes(sIN_tile, 0, 1)
                gIN_for_tma_partition = cute.group_modes(gIN_tile, 0, 1)
                tINs, tINg = cpasync.tma_partition(
                    tma_atom_in,
                    0,
                    cta_layout,
                    sIN_for_tma_partition,
                    gIN_for_tma_partition,
                )
                tINg_stage0 = tINg[(None, 0)]
                tINs_stage0 = tINs[(None, 0)]
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
                sOUT_for_tma_partition = cute.group_modes(sOUT_tile, 0, 1)
                gOUT_for_tma_partition = cute.group_modes(gOUT_tile, 0, 1)
                tOUTs, tOUTg = cpasync.tma_partition(
                    tma_atom_out,
                    0,
                    cta_layout,
                    sOUT_for_tma_partition,
                    gOUT_for_tma_partition,
                )
                tOUTs_stage0 = tOUTs[(None, 0)]
                tOUTg_stage0 = tOUTg[(None, 0)]
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
            m_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            blocked_scale_layout: cute.Layout,
            SCALE_DIM_M: cutlass.Constexpr[int],
            USE_RCEIL: cutlass.Constexpr[bool],
            IS_FULL_M_TILES: cutlass.Constexpr[bool],
            STAGE_COUNT: cutlass.Constexpr[int],
        ):
            """Main MXFP8 quantization kernel with warp specialization and TMA pipeline for 32x1 scaling.

            Warp roles:
            - Warp 0: Producer (TMA loads/stores)
            - Warps 1..compute_warps: Consumers (quantize in registers)

            Pipeline stages:
            - Stage 0: Load tile to shared memory, quantize, store to global
            - Stage 1 (if enabled): Prefetch next tile while processing current

            For 32x1 scaling, we quantize along the M dimension - each column has M//32 scales.

            Args:
                inp_mk: Input tensor in global memory (M, K)
                tma_atom_in: TMA copy atom for G2S
                tma_tensor_in: TMA tensor view for input
                out_mk: Output tensor in global memory (M, K)
                tma_atom_out: TMA copy atom for S2G
                tma_tensor_out: TMA tensor view for output
                scales_out_u8: Output scales tensor in global memory (M//32, K) or blocked layout
                M: M dimension size
                K: K dimension size
                m_blocks: Number of 32-element blocks in M
                m_cta_tiles: Number of tiles in M dimension
                k_cta_tiles: Number of tile groups in K dimension
                blocked_scale_layout: Layout for blocked scale output
                SCALE_DIM_M: Block size (32)
                USE_RCEIL: Whether using RCEIL mode
                IS_FULL_M_TILES: Whether M is perfectly tiled
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

            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            tma_mbar_ptr0 = storage.tma_mbar_ptr.data_ptr()
            tma_mbar_ptr1 = tma_mbar_ptr0
            if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                tma_mbar_ptr1 = tma_mbar_ptr0 + 1

            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(
                cute, TILE_M, TILE_K
            )
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

            # For 32x1 scaling: bidx is M tile group, bidy is K tile
            m_tile_group_idx = cutlass.Int64(bidx)
            k_tile = cutlass.Int64(bidy)
            k0 = k_tile * TILE_K
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                scales_tensor = cute.make_tensor(
                    scales_out_u8.iterator,
                    blocked_scale_layout,
                )
            else:
                scales_tensor = scales_out_u8

            for tile_step in cutlass.range_constexpr(M_TILES_PER_CTA):
                m_tile_eff = m_tile_group_idx * M_TILES_PER_CTA + tile_step
                m0 = m_tile_eff * TILE_M

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
                    tile_step == 0 or not (STAGE_COUNT > 1 and M_TILES_PER_CTA > 1)
                ):
                    gIN_tile = cute.local_tile(
                        tma_tensor_in, (TILE_M, TILE_K), (m_tile_eff, k_tile)
                    )
                    self._issue_tma_load(
                        tma_atom_in,
                        gIN_tile,
                        sIN_tile,
                        tma_mbar_ptr,
                        warp_idx,
                    )

                if cutlass.const_expr(STAGE_COUNT > 1 and M_TILES_PER_CTA > 1):
                    if cutlass.const_expr(tile_step + 1 < M_TILES_PER_CTA):
                        m_tile_next = m_tile_group_idx * M_TILES_PER_CTA + tile_step + 1
                        next_stage_idx = (tile_step + 1) % STAGE_COUNT
                        sIN_tile_next = sIN_tile0
                        tma_mbar_ptr_next = tma_mbar_ptr0
                        if cutlass.const_expr(STAGE_COUNT > 1):
                            tma_mbar_ptr_next = tma_mbar_ptr0 + next_stage_idx
                        if cutlass.const_expr(STAGE_COUNT > 1):
                            if next_stage_idx == 1:
                                sIN_tile_next = sIN_tile1

                        gIN_tile_next = cute.local_tile(
                            tma_tensor_in, (TILE_M, TILE_K), (m_tile_next, k_tile)
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
                    k_lane = (warp_idx - 1) * 32 + lane

                    for kk in cutlass.range_constexpr(K_ITERS_PER_LANE):
                        k_rel = k_lane + kk * K_THREADS
                        k = k0 + k_rel
                        if cutlass.const_expr(IS_FULL_M_TILES):
                            if k_rel < TILE_K:
                                # Buffer scales for vectorized store
                                scale_buffer = cute.make_rmem_tensor(
                                    (M_BLOCKS_PER_TILE,), cutlass.Uint8
                                )

                                for mb in cutlass.range_constexpr(M_BLOCKS_PER_TILE):
                                    m_base = mb * SCALE_DIM_M_VALUE
                                    vals_block = self._load_block_full_smem_to_reg(
                                        sIN_tile,
                                        k_rel,
                                        m_base,
                                    )

                                    amax = compute_amax(vals_block)

                                    scale_biased, inv_scale = compute_scale_from_amax(
                                        amax, USE_RCEIL
                                    )
                                    scale_buffer[mb] = cutlass.Uint8(scale_biased)

                                    self._quantize_block_then_store_reg_to_smem_full(
                                        vals_block,
                                        inv_scale,
                                        sOUT_tile,
                                        k_rel,
                                        m_base,
                                        USE_RCEIL,
                                    )

                                # Vectorized scale store
                                m_block_base = m_tile_eff * M_BLOCKS_PER_TILE
                                self._store_scales_reg_to_gmem_vec(
                                    scales_tensor,
                                    k,
                                    m_block_base,
                                    scale_buffer,
                                    cutlass.Int32(M_BLOCKS_PER_TILE),
                                    BLOCKED_SCALE_OUTPUT_VALUE,
                                )
                        else:
                            k_in_bounds = k < K
                            if k_rel < TILE_K and k_in_bounds:
                                # Buffer scales for vectorized store
                                scale_buffer = cute.make_rmem_tensor(
                                    (M_BLOCKS_PER_TILE,), cutlass.Uint8
                                )
                                num_valid_scales = cutlass.Int32(0)

                                for mb in cutlass.range_constexpr(M_BLOCKS_PER_TILE):
                                    m_block = m_tile_eff * M_BLOCKS_PER_TILE + mb
                                    if m_block < m_blocks:
                                        m_base = mb * SCALE_DIM_M_VALUE
                                        vals_block = self._load_block_tail_smem_to_reg(
                                            sIN_tile,
                                            m0,
                                            k_rel,
                                            m_base,
                                            M,
                                        )

                                        amax = compute_amax(vals_block)

                                        scale_biased, inv_scale = (
                                            compute_scale_from_amax(amax, USE_RCEIL)
                                        )
                                        scale_buffer[num_valid_scales] = cutlass.Uint8(
                                            scale_biased
                                        )
                                        num_valid_scales = num_valid_scales + 1

                                        self._quantize_block_then_store_reg_to_smem_tail(
                                            vals_block,
                                            inv_scale,
                                            sOUT_tile,
                                            m0,
                                            k_rel,
                                            m_base,
                                            M,
                                            USE_RCEIL,
                                        )

                                # Vectorized scale store
                                if num_valid_scales > 0:
                                    m_block_base = m_tile_eff * M_BLOCKS_PER_TILE
                                    self._store_scales_reg_to_gmem_vec(
                                        scales_tensor,
                                        k,
                                        m_block_base,
                                        scale_buffer,
                                        num_valid_scales,
                                        BLOCKED_SCALE_OUTPUT_VALUE,
                                    )

                gOUT_tile = cute.local_tile(
                    tma_tensor_out, (TILE_M, TILE_K), (m_tile_eff, k_tile)
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
            m_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            """Kernel launcher that sets up TMA descriptors and blocked scale layout.

            Args:
                inp_mk: Input tensor in global memory (M, K)
                out_mk: Output quantized data tensor in global memory (M, K)
                scales_out_u8: Output scales tensor in global memory (M//32, K) or blocked layout
                M: M dimension size
                K: K dimension size
                m_blocks: Number of 32-element blocks in M
                m_cta_tiles: Number of tiles in M dimension
                k_cta_tiles: Number of tile groups in K dimension
                stream: CUDA stream

            Storage locations:
                All tensors in global memory
            """
            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(
                cute, TILE_M, TILE_K
            )
            # Use tcgen05.CtaGroup.ONE for the optimised single-CTA Blackwell (SM 10.x) TMA load path.
            g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
                g2s_op,
                inp_mk,
                smem_layout_in,
                (TILE_M, TILE_K),
            )
            out_colmajor = cute.make_tensor(
                out_mk.iterator,
                cute.make_layout((M, K), stride=(1, M)),  # col major
            )
            tma_atom_out, tma_tensor_out = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_colmajor,
                smem_layout_out,
                (
                    TILE_M,
                    TILE_K,
                ),  # Keep original tile dimensions to match global memory expectation
            )

            blocked_scale_layout = cute.make_layout((1,))
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT_VALUE):
                # Use same blocked layout as regular 2D kernel but for (K, M//32) tensor
                # For scales shape (K, M//32): K is rows, M//32 is columns
                padded_scale_rows = cute.round_up(
                    K, 128
                )  # K rounded to multiple of 128 (first dim)
                padded_scale_cols = cute.round_up(
                    m_blocks, 4
                )  # M//32 rounded to multiple of 4 (second dim)
                m_block_tiles = padded_scale_rows // cutlass.Int64(
                    128
                )  # Number of 128-row tiles
                k_block_tiles = padded_scale_cols // cutlass.Int64(
                    4
                )  # Number of 4-col tiles
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
                m_blocks,
                m_cta_tiles,
                k_cta_tiles,
                blocked_scale_layout,
                SCALE_DIM_M=SCALE_DIM_M_VALUE,
                USE_RCEIL=(scaling_mode == "rceil"),
                IS_FULL_M_TILES=IS_FULL_M_TILES_VALUE,
                STAGE_COUNT=STAGE_COUNT_VALUE,
            ).launch(
                grid=(m_cta_tiles, k_cta_tiles, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    kernel = Mxfp8Quantize2dKernel32x1()

    m = cute.sym_int(divisibility=32)
    k = cute.sym_int(divisibility=128)
    mb = cute.sym_int()
    inp_stride0 = cute.sym_int()
    inp_stride1 = cute.sym_int()
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
        stride=(1, m),  # col major
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
            (k, mb),  # Changed to (K, M//32) to match torch._scaled_mm format
            stride=(scale_stride0, scale_stride1),
        )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        inp_mk=fake_inp,
        out_mk=fake_out,
        scales_out_u8=fake_scales,
        M=0,
        K=0,
        m_blocks=0,
        m_cta_tiles=1,
        k_cta_tiles=1,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def mxfp8_quantize_cutedsl_2d_32x1(
    x: torch.Tensor,
    block_size: int = 32,
    scaling_mode: str = "rceil",
    stage_count: int = 2,
    blocked_scale_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 2D tensor to MXFP8 format using CuTe DSL kernel with 32x1 scaling.

    Quantizes along the M dimension - each column has M//32 scales, one per block of 32 M elements.

    Args:
        x: Input tensor of shape (M, K)
        block_size: Block size for quantization along M (only 32 supported)
        scaling_mode: Scaling mode ("floor" or "rceil")
        stage_count: Number of pipeline stages (1 or 2)
        blocked_scale_output: Whether to output scales in blocked layout

    Returns:
        q_data: Quantized data in row-major layout with shape (M, K) (no padding on data)
        scales: Scales tensor with shape (K, M//32) or blocked layout compatible with torch._scaled_mm
                Same format as other dim1 MXFP8 kernels for consistency
    """
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert x.is_cuda, "Input tensor must be CUDA"
    assert block_size == 32, "Only block_size=32 is supported"
    M, K = x.shape
    assert M % block_size == 0, "M must be divisible by block_size for 32x1 scaling"

    _, config = _select_cutedsl_config(x.dtype, scaling_mode)
    compute_warps, tile_m, tile_k, m_tiles_per_cta = config
    assert stage_count >= 1, "stage_count must be >= 1"
    assert stage_count <= 2, "stage_count must be <= 2"
    is_full_m_tiles = M % (tile_m * m_tiles_per_cta) == 0
    is_sm_10x = torch.cuda.get_device_capability()[0] == 10
    if blocked_scale_output and not is_sm_10x:
        raise NotImplementedError(
            "blocked_scale_output is only supported on SM 10.x GPUs "
            "because it produces the tcgen05 blocked scale layout"
        )

    # For 32x1 scaling, only the scales need padding for 4x128 tiles
    # The output data tensor has the same dimensions as input (no padding)

    # Output in column-major layout: stride (1, M) - M fastest-changing
    q_data = torch.empty_strided(
        (M, K),
        (1, M),
        device=x.device,
        dtype=torch.float8_e4m3fn,
    )
    m_blocks = M // block_size
    if blocked_scale_output:
        # Use same blocked layout as regular 2D kernel for (K, M//32) tensor
        padded_scale_rows = (
            ceil_div(K, 128) * 128
        )  # K rounded to multiple of 128 (first dim)
        padded_scale_cols = (
            ceil_div(m_blocks, 4) * 4
        )  # M//32 rounded to multiple of 4 (second dim)
        scales_u8 = (
            torch.zeros(  # Initialize with zeros to match to_blocked() padding behavior
                (padded_scale_rows * padded_scale_cols,),
                device=x.device,
                dtype=torch.uint8,
            )
        )
    else:
        scales_u8 = torch.empty(
            (K, m_blocks),
            device=x.device,
            dtype=torch.uint8,
        )

    compiled = _compile_mxfp8_quantize_2d_cutedsl_32x1(
        str(x.dtype),
        scaling_mode,
        compute_warps,
        tile_m,
        tile_k,
        stage_count,
        m_tiles_per_cta,
        is_full_m_tiles,
        blocked_scale_output,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    m_cta_tiles = ceil_div(M, tile_m * m_tiles_per_cta)
    k_cta_tiles = ceil_div(K, tile_k)

    compiled(
        x,
        q_data,
        scales_u8,
        int(M),
        int(K),
        int(m_blocks),
        int(m_cta_tiles),
        int(k_cta_tiles),
        stream,
    )

    return q_data, scales_u8.view(torch.float8_e8m0fnu)
