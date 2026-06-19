# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fused NVFP4 (E2M1, block 16, E4M3 two-level scale) +/- RHT CuTeDSL cast.

Clones the structure of the fused MXFP4 + RHT CuTeDSL kernel
(``mxfp4_rht_quantize.py``) and applies the NVFP4 deltas:

* block size is 16 (not 32): each quantization block has 16 elements, packed to
  8 ``E2M1x2`` bytes (8 ``cvt.rn.satfinite.e2m1x2.f32`` calls per block, written
  as 2 ``uint32`` stores). The output qdata is still half-width ``[M, K // 2]``
  ``uint8`` row-major; a K-tile of 128 holds 8 sixteen-element blocks;
* the block scale is the NVFP4 two-level E4M3 scale: per block,
  ``compute_amax`` -> ``compute_nvfp4_scale_e4m3(amax, global_scale)`` ->
  ``(e4m3_byte, inv_scale)``. The E4M3 byte is stored as ``float8_e4m3fn`` and
  the data is quantized with ``inv_scale`` (``code = e2m1(val * inv_scale)``);
* the Random Hadamard Transform is optional. When a length-16 ``sign_vector`` is
  provided the consumer applies the register-resident FWHT(16) + sign transform
  (``fwht.fwht16_sign``) before amax / scale / packing; when it is empty (``[]``)
  the transform is skipped via a compile-time ``apply_rht`` constexpr so the
  no-RHT path carries no FWHT overhead (plain NVFP4 cast);
* values are clamped to ``+-6.0`` (``F4_E2M1_MAX``) before the E2M1 cvt;
* the E4M3 scale byte is written either in the cuBLAS-blocked padded layout
  (``is_swizzled_scales=True``) or as a plain ``(M, K // 16)`` tensor.

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
    compute_nvfp4_scale_e4m3,
)
from .fwht import fwht16_sign

# Config format:
# (compute_warps, tile_m, tile_k, k_tiles_per_cta)
_CUTEDSL_CONFIGS = {
    "bf16_default": (4, 128, 128, 4),
    "fallback": (6, 128, 128, 2),
}


def _select_cutedsl_config(
    input_dtype: torch.dtype,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """Select kernel configuration based on input dtype."""
    if input_dtype == torch.bfloat16:
        config_name = "bf16_default"
    else:
        config_name = "fallback"
    return config_name, _CUTEDSL_CONFIGS[config_name]


def _make_tile_smem_layouts(tile_m: int, tile_k: int):
    """Row-major smem layouts for the input ``(TILE_M, TILE_K)`` and the
    half-width output ``(TILE_M, TILE_K // 2)`` tiles."""
    import cutlass.cute as cute

    smem_layout_in = cute.make_layout(
        (tile_m, tile_k),
        stride=(tile_k, 1),
    )
    smem_layout_out = cute.make_layout(
        (tile_m, tile_k // 2),
        stride=(tile_k // 2, 1),
    )
    return smem_layout_in, smem_layout_out


@functools.cache
def _compile_nvfp4_rht_quantize_2d_cutedsl(
    input_dtype_name: str,
    apply_rht: bool,
    compute_warps: int,
    tile_m: int,
    tile_k: int,
    requested_stage_count: int,
    k_tiles_per_cta: int,
    blocked_scale_output: bool,
):
    """Compile the fused 2D NVFP4 (+/- RHT) quantize kernel using CuTeDSL.

    Warp-specialized TMA kernel mirroring the MXFP4 1x32 template:
    - warp 0: producer (TMA global->shared input, shared->global half-width
      output);
    - warps [1..compute_warps]: consumers (optional FWHT(16) + quantize + E2M1
      pack).
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
            f"Unsupported input dtype for CuTeDSL nvfp4 quantize_2d: {input_dtype_name}"
        )

    COMPUTE_WARPS = compute_warps
    TILE_M = tile_m
    TILE_K = tile_k
    TILE_K_HALF = tile_k // 2
    K_TILES_PER_CTA = k_tiles_per_cta
    BLOCKED_SCALE_OUTPUT_VALUE = blocked_scale_output
    APPLY_RHT_VALUE = apply_rht

    THREADS_PER_BLOCK = (1 + COMPUTE_WARPS) * 32
    assert COMPUTE_WARPS >= 1
    assert TILE_M > 0 and TILE_K > 0
    assert TILE_K % 16 == 0

    SCALE_DIM_K_VALUE = 16
    SCALE_DIM_K_HALF = SCALE_DIM_K_VALUE // 2  # 8 packed bytes per block
    K_BLOCKS_PER_TILE = TILE_K // SCALE_DIM_K_VALUE
    assert K_BLOCKS_PER_TILE > 0
    assert requested_stage_count >= 1
    assert requested_stage_count <= 2
    assert K_TILES_PER_CTA >= 1
    STAGE_COUNT_VALUE = min(requested_stage_count, K_TILES_PER_CTA)

    input_elem_bytes = 4 if input_dtype_name == "torch.float32" else 2
    TILE_COPY_BYTES_IN = TILE_M * TILE_K * input_elem_bytes
    # Half-width uint8 output: 1 byte per packed E2M1 pair.
    TILE_COPY_BYTES_OUT = TILE_M * TILE_K_HALF  # noqa: F841 (documented contract)
    M_THREADS = COMPUTE_WARPS * 32
    M_ITERS_PER_LANE = ceil_div(TILE_M, M_THREADS)

    # F4_E2M1_MAX == 6.0; clamp pre-cvt values before the E2M1 conversion.
    F4_MAX = cutlass.Float32(6.0)

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
                cutlass.Uint8, STAGE_COUNT_VALUE * TILE_M * TILE_K_HALF
            ],
            128,
        ]

    class Nvfp4RhtQuantize2dKernel:
        @cute.jit
        def _load_block_full_smem_to_reg(
            self,
            sIN_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
        ):
            """Load a full 16-element quantization block from smem to registers."""
            vals_block = cute.make_rmem_tensor((SCALE_DIM_K_VALUE,), cutlass.Float32)
            for i in cutlass.range_constexpr(SCALE_DIM_K_VALUE):
                vals_block[i] = cutlass.Float32(sIN_tile[m_rel, k_base + i])
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
            """Store scales from registers to gmem (uint32 vectorized for blocked)."""
            if cutlass.const_expr(BLOCKED_SCALE_OUTPUT):
                if num_scales == 4:
                    scales_tensor_u32 = cute.recast_tensor(
                        scales_tensor, cutlass.Uint32
                    )
                    scale_buffer_u32 = cute.recast_tensor(scale_buffer, cutlass.Uint32)
                    scales_tensor_u32[m, k_block_base // cutlass.Int64(4)] = (
                        scale_buffer_u32[0]
                    )
                else:
                    for i in range(num_scales):
                        k_block = k_block_base + i
                        scales_tensor[m, k_block] = scale_buffer[i]
            else:
                for i in range(num_scales):
                    k_block = k_block_base + i
                    scales_tensor[m, k_block] = scale_buffer[i]

        @cute.jit
        def _store_q_e2m1_block_to_smem(
            self,
            packed_bytes: cute.Tensor,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            sout_base: cutlass.Int32,
        ):
            """Store the 8 packed E2M1x2 bytes of a block via 2 uint32 writes.

            ``packed_bytes`` is a length-8 ``Uint8`` register fragment holding
            the bytes for one 16-element block. ``sout_base`` is the byte offset
            of the block within the half-width output tile (``k_base // 2``).
            """
            sOUT_tile_u32 = cute.recast_tensor(sOUT_tile, cutlass.Uint32)
            packed_u32 = cute.recast_tensor(packed_bytes, cutlass.Uint32)
            base_u32 = sout_base // cutlass.Int32(4)
            for w in cutlass.range_constexpr(SCALE_DIM_K_HALF // 4):
                sOUT_tile_u32[m_rel, base_u32 + w] = packed_u32[w]

        @cute.jit
        def _quantize_block_then_store_reg_to_smem_full(
            self,
            rht_block: cute.Tensor,
            inv_scale: cutlass.Float32,
            sOUT_tile: cute.Tensor,
            m_rel: cutlass.Int32,
            k_base: cutlass.Int32,
        ):
            """Scale, clamp, pack 16 values to 8 E2M1x2 bytes.

            ``rht_block`` is the (optionally post-FWHT) length-16 ``Float32``
            fragment. Even column ``2p`` -> low nibble, odd column ``2p + 1`` ->
            high nibble of output byte ``p`` (validated bit-exactly in Task 1).
            """
            packed_bytes = cute.make_rmem_tensor((SCALE_DIM_K_HALF,), cutlass.Uint8)
            for p in cutlass.range_constexpr(SCALE_DIM_K_HALF):
                lo = cutlass.Float32(rht_block[2 * p]) * inv_scale
                hi = cutlass.Float32(rht_block[2 * p + 1]) * inv_scale
                # Clamp to +-F4_E2M1_MAX before the cvt.
                if lo > F4_MAX:
                    lo = F4_MAX
                if lo < -F4_MAX:
                    lo = -F4_MAX
                if hi > F4_MAX:
                    hi = F4_MAX
                if hi < -F4_MAX:
                    hi = -F4_MAX
                packed_bytes[p] = _cvt_rn_satfinite_e2m1x2_f32(hi, lo)
            sout_base = k_base // cutlass.Int32(2)
            self._store_q_e2m1_block_to_smem(packed_bytes, sOUT_tile, m_rel, sout_base)

        @cute.jit
        def _issue_tma_load(
            self,
            tma_atom_in: cute.CopyAtom,
            gIN_tile: cute.Tensor,
            sIN_tile: cute.Tensor,
            tma_mbar_ptr: cutlass.Int64,
            warp_idx: cutlass.Int32,
        ):
            """Issue TMA load from global to shared memory (producer warp only)."""
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
                        tma_mbar_ptr, TILE_COPY_BYTES_IN
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
            """Issue TMA store from shared to global memory (producer warp only)."""
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
            sign_vec: cute.Tensor,
            global_scale: cutlass.Float32,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            blocked_scale_layout: cute.Layout,
            SCALE_DIM_K: cutlass.Constexpr[int],
            APPLY_RHT: cutlass.Constexpr[bool],
            STAGE_COUNT: cutlass.Constexpr[int],
        ):
            """Main fused NVFP4 (+/- RHT) quantize kernel (warp-specialized TMA)."""
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

            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(TILE_M, TILE_K)
            staged_layout_in = cute.make_layout(
                (STAGE_COUNT_VALUE, TILE_M, TILE_K),
                stride=(TILE_M * TILE_K, TILE_K, 1),
            )
            staged_layout_out = cute.make_layout(
                (STAGE_COUNT_VALUE, TILE_M, TILE_K_HALF),
                stride=(TILE_M * TILE_K_HALF, TILE_K_HALF, 1),
            )
            sIN_staged = storage.in_smem.get_tensor(staged_layout_in)
            sOUT_staged = storage.out_smem.get_tensor(staged_layout_out)
            stage_elems_in = TILE_M * TILE_K
            stage_elems_out = TILE_M * TILE_K_HALF
            sIN_tile0 = cute.make_tensor(
                sIN_staged.iterator + 0 * stage_elems_in, smem_layout_in
            )
            sOUT_tile0 = cute.make_tensor(
                sOUT_staged.iterator + 0 * stage_elems_out, smem_layout_out
            )
            sIN_tile1 = sIN_tile0
            sOUT_tile1 = sOUT_tile0
            if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                sIN_tile1 = cute.make_tensor(
                    sIN_staged.iterator + 1 * stage_elems_in, smem_layout_in
                )
                sOUT_tile1 = cute.make_tensor(
                    sOUT_staged.iterator + 1 * stage_elems_out, smem_layout_out
                )

            if tidx == 0:
                cpasync.prefetch_descriptor(tma_atom_in)
                cpasync.prefetch_descriptor(tma_atom_out)
                cute.arch.mbarrier_init(tma_mbar_ptr0, 1)
                if cutlass.const_expr(STAGE_COUNT_VALUE > 1):
                    cute.arch.mbarrier_init(tma_mbar_ptr1, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # Load the length-16 sign vector into registers (broadcast over rows).
            # Only needed for the RHT path; the no-RHT path skips the FWHT entirely.
            sign_reg = cute.make_rmem_tensor((SCALE_DIM_K_VALUE,), cutlass.Float32)
            if cutlass.const_expr(APPLY_RHT):
                for j in cutlass.range_constexpr(SCALE_DIM_K_VALUE):
                    sign_reg[j] = cutlass.Float32(sign_vec[j])

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
                        if m_rel < TILE_M:
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

                                # Optional fused RHT: in-register FWHT(16) + sign.
                                if cutlass.const_expr(APPLY_RHT):
                                    fwht16_sign(vals_block, sign_reg)

                                amax = compute_amax(vals_block)
                                e4m3_byte, inv_scale = compute_nvfp4_scale_e4m3(
                                    amax, global_scale
                                )
                                scale_buffer[kb] = e4m3_byte

                                self._quantize_block_then_store_reg_to_smem_full(
                                    vals_block,
                                    inv_scale,
                                    sOUT_tile,
                                    m_rel,
                                    k_base,
                                )

                            k_block_base = k_tile_eff * K_BLOCKS_PER_TILE
                            self._store_scales_reg_to_gmem_vec(
                                scales_tensor,
                                m,
                                k_block_base,
                                scale_buffer,
                                cutlass.Int32(K_BLOCKS_PER_TILE),
                                BLOCKED_SCALE_OUTPUT_VALUE,
                            )

                gOUT_tile = cute.local_tile(
                    tma_tensor_out, (TILE_M, TILE_K_HALF), (m_tile, k_tile_eff)
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
            sign_vec: cute.Tensor,
            global_scale: cutlass.Float32,
            M: cutlass.Int64,
            K: cutlass.Int64,
            k_blocks: cutlass.Int64,
            m_cta_tiles: cutlass.Int64,
            k_cta_tiles: cutlass.Int64,
            stream: cuda.CUstream,
        ):
            """Kernel launcher: set up TMA descriptors and blocked scale layout."""
            smem_layout_in, smem_layout_out = _make_tile_smem_layouts(TILE_M, TILE_K)
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
                (TILE_M, TILE_K_HALF),
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
                sign_vec,
                global_scale,
                M,
                K,
                k_blocks,
                m_cta_tiles,
                k_cta_tiles,
                blocked_scale_layout,
                SCALE_DIM_K=SCALE_DIM_K_VALUE,
                APPLY_RHT=APPLY_RHT_VALUE,
                STAGE_COUNT=STAGE_COUNT_VALUE,
            ).launch(
                grid=(k_cta_tiles, m_cta_tiles, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                stream=stream,
            )

    kernel = Nvfp4RhtQuantize2dKernel()

    m = cute.sym_int(divisibility=128)
    k = cute.sym_int(divisibility=128)
    k_half = cute.sym_int(divisibility=64)
    kb = cute.sym_int()
    inp_stride0 = cute.sym_int()
    inp_stride1 = cute.sym_int()
    out_stride0 = cute.sym_int()
    out_stride1 = cute.sym_int()
    scale_stride0 = cute.sym_int()
    scale_stride1 = cute.sym_int()
    sign_stride0 = cute.sym_int()

    fake_inp = make_fake_tensor(
        INPUT_CUTLASS_DTYPE,
        (m, k),
        stride=(inp_stride0, inp_stride1),
    )
    fake_out = make_fake_tensor(
        cutlass.Uint8,
        (m, k_half),
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
    fake_sign = make_fake_tensor(
        cutlass.Int32,
        (16,),
        stride=(sign_stride0,),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        inp_mk=fake_inp,
        out_mk=fake_out,
        scales_out_u8=fake_scales,
        sign_vec=fake_sign,
        global_scale=cutlass.Float32(1.0),
        M=0,
        K=0,
        k_blocks=0,
        m_cta_tiles=1,
        k_cta_tiles=1,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def _nvfp4_rht_quantize_cutedsl_impl(
    x: torch.Tensor,
    global_scale: float,
    sign_vector: list[int],
    block_size: int = 16,
    is_swizzled_scales: bool = True,
    stage_count: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host wrapper: launch the fused NVFP4 (+/- RHT) CuTeDSL quantize kernel.

    Args:
        x: 2D contiguous bf16/fp32 input ``(M, K)`` with ``M % 128 == 0`` and
            ``K % 128 == 0``.
        global_scale: per-tensor multiplicative global scale (the reciprocal of
            torchao's ``per_tensor_scale``).
        sign_vector: length-16 list of ``{-1, +1}`` for the RHT sign multiply, or
            an empty list ``[]`` to skip the transform (plain NVFP4 cast).
        block_size: only 16 is supported.
        is_swizzled_scales: write scales in the cuBLAS-blocked padded layout
            (``True``) or a plain ``(M, K // 16)`` tensor (``False``).
        stage_count: pipeline stages (1 or 2).

    Returns:
        ``(qdata, scales)`` where ``qdata`` is row-major ``(M, K // 2)`` uint8
        (packed E2M1x2) and ``scales`` is ``float8_e4m3fn`` in the requested
        layout.
    """
    import cuda.bindings.driver as cuda

    assert x.is_cuda, "Input tensor must be CUDA"
    assert x.dim() == 2, "Input tensor must be 2D"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dtype in (
        torch.float32,
        torch.bfloat16,
    ), "Input tensor must be float32 or bfloat16"
    assert block_size == 16, "Only block_size=16 is supported"
    apply_rht = len(sign_vector) > 0
    if apply_rht:
        assert len(sign_vector) == 16, "sign_vector must have length 16 (or be empty)"

    M, K = x.shape
    assert K % 16 == 0, "K must be divisible by 16"
    assert M % 128 == 0, "M must be divisible by 128 (TMA tiling)"
    assert K % 128 == 0, "K must be divisible by 128 (TMA tiling)"

    _, config = _select_cutedsl_config(x.dtype)
    compute_warps, tile_m, tile_k, k_tiles_per_cta = config
    assert stage_count >= 1, "stage_count must be >= 1"
    assert stage_count <= 2, "stage_count must be <= 2"

    k_blocks = K // block_size

    # Half-width row-major packed output: stride (K // 2, 1).
    q_data = torch.empty_strided(
        (M, K // 2),
        (K // 2, 1),
        device=x.device,
        dtype=torch.uint8,
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
        scales_u8 = torch.empty(
            (M, k_blocks),
            device=x.device,
            dtype=torch.uint8,
        )

    # The kernel always reads a length-16 sign tensor; for the no-RHT path it is
    # never dereferenced (guarded by the ``apply_rht`` constexpr), so a zero
    # placeholder is fine.
    sign_src = sign_vector if apply_rht else [0] * 16
    sign_dev = torch.tensor(
        [int(s) for s in sign_src], device=x.device, dtype=torch.int32
    )

    compiled = _compile_nvfp4_rht_quantize_2d_cutedsl(
        str(x.dtype),
        apply_rht,
        compute_warps,
        tile_m,
        tile_k,
        stage_count,
        k_tiles_per_cta,
        is_swizzled_scales,
    )

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    m_cta_tiles = ceil_div(M, tile_m)
    k_cta_tiles = ceil_div(K, tile_k * k_tiles_per_cta)

    compiled(
        x,
        q_data,
        scales_u8,
        sign_dev,
        float(global_scale),
        int(M),
        int(K),
        int(k_blocks),
        int(m_cta_tiles),
        int(k_cta_tiles),
        stream,
    )

    scales = scales_u8.view(torch.float8_e4m3fn)
    scales = (
        scales.view(padded_scale_rows, padded_scale_cols)
        if is_swizzled_scales
        else scales.view(M, k_blocks)
    )
    return q_data, scales


@torch.library.custom_op("torchao::nvfp4_rht_quantize_cutedsl", mutates_args=())
def nvfp4_rht_quantize_cutedsl(
    x: torch.Tensor,
    global_scale: float,
    sign_vector: list[int],
    block_size: int = 16,
    is_swizzled_scales: bool = True,
    stage_count: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _nvfp4_rht_quantize_cutedsl_impl(
        x,
        global_scale,
        sign_vector,
        block_size=block_size,
        is_swizzled_scales=is_swizzled_scales,
        stage_count=stage_count,
    )


@nvfp4_rht_quantize_cutedsl.register_fake
def _(
    x: torch.Tensor,
    global_scale: float,
    sign_vector: list[int],
    block_size: int = 16,
    is_swizzled_scales: bool = True,
    stage_count: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    q = torch.empty_strided(
        (m, k // 2), (k // 2, 1), device=x.device, dtype=torch.uint8
    )  # row-major pinned
    kb = k // block_size
    if is_swizzled_scales:
        scales = x.new_empty(
            (ceil_div(m, 128) * 128, ceil_div(kb, 4) * 4),
            dtype=torch.float8_e4m3fn,
        )
    else:
        scales = x.new_empty((m, kb), dtype=torch.float8_e4m3fn)
    return q, scales


def nvfp4_rht_quantize_cutedsl_2d(
    x: torch.Tensor,
    global_scale,
    sign_vector=None,
    block_size: int = 16,
    is_swizzled_scales: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gated public wrapper for the fused NVFP4 (+/- RHT) CuTeDSL quantize op.

    ``sign_vector=None`` (or an empty list) selects the plain NVFP4 cast; a
    length-16 ``sign_vector`` enables the fused RHT. Raises
    ``NotImplementedError`` (with the missing-runtime detail) when the CuTeDSL
    runtime / SM 10.x / CUDA >= 12.8 requirements are not met.
    """
    from torchao.prototype.mx_formats.cutedsl import (
        _mxfp4_rht_cutedsl_kernels_available,
    )

    if not _mxfp4_rht_cutedsl_kernels_available:
        from torchao.prototype.mx_formats.cutedsl.cute_utils import (
            _missing_cutedsl_runtime_packages,
        )

        raise NotImplementedError(
            "nvfp4_rht_quantize_cutedsl requires CUDA SM10.x, CUDA>=12.8, and: "
            f"{_missing_cutedsl_runtime_packages() or 'nvidia-cutlass-dsl'}"
        )
    sv = list(sign_vector) if sign_vector is not None else []
    return nvfp4_rht_quantize_cutedsl(
        x, float(global_scale), sv, block_size, is_swizzled_scales
    )
