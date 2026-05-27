# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL dual-output MXFP8 quantization for MoE backward grad_out."""

import functools
from typing import Tuple

import torch
from torch import Tensor

from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _cutedsl_runtime_available,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.utils import ceil_div

_TILE_M = 128
_TILE_N = 128
_BLOCK_SIZE = 32
_COMPUTE_WARPS_PER_DIRECTION = 4

_STREAMS: dict[int, Tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}


def _get_streams(device: torch.device) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    device_idx = device.index
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    if device_idx not in _STREAMS:
        _STREAMS[device_idx] = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
        )
    return _STREAMS[device_idx]


def cutedsl_mxfp8_quantize_dim0_dim1_streams(
    x: Tensor,
    scaling_mode: str = "rceil",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize ``x`` for both dgrad and wgrad using existing CuTeDSL kernels.

    Returns the same logical outputs as ``triton_mxfp8_quantize_dim0_dim1``:
    row-major ``(M, N)`` qdata/scales for dgrad and row-major ``(N, M)``
    qdata/scales for wgrad.  This uses two CUDA streams to overlap the existing
    single-direction CuTeDSL kernels.
    """
    assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.ndim == 2, f"x must be 2D, got {x.ndim}D"
    assert scaling_mode in ("rceil", "floor"), (
        f"scaling_mode must be 'rceil' or 'floor', got {scaling_mode}"
    )
    M, N = x.shape
    assert M % 128 == 0, f"M must be a multiple of 128, got M={M}"
    assert N % 128 == 0, f"N must be a multiple of 128, got N={N}"

    cur_stream = torch.cuda.current_stream(device=x.device)
    stream_dim0, stream_dim1 = _get_streams(x.device)
    stream_dim0.wait_stream(cur_stream)
    stream_dim1.wait_stream(cur_stream)

    out_dim0 = None
    out_dim1 = None
    with torch.cuda.stream(stream_dim0):
        out_dim0 = mxfp8_quantize_2d_1x32_cutedsl(
            x,
            scaling_mode=scaling_mode,
            stage_count=2,
        )
    with torch.cuda.stream(stream_dim1):
        qdata_dim1_col_major, scales_dim1 = mxfp8_quantize_2d_32x1_cutedsl(
            x,
            scaling_mode=scaling_mode,
            stage_count=2,
            blocked_scale_output=True,
        )
        out_dim1 = (qdata_dim1_col_major.t(), scales_dim1)

    cur_stream.wait_stream(stream_dim0)
    cur_stream.wait_stream(stream_dim1)
    qdata_dim0, scales_dim0 = out_dim0
    qdata_dim1_t, scales_dim1 = out_dim1
    return qdata_dim0, qdata_dim1_t, scales_dim0, scales_dim1


if _cutedsl_runtime_available():
    from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
        compute_amax,
        compute_scale_from_amax,
        load_vals_chunk_full,
    )

    @functools.cache
    def _compile_mxfp8_grad_quantize_single_read_cutedsl(
        input_dtype_name: str,
        scaling_mode: str,
    ):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.nvgpu import cpasync, tcgen05
        from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

        if input_dtype_name == "torch.bfloat16":
            INPUT_CUTLASS_DTYPE = cutlass.BFloat16
        else:
            raise ValueError(
                "single-read grad_out MXFP8 quantization currently supports "
                f"only bfloat16 input, got {input_dtype_name}"
            )

        TILE_M = _TILE_M
        TILE_N = _TILE_N
        BLOCK_SIZE = _BLOCK_SIZE
        COMPUTE_WARPS_PER_DIRECTION = _COMPUTE_WARPS_PER_DIRECTION
        THREADS_PER_BLOCK = (1 + 2 * COMPUTE_WARPS_PER_DIRECTION) * 32
        BLOCKS_PER_TILE_M = TILE_M // BLOCK_SIZE
        BLOCKS_PER_TILE_N = TILE_N // BLOCK_SIZE
        M_THREADS = COMPUTE_WARPS_PER_DIRECTION * 32
        N_THREADS = COMPUTE_WARPS_PER_DIRECTION * 32
        M_ITERS_PER_LANE = ceil_div(TILE_M, M_THREADS)
        N_ITERS_PER_LANE = ceil_div(TILE_N, N_THREADS)

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            in_smem: cute.struct.Align[
                cute.struct.MemRange[INPUT_CUTLASS_DTYPE, TILE_M * TILE_N],
                128,
            ]
            out_dim0_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, TILE_M * TILE_N],
                128,
            ]
            out_dim1_smem: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, TILE_M * TILE_N],
                128,
            ]

        class Mxfp8GradQuantizeSingleReadKernel:
            @cute.jit
            def _load_dim0_block(
                self,
                sIN_tile: cute.Tensor,
                m_rel: cutlass.Int32,
                n_base: cutlass.Int32,
            ):
                vals_block = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.Float32)
                for i in range(BLOCK_SIZE):
                    vals_block[i] = cutlass.Float32(sIN_tile[m_rel, n_base + i])
                return vals_block

            @cute.jit
            def _load_dim1_block(
                self,
                sIN_tile: cute.Tensor,
                n_rel: cutlass.Int32,
                m_base: cutlass.Int32,
            ):
                vals_block = cute.make_rmem_tensor((BLOCK_SIZE,), cutlass.Float32)
                for i in range(BLOCK_SIZE):
                    vals_block[i] = cutlass.Float32(sIN_tile[m_base + i, n_rel])
                return vals_block

            @cute.jit
            def _store_q_dim0_chunk(
                self,
                q_fp8_vals4: cute.Tensor,
                sOUT_tile: cute.Tensor,
                m_rel: cutlass.Int32,
                n_base: cutlass.Int32,
            ):
                sOUT_tile_u32 = cute.recast_tensor(sOUT_tile, cutlass.Uint32)
                q_fp8_vals4_u32 = cute.recast_tensor(q_fp8_vals4, cutlass.Uint32)
                sOUT_tile_u32[m_rel, n_base // cutlass.Int32(4)] = q_fp8_vals4_u32[0]

            @cute.jit
            def _store_q_dim1_chunk(
                self,
                q_fp8_vals4: cute.Tensor,
                sOUT_tile: cute.Tensor,
                n_rel: cutlass.Int32,
                m_base: cutlass.Int32,
            ):
                sOUT_tile_u32 = cute.recast_tensor(sOUT_tile, cutlass.Uint32)
                q_fp8_vals4_u32 = cute.recast_tensor(q_fp8_vals4, cutlass.Uint32)
                sOUT_tile_u32[n_rel, m_base // cutlass.Int32(4)] = q_fp8_vals4_u32[0]

            @cute.jit
            def _quantize_dim0_block_to_smem(
                self,
                vals_block: cute.Tensor,
                inv_scale: cutlass.Float32,
                sOUT_tile: cute.Tensor,
                m_rel: cutlass.Int32,
                n_base: cutlass.Int32,
                USE_RCEIL: cutlass.Constexpr[bool],
            ):
                for chunk in cutlass.range_constexpr(BLOCK_SIZE // 4):
                    local_base = chunk * 4
                    vals_chunk = load_vals_chunk_full(vals_block, local_base)
                    q_vals4_vec = vals_chunk.load() * inv_scale
                    if not cutlass.const_expr(USE_RCEIL):
                        q_vals4_vec = cute.where(
                            q_vals4_vec > cutlass.Float32(448.0),
                            cutlass.Float32(448.0),
                            q_vals4_vec,
                        )
                        q_vals4_vec = cute.where(
                            q_vals4_vec < cutlass.Float32(-448.0),
                            cutlass.Float32(-448.0),
                            q_vals4_vec,
                        )
                    q_fp8_vec4 = q_vals4_vec.to(cutlass.Float8E4M3FN)
                    q_fp8_vals4 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
                    q_fp8_vals4.store(q_fp8_vec4)
                    self._store_q_dim0_chunk(
                        q_fp8_vals4, sOUT_tile, m_rel, n_base + local_base
                    )

            @cute.jit
            def _quantize_dim1_block_to_smem(
                self,
                vals_block: cute.Tensor,
                inv_scale: cutlass.Float32,
                sOUT_tile: cute.Tensor,
                n_rel: cutlass.Int32,
                m_base: cutlass.Int32,
                USE_RCEIL: cutlass.Constexpr[bool],
            ):
                for chunk in cutlass.range_constexpr(BLOCK_SIZE // 4):
                    local_base = chunk * 4
                    vals_chunk = load_vals_chunk_full(vals_block, local_base)
                    q_vals4_vec = vals_chunk.load() * inv_scale
                    if not cutlass.const_expr(USE_RCEIL):
                        q_vals4_vec = cute.where(
                            q_vals4_vec > cutlass.Float32(448.0),
                            cutlass.Float32(448.0),
                            q_vals4_vec,
                        )
                        q_vals4_vec = cute.where(
                            q_vals4_vec < cutlass.Float32(-448.0),
                            cutlass.Float32(-448.0),
                            q_vals4_vec,
                        )
                    q_fp8_vec4 = q_vals4_vec.to(cutlass.Float8E4M3FN)
                    q_fp8_vals4 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
                    q_fp8_vals4.store(q_fp8_vec4)
                    self._store_q_dim1_chunk(
                        q_fp8_vals4, sOUT_tile, n_rel, m_base + local_base
                    )

            @cute.jit
            def _store_scales_vec(
                self,
                scales_tensor: cute.Tensor,
                row: cutlass.Int64,
                col_block_base: cutlass.Int64,
                scale_buffer: cute.Tensor,
                NUM_SCALE_GROUPS: cutlass.Constexpr[int],
            ):
                scales_tensor_u32 = cute.recast_tensor(scales_tensor, cutlass.Uint32)
                scale_buffer_u32 = cute.recast_tensor(scale_buffer, cutlass.Uint32)
                for group in cutlass.range_constexpr(NUM_SCALE_GROUPS):
                    scales_tensor_u32[
                        row, col_block_base // cutlass.Int64(4) + group
                    ] = scale_buffer_u32[group]

            @cute.jit
            def _issue_tma_load(
                self,
                tma_atom_in: cute.CopyAtom,
                gIN_tile: cute.Tensor,
                sIN_tile: cute.Tensor,
                tma_mbar_ptr: cutlass.Int64,
                warp_idx: cutlass.Int32,
            ):
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
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            tma_mbar_ptr,
                            TILE_M * TILE_N * 2,
                        )
                    cute.copy(
                        tma_atom_in,
                        tINg[(None, 0)],
                        tINs[(None, 0)],
                        tma_bar_ptr=tma_mbar_ptr,
                    )

            @cute.jit
            def _issue_two_tma_stores(
                self,
                tma_atom_out_dim0: cute.CopyAtom,
                gOUT_dim0_tile: cute.Tensor,
                sOUT_dim0_tile: cute.Tensor,
                tma_atom_out_dim1: cute.CopyAtom,
                gOUT_dim1_tile: cute.Tensor,
                sOUT_dim1_tile: cute.Tensor,
                warp_idx: cutlass.Int32,
            ):
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                if warp_idx == 0:
                    cta_layout = cute.make_layout((1,))
                    sOUT_dim0_for_tma_partition = cute.group_modes(sOUT_dim0_tile, 0, 1)
                    gOUT_dim0_for_tma_partition = cute.group_modes(gOUT_dim0_tile, 0, 1)
                    tOUT_dim0_s, tOUT_dim0_g = cpasync.tma_partition(
                        tma_atom_out_dim0,
                        0,
                        cta_layout,
                        sOUT_dim0_for_tma_partition,
                        gOUT_dim0_for_tma_partition,
                    )
                    cute.copy(
                        tma_atom_out_dim0,
                        tOUT_dim0_s[(None, 0)],
                        tOUT_dim0_g[(None, 0)],
                    )

                    sOUT_dim1_for_tma_partition = cute.group_modes(sOUT_dim1_tile, 0, 1)
                    gOUT_dim1_for_tma_partition = cute.group_modes(gOUT_dim1_tile, 0, 1)
                    tOUT_dim1_s, tOUT_dim1_g = cpasync.tma_partition(
                        tma_atom_out_dim1,
                        0,
                        cta_layout,
                        sOUT_dim1_for_tma_partition,
                        gOUT_dim1_for_tma_partition,
                    )
                    cute.copy(
                        tma_atom_out_dim1,
                        tOUT_dim1_s[(None, 0)],
                        tOUT_dim1_g[(None, 0)],
                    )

            @cute.kernel
            def kernel(
                self,
                inp_mn: cute.Tensor,
                tma_atom_in: cute.CopyAtom,
                tma_tensor_in: cute.Tensor,
                out_dim0_mn: cute.Tensor,
                tma_atom_out_dim0: cute.CopyAtom,
                tma_tensor_out_dim0: cute.Tensor,
                out_dim1_nm: cute.Tensor,
                tma_atom_out_dim1: cute.CopyAtom,
                tma_tensor_out_dim1: cute.Tensor,
                scales_dim0_u8: cute.Tensor,
                scales_dim1_u8: cute.Tensor,
                M: cutlass.Int64,
                N: cutlass.Int64,
                scales_dim0_layout: cute.Layout,
                scales_dim1_layout: cute.Layout,
                USE_RCEIL: cutlass.Constexpr[bool],
            ):
                tidx, _, _ = cute.arch.thread_idx()
                warp_idx = cute.arch.warp_idx()
                warp_idx = cute.arch.make_warp_uniform(warp_idx)
                bid_n, bid_m, _ = cute.arch.block_idx()

                smem_allocator = utils.SmemAllocator()
                storage = smem_allocator.allocate(SharedStorage)
                tma_mbar_ptr = storage.tma_mbar_ptr.data_ptr()

                smem_layout_in = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
                smem_layout_out_dim0 = cute.make_layout(
                    (TILE_M, TILE_N), stride=(TILE_N, 1)
                )
                smem_layout_out_dim1 = cute.make_layout(
                    (TILE_N, TILE_M), stride=(TILE_M, 1)
                )
                sIN_tile = storage.in_smem.get_tensor(smem_layout_in)
                sOUT_dim0_tile = storage.out_dim0_smem.get_tensor(smem_layout_out_dim0)
                sOUT_dim1_tile = storage.out_dim1_smem.get_tensor(smem_layout_out_dim1)

                if tidx == 0:
                    cpasync.prefetch_descriptor(tma_atom_in)
                    cpasync.prefetch_descriptor(tma_atom_out_dim0)
                    cpasync.prefetch_descriptor(tma_atom_out_dim1)
                    cute.arch.mbarrier_init(tma_mbar_ptr, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.sync_threads()

                gIN_tile = cute.local_tile(
                    tma_tensor_in, (TILE_M, TILE_N), (bid_m, bid_n)
                )
                self._issue_tma_load(
                    tma_atom_in,
                    gIN_tile,
                    sIN_tile,
                    tma_mbar_ptr,
                    warp_idx,
                )

                if warp_idx >= 1 and warp_idx <= COMPUTE_WARPS_PER_DIRECTION:
                    cute.arch.mbarrier_wait(tma_mbar_ptr, 0)
                    lane = tidx % 32
                    m_lane = (warp_idx - 1) * 32 + lane

                    scales_dim0 = cute.make_tensor(
                        scales_dim0_u8.iterator,
                        scales_dim0_layout,
                    )

                    for mm in cutlass.range_constexpr(M_ITERS_PER_LANE):
                        m_rel = m_lane + mm * M_THREADS
                        m = bid_m * TILE_M + m_rel
                        if m_rel < TILE_M:
                            scale_buffer = cute.make_rmem_tensor(
                                (BLOCKS_PER_TILE_N,), cutlass.Uint8
                            )
                            for nb in cutlass.range_constexpr(BLOCKS_PER_TILE_N):
                                n_base = nb * BLOCK_SIZE
                                vals_block = self._load_dim0_block(
                                    sIN_tile,
                                    m_rel,
                                    n_base,
                                )
                                amax = compute_amax(vals_block)
                                scale_biased, inv_scale = compute_scale_from_amax(
                                    amax, USE_RCEIL
                                )
                                scale_buffer[nb] = cutlass.Uint8(scale_biased)
                                self._quantize_dim0_block_to_smem(
                                    vals_block,
                                    inv_scale,
                                    sOUT_dim0_tile,
                                    m_rel,
                                    n_base,
                                    USE_RCEIL,
                                )
                            col_block_base = bid_n * BLOCKS_PER_TILE_N
                            self._store_scales_vec(
                                scales_dim0,
                                m,
                                col_block_base,
                                scale_buffer,
                                BLOCKS_PER_TILE_N // 4,
                            )

                if (
                    warp_idx >= COMPUTE_WARPS_PER_DIRECTION + 1
                    and warp_idx <= 2 * COMPUTE_WARPS_PER_DIRECTION
                ):
                    cute.arch.mbarrier_wait(tma_mbar_ptr, 0)
                    lane = tidx % 32
                    n_lane = (warp_idx - COMPUTE_WARPS_PER_DIRECTION - 1) * 32 + lane
                    scales_dim1 = cute.make_tensor(
                        scales_dim1_u8.iterator,
                        scales_dim1_layout,
                    )

                    for nn in cutlass.range_constexpr(N_ITERS_PER_LANE):
                        n_rel = n_lane + nn * N_THREADS
                        n = bid_n * TILE_N + n_rel
                        if n_rel < TILE_N:
                            scale_buffer = cute.make_rmem_tensor(
                                (BLOCKS_PER_TILE_M,), cutlass.Uint8
                            )
                            for mb in cutlass.range_constexpr(BLOCKS_PER_TILE_M):
                                m_base = mb * BLOCK_SIZE
                                vals_block = self._load_dim1_block(
                                    sIN_tile,
                                    n_rel,
                                    m_base,
                                )
                                amax = compute_amax(vals_block)
                                scale_biased, inv_scale = compute_scale_from_amax(
                                    amax, USE_RCEIL
                                )
                                scale_buffer[mb] = cutlass.Uint8(scale_biased)
                                self._quantize_dim1_block_to_smem(
                                    vals_block,
                                    inv_scale,
                                    sOUT_dim1_tile,
                                    n_rel,
                                    m_base,
                                    USE_RCEIL,
                                )
                            col_block_base = bid_m * BLOCKS_PER_TILE_M
                            self._store_scales_vec(
                                scales_dim1,
                                n,
                                col_block_base,
                                scale_buffer,
                                BLOCKS_PER_TILE_M // 4,
                            )

                gOUT_dim0_tile = cute.local_tile(
                    tma_tensor_out_dim0,
                    (TILE_M, TILE_N),
                    (bid_m, bid_n),
                )
                gOUT_dim1_tile = cute.local_tile(
                    tma_tensor_out_dim1,
                    (TILE_N, TILE_M),
                    (bid_n, bid_m),
                )
                self._issue_two_tma_stores(
                    tma_atom_out_dim0,
                    gOUT_dim0_tile,
                    sOUT_dim0_tile,
                    tma_atom_out_dim1,
                    gOUT_dim1_tile,
                    sOUT_dim1_tile,
                    warp_idx,
                )

            @cute.jit
            def __call__(
                self,
                inp_mn: cute.Tensor,
                out_dim0_mn: cute.Tensor,
                out_dim1_nm: cute.Tensor,
                scales_dim0_u8: cute.Tensor,
                scales_dim1_u8: cute.Tensor,
                M: cutlass.Int64,
                N: cutlass.Int64,
                stream: cuda.CUstream,
            ):
                smem_layout_in = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
                smem_layout_out_dim0 = cute.make_layout(
                    (TILE_M, TILE_N), stride=(TILE_N, 1)
                )
                smem_layout_out_dim1 = cute.make_layout(
                    (TILE_N, TILE_M), stride=(TILE_M, 1)
                )
                g2s_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
                tma_atom_in, tma_tensor_in = cpasync.make_tiled_tma_atom(
                    g2s_op,
                    inp_mn,
                    smem_layout_in,
                    (TILE_M, TILE_N),
                )
                tma_atom_out_dim0, tma_tensor_out_dim0 = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileS2GOp(),
                    out_dim0_mn,
                    smem_layout_out_dim0,
                    (TILE_M, TILE_N),
                )
                tma_atom_out_dim1, tma_tensor_out_dim1 = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileS2GOp(),
                    out_dim1_nm,
                    smem_layout_out_dim1,
                    (TILE_N, TILE_M),
                )

                scale_cols_dim0 = cute.round_up(N // cutlass.Int64(BLOCK_SIZE), 4)
                scale_cols_dim1 = cute.round_up(M // cutlass.Int64(BLOCK_SIZE), 4)
                scales_dim0_layout = cute.make_layout(
                    (
                        (32, 4, M // cutlass.Int64(128)),
                        (4, scale_cols_dim0 // cutlass.Int64(4)),
                    ),
                    stride=(
                        (16, 4, cutlass.Int64(128) * scale_cols_dim0),
                        (1, cutlass.Int64(512)),
                    ),
                )
                scales_dim1_layout = cute.make_layout(
                    (
                        (32, 4, N // cutlass.Int64(128)),
                        (4, scale_cols_dim1 // cutlass.Int64(4)),
                    ),
                    stride=(
                        (16, 4, cutlass.Int64(128) * scale_cols_dim1),
                        (1, cutlass.Int64(512)),
                    ),
                )
                self.kernel(
                    inp_mn,
                    tma_atom_in,
                    tma_tensor_in,
                    out_dim0_mn,
                    tma_atom_out_dim0,
                    tma_tensor_out_dim0,
                    out_dim1_nm,
                    tma_atom_out_dim1,
                    tma_tensor_out_dim1,
                    scales_dim0_u8,
                    scales_dim1_u8,
                    M,
                    N,
                    scales_dim0_layout,
                    scales_dim1_layout,
                    USE_RCEIL=(scaling_mode == "rceil"),
                ).launch(
                    grid=(cute.ceil_div(N, TILE_N), cute.ceil_div(M, TILE_M), 1),
                    block=(THREADS_PER_BLOCK, 1, 1),
                    cluster=(1, 1, 1),
                    smem=SharedStorage.size_in_bytes(),  # pyrefly: ignore [missing-attribute]
                    stream=stream,
                )

        kernel = Mxfp8GradQuantizeSingleReadKernel()

        m = cute.sym_int(divisibility=128)
        n = cute.sym_int(divisibility=128)
        inp_stride_m = cute.sym_int()
        inp_stride_n = cute.sym_int()
        out0_stride_m = cute.sym_int()
        out0_stride_n = cute.sym_int()
        out1_stride_n = cute.sym_int()
        out1_stride_m = cute.sym_int()
        scale_stride = cute.sym_int()
        fake_inp = make_fake_tensor(
            INPUT_CUTLASS_DTYPE,
            (m, n),
            stride=(inp_stride_m, inp_stride_n),
        )
        fake_out_dim0 = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (m, n),
            stride=(out0_stride_m, out0_stride_n),
        )
        fake_out_dim1 = make_fake_tensor(
            cutlass.Float8E4M3FN,
            (n, m),
            stride=(out1_stride_n, out1_stride_m),
        )
        fake_scales_dim0 = make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(),),
            stride=(scale_stride,),
        )
        fake_scales_dim1 = make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(),),
            stride=(scale_stride,),
        )
        fake_stream = make_fake_stream()

        return cute.compile(
            kernel,
            inp_mn=fake_inp,
            out_dim0_mn=fake_out_dim0,
            out_dim1_nm=fake_out_dim1,
            scales_dim0_u8=fake_scales_dim0,
            scales_dim1_u8=fake_scales_dim1,
            M=0,
            N=0,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )


def cutedsl_mxfp8_quantize_dim0_dim1_single_read(
    x: Tensor,
    scaling_mode: str = "rceil",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Single-read CuTeDSL MXFP8 quantization of ``grad_out`` for backward.

    This reads a row-major ``(M, N)`` bf16 tensor once and emits both MXFP8
    operands needed by grouped-GEMM backward:

    * ``qdata_dim0`` / ``scales_dim0`` for ``grad_out @ weight``.
    * ``qdata_dim1_t`` / ``scales_dim1`` for ``grad_out.T @ x``.

    V1 intentionally targets the review shape class where ``M`` and ``N`` are
    multiples of 128, so both scale tensors can be written directly to the
    tcgen05 blocked layout without tail handling.
    """
    if not _cutedsl_runtime_available():
        raise NotImplementedError("CuTeDSL runtime packages are not available")
    assert x.dtype == torch.bfloat16, f"x must be bfloat16, got {x.dtype}"
    assert x.is_cuda, "x must be CUDA"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.ndim == 2, f"x must be 2D, got {x.ndim}D"
    assert scaling_mode in ("rceil", "floor"), (
        f"scaling_mode must be 'rceil' or 'floor', got {scaling_mode}"
    )
    M, N = x.shape
    assert M % _TILE_M == 0, f"M must be a multiple of {_TILE_M}, got {M}"
    assert N % _TILE_N == 0, f"N must be a multiple of {_TILE_N}, got {N}"
    if torch.cuda.get_device_capability()[0] != 10:
        raise NotImplementedError("MXFP8 CuTeDSL kernels require CUDA SM 10.x")

    qdata_dim0 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    qdata_dim1_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_cols_dim0 = ceil_div(N // _BLOCK_SIZE, 4) * 4
    scale_cols_dim1 = ceil_div(M // _BLOCK_SIZE, 4) * 4
    scales_dim0_u8 = torch.empty(
        (M * scale_cols_dim0,), dtype=torch.uint8, device=x.device
    )
    scales_dim1_u8 = torch.empty(
        (N * scale_cols_dim1,), dtype=torch.uint8, device=x.device
    )

    compiled = _compile_mxfp8_grad_quantize_single_read_cutedsl(
        str(x.dtype),
        scaling_mode,
    )

    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        x,
        qdata_dim0,
        qdata_dim1_t,
        scales_dim0_u8,
        scales_dim1_u8,
        int(M),
        int(N),
        stream,
    )
    return (
        qdata_dim0,
        qdata_dim1_t,
        scales_dim0_u8.view(torch.float8_e8m0fnu),
        scales_dim1_u8.view(torch.float8_e8m0fnu),
    )


__all__ = [
    "cutedsl_mxfp8_quantize_dim0_dim1_single_read",
    "cutedsl_mxfp8_quantize_dim0_dim1_streams",
]
