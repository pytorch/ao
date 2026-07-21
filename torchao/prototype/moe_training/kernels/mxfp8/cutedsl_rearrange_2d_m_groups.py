# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch

from torchao.utils import ceil_div


@functools.cache
def _compile_mx_block_rearrange_2d_m_groups_cutedsl(
    num_groups: int,
    chunk_width: int,
    cols_multiple_of_16: bool,
    all_col_chunks_full: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import cpasync
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    NUM_GROUPS = num_groups
    SF_ROWS = 128
    SF_COLS = 4
    CHUNK_WIDTH = chunk_width
    THREADS_PER_ROW = CHUNK_WIDTH // 16
    THREADS_PER_BLOCK = SF_ROWS * THREADS_PER_ROW
    SF_TILE_BYTES = SF_ROWS * SF_COLS
    SF_TILES_PER_CHUNK = CHUNK_WIDTH // SF_COLS
    CHUNK_SIZE_BYTES = SF_ROWS * CHUNK_WIDTH
    COLS_MULTIPLE_OF_16 = cols_multiple_of_16
    ALL_COL_CHUNKS_FULL = all_col_chunks_full

    @cute.struct
    class SharedStorage:
        group_data: cute.struct.MemRange[cutlass.Int32, NUM_GROUPS * 2]
        group_bounds: cute.struct.MemRange[cutlass.Int32, 4]
        out_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, CHUNK_SIZE_BYTES],
            128,
        ]

    class MXBlockRearrange2dMGroups:
        @cute.jit
        def _ceil_div(self, a: cutlass.Int32, b: cutlass.Int32):
            return (a + b - 1) // b

        @cute.jit
        def _fill_group_data(
            self,
            input_offsets: cute.Tensor,
            group_data: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                chunk_cumsum = cutlass.Int32(0)
                prev_offset = cutlass.Int32(0)
                for g in range(NUM_GROUPS):
                    end = cutlass.Int32((input_offsets.iterator + g).load())
                    group_size = end - prev_offset
                    chunks_in_group = self._ceil_div(group_size, SF_ROWS)
                    chunk_cumsum = chunk_cumsum + chunks_in_group
                    (group_data.iterator + g).store(cutlass.Int32(chunks_in_group))
                    (group_data.iterator + NUM_GROUPS + g).store(
                        cutlass.Int32(chunk_cumsum)
                    )
                    prev_offset = end
            cute.arch.sync_threads()

        @cute.jit
        def _fill_group_bounds(
            self,
            input_offsets: cute.Tensor,
            row_chunk: cutlass.Int32,
            group_data: cute.Tensor,
            group_bounds: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            if tidx == 0:
                group = cutlass.Int32(0)
                chunk_idx = cutlass.Int32(0)
                active = cutlass.Int32(0)
                chunk_cumsum_before = cutlass.Int32(0)
                for g in range(NUM_GROUPS):
                    chunk_cumsum = cutlass.Int32(
                        (group_data.iterator + NUM_GROUPS + g).load()
                    )
                    if row_chunk < chunk_cumsum and active == 0:
                        group = cutlass.Int32(g)
                        chunk_idx = row_chunk - chunk_cumsum_before
                        active = cutlass.Int32(1)
                    chunk_cumsum_before = chunk_cumsum
                group_start = cutlass.Int32(0)
                group_end = cutlass.Int32(0)
                input_row_base = cutlass.Int32(0)
                if active != 0:
                    if group > 0:
                        group_start = cutlass.Int32(
                            (input_offsets.iterator + group - 1).load()
                        )
                    group_end = cutlass.Int32((input_offsets.iterator + group).load())
                    input_row_base = group_start + chunk_idx * SF_ROWS
                (group_bounds.iterator + 0).store(group_start)
                (group_bounds.iterator + 1).store(group_end)
                (group_bounds.iterator + 2).store(input_row_base)
                (group_bounds.iterator + 3).store(active)
            cute.arch.sync_threads()

        @cute.jit
        def _load4(
            self,
            input_scales: cute.Tensor,
            input_offset: cutlass.Int32,
            input_col: cutlass.Int32,
            cols: cutlass.Int32,
        ):
            value = cutlass.Uint32(0)
            if input_col + 3 < cols and input_offset % 4 == 0:
                value = (
                    cute.recast_ptr(
                        input_scales.iterator + input_offset, dtype=cutlass.Uint32
                    )
                    .load()
                    .to(cutlass.Uint32)
                )
            elif input_col < cols:
                value = (input_scales.iterator + input_offset).load().to(cutlass.Uint32)
                if input_col + 1 < cols:
                    value = (
                        value
                        | (
                            (input_scales.iterator + input_offset + 1)
                            .load()
                            .to(cutlass.Uint32)
                            << 8
                        )
                    ).to(cutlass.Uint32)
                if input_col + 2 < cols:
                    value = (
                        value
                        | (
                            (input_scales.iterator + input_offset + 2)
                            .load()
                            .to(cutlass.Uint32)
                            << 16
                        )
                    ).to(cutlass.Uint32)
                if input_col + 3 < cols:
                    value = (
                        value
                        | (
                            (input_scales.iterator + input_offset + 3)
                            .load()
                            .to(cutlass.Uint32)
                            << 24
                        )
                    ).to(cutlass.Uint32)
            return value

        @cute.jit
        def _process_chunk(
            self,
            input_scales: cute.Tensor,
            input_offsets: cute.Tensor,
            output_scales: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            padded_cols: cutlass.Int32,
            row_chunk: cutlass.Int32,
            group_data: cute.Tensor,
            group_bounds: cute.Tensor,
            sOUT_tile: cute.Tensor,
            bulk_store_atom: cute.CopyAtom,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            col_chunk_pid, _, _ = cute.arch.block_idx()
            col_chunk = cutlass.Int32(col_chunk_pid)
            row_idx = cutlass.Int32(tidx // THREADS_PER_ROW)
            col_lane = cutlass.Int32(tidx - (tidx // THREADS_PER_ROW) * THREADS_PER_ROW)
            self._fill_group_bounds(
                input_offsets,
                row_chunk,
                group_data,
                group_bounds,
            )
            group_end = cutlass.Int32((group_bounds.iterator + 1).load())
            input_row_base = cutlass.Int32((group_bounds.iterator + 2).load())
            active = cutlass.Int32((group_bounds.iterator + 3).load())
            input_row = cutlass.Int32(0)
            chunk_col_base = cutlass.Int32(0)
            cols_in_chunk = cutlass.Int32(0)
            if active != 0:
                input_row = input_row_base + row_idx
                chunk_col_base = col_chunk * CHUNK_WIDTH + col_lane * 16
                if not ALL_COL_CHUNKS_FULL:
                    cols_in_chunk = cols - col_chunk * CHUNK_WIDTH
                    if cols_in_chunk > CHUNK_WIDTH:
                        cols_in_chunk = cutlass.Int32(CHUNK_WIDTH)

            row_valid = cutlass.Int32(0)
            if active != 0 and input_row < group_end and input_row < rows:
                row_valid = cutlass.Int32(1)
            full_col_chunk = cutlass.Int32(0)
            if not ALL_COL_CHUNKS_FULL:
                if cols_in_chunk == CHUNK_WIDTH:
                    full_col_chunk = cutlass.Int32(1)
            col_valid = cutlass.Int32(0)
            if ALL_COL_CHUNKS_FULL:
                if active != 0:
                    col_valid = cutlass.Int32(1)
            else:
                if active != 0 and col_lane * 16 < cols_in_chunk:
                    col_valid = cutlass.Int32(1)

            if col_valid != 0:
                input_offset = input_row * cols + chunk_col_base
                r_div_32 = row_idx // 32
                r_mod_32 = row_idx - r_div_32 * 32
                blocked_layout_row_offset = r_mod_32 * 16 + r_div_32 * 4
                thread_col_start = col_lane * 16
                smem_offset0 = blocked_layout_row_offset + col_lane * 4 * SF_TILE_BYTES
                smem_offset1 = (
                    blocked_layout_row_offset + (col_lane * 4 + 1) * SF_TILE_BYTES
                )
                smem_offset2 = (
                    blocked_layout_row_offset + (col_lane * 4 + 2) * SF_TILE_BYTES
                )
                smem_offset3 = (
                    blocked_layout_row_offset + (col_lane * 4 + 3) * SF_TILE_BYTES
                )
                value0 = cutlass.Uint32(0)
                value1 = cutlass.Uint32(0)
                value2 = cutlass.Uint32(0)
                value3 = cutlass.Uint32(0)
                if ALL_COL_CHUNKS_FULL:
                    if row_valid != 0:
                        input_ptr = (input_scales.iterator + input_offset).align(16)
                        value = cute.recast_ptr(
                            input_ptr,
                            dtype=cutlass.Int128,
                        )
                        value = value.load()
                        value0 = value.to(cutlass.Uint32)
                        value1 = (value >> 32).to(cutlass.Uint32)
                        value2 = (value >> 64).to(cutlass.Uint32)
                        value3 = (value >> 96).to(cutlass.Uint32)
                    cute.recast_ptr(
                        sOUT_tile.iterator + smem_offset0, dtype=cutlass.Uint32
                    ).store(value0)
                    cute.recast_ptr(
                        sOUT_tile.iterator + smem_offset1, dtype=cutlass.Uint32
                    ).store(value1)
                    cute.recast_ptr(
                        sOUT_tile.iterator + smem_offset2, dtype=cutlass.Uint32
                    ).store(value2)
                    cute.recast_ptr(
                        sOUT_tile.iterator + smem_offset3, dtype=cutlass.Uint32
                    ).store(value3)
                elif COLS_MULTIPLE_OF_16:
                    if full_col_chunk != 0:
                        if row_valid != 0:
                            input_ptr = (input_scales.iterator + input_offset).align(16)
                            value = cute.recast_ptr(
                                input_ptr,
                                dtype=cutlass.Int128,
                            )
                            value = value.load()
                            value0 = value.to(cutlass.Uint32)
                            value1 = (value >> 32).to(cutlass.Uint32)
                            value2 = (value >> 64).to(cutlass.Uint32)
                            value3 = (value >> 96).to(cutlass.Uint32)
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset0, dtype=cutlass.Uint32
                        ).store(value0)
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset1, dtype=cutlass.Uint32
                        ).store(value1)
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset2, dtype=cutlass.Uint32
                        ).store(value2)
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset3, dtype=cutlass.Uint32
                        ).store(value3)
                    else:
                        if row_valid != 0:
                            value0 = self._load4(
                                input_scales,
                                input_offset,
                                chunk_col_base,
                                cols,
                            )
                            value1 = self._load4(
                                input_scales,
                                input_offset + 4,
                                chunk_col_base + 4,
                                cols,
                            )
                            value2 = self._load4(
                                input_scales,
                                input_offset + 8,
                                chunk_col_base + 8,
                                cols,
                            )
                            value3 = self._load4(
                                input_scales,
                                input_offset + 12,
                                chunk_col_base + 12,
                                cols,
                            )
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset0, dtype=cutlass.Uint32
                        ).store(value0)
                        if thread_col_start + 4 < cols_in_chunk:
                            cute.recast_ptr(
                                sOUT_tile.iterator + smem_offset1,
                                dtype=cutlass.Uint32,
                            ).store(value1)
                        if thread_col_start + 8 < cols_in_chunk:
                            cute.recast_ptr(
                                sOUT_tile.iterator + smem_offset2,
                                dtype=cutlass.Uint32,
                            ).store(value2)
                        if thread_col_start + 12 < cols_in_chunk:
                            cute.recast_ptr(
                                sOUT_tile.iterator + smem_offset3,
                                dtype=cutlass.Uint32,
                            ).store(value3)
                else:
                    if (
                        row_valid != 0
                        and chunk_col_base + 8 <= cols
                        and input_offset % 8 == 0
                    ):
                        input_ptr = (input_scales.iterator + input_offset).align(8)
                        value = cute.recast_ptr(input_ptr, dtype=cutlass.Uint64).load()
                        value0 = (value & cutlass.Uint64(0xFFFFFFFF)).to(cutlass.Uint32)
                        value1 = ((value >> 32) & cutlass.Uint64(0xFFFFFFFF)).to(
                            cutlass.Uint32
                        )
                        value2 = self._load4(
                            input_scales,
                            input_offset + 8,
                            chunk_col_base + 8,
                            cols,
                        )
                        value3 = self._load4(
                            input_scales,
                            input_offset + 12,
                            chunk_col_base + 12,
                            cols,
                        )
                    elif row_valid != 0:
                        value0 = self._load4(
                            input_scales,
                            input_offset,
                            chunk_col_base,
                            cols,
                        )
                        value1 = self._load4(
                            input_scales,
                            input_offset + 4,
                            chunk_col_base + 4,
                            cols,
                        )
                        value2 = self._load4(
                            input_scales,
                            input_offset + 8,
                            chunk_col_base + 8,
                            cols,
                        )
                        value3 = self._load4(
                            input_scales,
                            input_offset + 12,
                            chunk_col_base + 12,
                            cols,
                        )
                    cute.recast_ptr(
                        sOUT_tile.iterator + smem_offset0, dtype=cutlass.Uint32
                    ).store(value0)
                    if thread_col_start + 4 < cols_in_chunk:
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset1, dtype=cutlass.Uint32
                        ).store(value1)
                    if thread_col_start + 8 < cols_in_chunk:
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset2, dtype=cutlass.Uint32
                        ).store(value2)
                    if thread_col_start + 12 < cols_in_chunk:
                        cute.recast_ptr(
                            sOUT_tile.iterator + smem_offset3, dtype=cutlass.Uint32
                        ).store(value3)

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if tidx == 0 and active != 0:
                sf_tiles_per_row = padded_cols // SF_COLS
                out_tile_base = (
                    row_chunk * sf_tiles_per_row + col_chunk * SF_TILES_PER_CHUNK
                )
                if ALL_COL_CHUNKS_FULL:
                    for tile in range(SF_TILES_PER_CHUNK):
                        src = cute.make_tensor(
                            sOUT_tile.iterator + tile * SF_TILE_BYTES,
                            cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                        )
                        dst = cute.make_tensor(
                            output_scales.iterator
                            + (out_tile_base + tile) * SF_TILE_BYTES,
                            cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                        )
                        cute.copy(bulk_store_atom, src, dst)
                elif full_col_chunk != 0:
                    for tile in range(SF_TILES_PER_CHUNK):
                        src = cute.make_tensor(
                            sOUT_tile.iterator + tile * SF_TILE_BYTES,
                            cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                        )
                        dst = cute.make_tensor(
                            output_scales.iterator
                            + (out_tile_base + tile) * SF_TILE_BYTES,
                            cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                        )
                        cute.copy(bulk_store_atom, src, dst)
                else:
                    valid_sf_tiles = (cols_in_chunk + SF_COLS - 1) // SF_COLS
                    for tile in range(SF_TILES_PER_CHUNK):
                        if tile < valid_sf_tiles:
                            src = cute.make_tensor(
                                sOUT_tile.iterator + tile * SF_TILE_BYTES,
                                cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                            )
                            dst = cute.make_tensor(
                                output_scales.iterator
                                + (out_tile_base + tile) * SF_TILE_BYTES,
                                cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                            )
                            cute.copy(bulk_store_atom, src, dst)
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        @cute.kernel
        def kernel(
            self,
            input_scales: cute.Tensor,
            input_offsets: cute.Tensor,
            output_scales: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            padded_cols: cutlass.Int32,
            bulk_store_atom: cute.CopyAtom,
        ):
            _, row_superblock_pid, _ = cute.arch.block_idx()
            row_superblock = cutlass.Int32(row_superblock_pid)
            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            group_data = storage.group_data.get_tensor(
                cute.make_layout((NUM_GROUPS * 2,), stride=(1,))
            )
            group_bounds = storage.group_bounds.get_tensor(
                cute.make_layout((4,), stride=(1,))
            )
            sOUT_tile = storage.out_smem.get_tensor(
                cute.make_layout((CHUNK_SIZE_BYTES,), stride=(1,))
            )
            self._fill_group_data(input_offsets, group_data)
            self._process_chunk(
                input_scales,
                input_offsets,
                output_scales,
                rows,
                cols,
                padded_cols,
                row_superblock,
                group_data,
                group_bounds,
                sOUT_tile,
                bulk_store_atom,
            )

        @cute.jit
        def __call__(
            self,
            input_scales: cute.Tensor,
            input_offsets: cute.Tensor,
            output_scales: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            num_row_chunks: cutlass.Int32,
            num_col_chunks: cutlass.Int32,
            padded_cols: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            bulk_store_atom = cute.make_copy_atom(
                cpasync.CopyBulkS2GOp(),
                cutlass.Uint8,
                num_bits_per_copy=SF_TILE_BYTES * 8,
            )
            self.kernel(
                input_scales,
                input_offsets,
                output_scales,
                rows,
                cols,
                padded_cols,
                bulk_store_atom,
            ).launch(
                grid=(num_col_chunks, num_row_chunks, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),
                stream=stream,
            )

    kernel = MXBlockRearrange2dMGroups()

    m = cute.sym_int(divisibility=128)
    c = cute.sym_int(divisibility=16) if cols_multiple_of_16 else cute.sym_int()
    out_n = cute.sym_int()
    offs_stride = cute.sym_int()
    out_stride = cute.sym_int()

    fake_input = make_fake_tensor(
        cutlass.Uint8,
        (m, c),
        stride=(c, 1),
        assumed_align=16,
    )
    fake_output = make_fake_tensor(
        cutlass.Uint8,
        (out_n,),
        stride=(out_stride,),
    )
    fake_offsets = make_fake_tensor(
        cutlass.Int32,
        (NUM_GROUPS,),
        stride=(offs_stride,),
    )
    fake_stream = make_fake_stream()

    return cute.compile(
        kernel,
        input_scales=fake_input,
        input_offsets=fake_offsets,
        output_scales=fake_output,
        rows=0,
        cols=0,
        num_row_chunks=1,
        num_col_chunks=1,
        padded_cols=16,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def mx_block_rearrange_2d_m_groups_cutedsl(
    scales_tensor: torch.Tensor,
    input_offsets: torch.Tensor,
    chunk_width: int | None = None,
) -> torch.Tensor:
    assert scales_tensor.ndim == 2
    assert scales_tensor.element_size() == 1
    assert input_offsets.ndim == 1
    assert input_offsets.dtype == torch.int32
    assert scales_tensor.is_cuda
    assert input_offsets.is_cuda
    assert scales_tensor.is_contiguous()
    rows, cols = scales_tensor.shape
    num_groups = input_offsets.shape[0]
    padded_rows = rows + num_groups * 128
    padded_cols = ceil_div(cols, 4) * 4
    if chunk_width is None:
        if cols >= 64:
            chunk_width = 64
        elif cols >= 32:
            chunk_width = 32
        else:
            chunk_width = 16
    assert chunk_width in (16, 32, 64, 128)
    output = torch.zeros(
        (padded_rows, padded_cols),
        device=scales_tensor.device,
        dtype=torch.uint8,
    )
    compiled = _compile_mx_block_rearrange_2d_m_groups_cutedsl(
        num_groups,
        chunk_width,
        cols % 16 == 0,
        cols % chunk_width == 0,
    )
    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        scales_tensor.view(torch.uint8),
        input_offsets,
        output.view(-1),
        int(rows),
        int(cols),
        int(ceil_div(rows, 128) + num_groups),
        int(ceil_div(cols, chunk_width)),
        int(padded_cols),
        stream,
    )
    return output.view(scales_tensor.dtype)
