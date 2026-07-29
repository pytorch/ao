# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch

from torchao.utils import ceil_div


@functools.cache
def _compile_mx_block_rearrange_2d_k_groups_cutedsl(
    num_groups: int,
    chunk_width: int,
    cols_multiple_of_16: bool,
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

    @cute.struct
    class SharedStorage:
        out_smem: cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, CHUNK_SIZE_BYTES],
            128,
        ]

    class MXBlockRearrange2dKGroups:
        @cute.jit
        def _ceil_div(self, a: cutlass.Int32, b: cutlass.Int32):
            return (a + b - 1) // b

        @cute.jit
        def _group_start_after_padding(
            self,
            input_offsets: cute.Tensor,
            group: cutlass.Int32,
        ):
            start_after_padding = cutlass.Int32(0)
            prev = cutlass.Int32(0)
            for g in range(NUM_GROUPS):
                end = cutlass.Int32((input_offsets.iterator + g).load())
                if cutlass.Int32(g) < group:
                    group_size = end - prev
                    start_after_padding = (
                        start_after_padding
                        + self._ceil_div(
                            group_size,
                            SF_COLS,
                        )
                        * SF_COLS
                    )
                prev = end
            return start_after_padding

        @cute.jit
        def _load4(
            self,
            input_scales: cute.Tensor,
            input_offset: cutlass.Int32,
            input_col: cutlass.Int32,
            group_end: cutlass.Int32,
        ):
            value = cutlass.Uint32(0)
            if input_col + 3 < group_end and input_offset % 4 == 0:
                value = (
                    cute.recast_ptr(
                        input_scales.iterator + input_offset,
                        dtype=cutlass.Uint32,
                    )
                    .load()
                    .to(cutlass.Uint32)
                )
            elif input_col < group_end:
                value = (input_scales.iterator + input_offset).load().to(cutlass.Uint32)
                if input_col + 1 < group_end:
                    value = (
                        value
                        | (
                            (input_scales.iterator + input_offset + 1)
                            .load()
                            .to(cutlass.Uint32)
                            << 8
                        )
                    ).to(cutlass.Uint32)
                if input_col + 2 < group_end:
                    value = (
                        value
                        | (
                            (input_scales.iterator + input_offset + 2)
                            .load()
                            .to(cutlass.Uint32)
                            << 16
                        )
                    ).to(cutlass.Uint32)
                if input_col + 3 < group_end:
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

        @cute.kernel
        def kernel(
            self,
            input_scales: cute.Tensor,
            input_offsets: cute.Tensor,
            output_scales: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            padded_rows: cutlass.Int32,
            bulk_store_atom: cute.CopyAtom,
        ):
            flat_col_chunk_pid, row_block_pid, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            flat_col_chunk = cutlass.Int32(flat_col_chunk_pid)
            row_idx = cutlass.Int32(tidx // THREADS_PER_ROW)
            col_lane = cutlass.Int32(tidx - (tidx // THREADS_PER_ROW) * THREADS_PER_ROW)
            row = cutlass.Int32(row_block_pid * SF_ROWS + row_idx)

            group = cutlass.Int32(0)
            group_start = cutlass.Int32(0)
            group_end = cutlass.Int32(0)
            col_chunk = cutlass.Int32(0)
            active = cutlass.Int32(0)
            chunk_cumsum = cutlass.Int32(0)
            prev_end = cutlass.Int32(0)
            for g in range(NUM_GROUPS):
                end = cutlass.Int32((input_offsets.iterator + g).load())
                group_size_for_scan = end - prev_end
                chunks_in_group = self._ceil_div(group_size_for_scan, CHUNK_WIDTH)
                next_chunk_cumsum = chunk_cumsum + chunks_in_group
                if flat_col_chunk < next_chunk_cumsum and active == 0:
                    group = cutlass.Int32(g)
                    group_start = prev_end
                    group_end = end
                    col_chunk = flat_col_chunk - chunk_cumsum
                    active = cutlass.Int32(1)
                chunk_cumsum = next_chunk_cumsum
                prev_end = end

            group_size = group_end - group_start
            group_col_blocks = self._ceil_div(group_size, SF_COLS)

            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(SharedStorage)
            sOUT_tile = storage.out_smem.get_tensor(
                cute.make_layout((SF_TILE_BYTES,), stride=(1,))
            )

            value = cutlass.Uint32(0)
            value1 = cutlass.Uint32(0)
            value2 = cutlass.Uint32(0)
            value3 = cutlass.Uint32(0)
            input_col = group_start + col_chunk * CHUNK_WIDTH + col_lane * 16
            if active != 0 and row < rows:
                input_offset = row * cols + input_col
                if (
                    COLS_MULTIPLE_OF_16
                    and input_col + 15 < group_end
                    and input_offset % 16 == 0
                ):
                    input_ptr = (input_scales.iterator + input_offset).align(16)
                    packed = cute.recast_ptr(input_ptr, dtype=cutlass.Int128).load()
                    value = packed.to(cutlass.Uint32)
                    value1 = (packed >> 32).to(cutlass.Uint32)
                    value2 = (packed >> 64).to(cutlass.Uint32)
                    value3 = (packed >> 96).to(cutlass.Uint32)
                else:
                    value = self._load4(
                        input_scales,
                        input_offset,
                        input_col,
                        group_end,
                    )
                    value1 = self._load4(
                        input_scales,
                        input_offset + 4,
                        input_col + 4,
                        group_end,
                    )
                    value2 = self._load4(
                        input_scales,
                        input_offset + 8,
                        input_col + 8,
                        group_end,
                    )
                    value3 = self._load4(
                        input_scales,
                        input_offset + 12,
                        input_col + 12,
                        group_end,
                    )

            r_div_32 = row_idx // 32
            r_mod_32 = row_idx - r_div_32 * 32
            smem_offset = r_mod_32 * 16 + r_div_32 * 4
            smem_offset0 = smem_offset + col_lane * 4 * SF_TILE_BYTES
            smem_offset1 = smem_offset + (col_lane * 4 + 1) * SF_TILE_BYTES
            smem_offset2 = smem_offset + (col_lane * 4 + 2) * SF_TILE_BYTES
            smem_offset3 = smem_offset + (col_lane * 4 + 3) * SF_TILE_BYTES
            cute.recast_ptr(
                sOUT_tile.iterator + smem_offset0, dtype=cutlass.Uint32
            ).store(value)
            if input_col + 4 < group_end:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset1, dtype=cutlass.Uint32
                ).store(value1)
            if input_col + 8 < group_end:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset2, dtype=cutlass.Uint32
                ).store(value2)
            if input_col + 12 < group_end:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset3, dtype=cutlass.Uint32
                ).store(value3)
            if input_col + 4 >= group_end and col_lane * 16 + 4 < CHUNK_WIDTH:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset1, dtype=cutlass.Uint32
                ).store(cutlass.Uint32(0))
            if input_col + 8 >= group_end and col_lane * 16 + 8 < CHUNK_WIDTH:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset2, dtype=cutlass.Uint32
                ).store(cutlass.Uint32(0))
            if input_col + 12 >= group_end and col_lane * 16 + 12 < CHUNK_WIDTH:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset3, dtype=cutlass.Uint32
                ).store(cutlass.Uint32(0))
            if input_col >= group_end:
                cute.recast_ptr(
                    sOUT_tile.iterator + smem_offset0, dtype=cutlass.Uint32
                ).store(cutlass.Uint32(0))

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if tidx == 0 and active != 0:
                group_start_after_padding = self._group_start_after_padding(
                    input_offsets,
                    group,
                )
                out_group_base = group_start_after_padding * padded_rows
                out_base = (
                    out_group_base
                    + row_block_pid * group_col_blocks * SF_TILE_BYTES
                    + col_chunk * SF_TILES_PER_CHUNK * SF_TILE_BYTES
                )
                valid_tiles = group_col_blocks - col_chunk * SF_TILES_PER_CHUNK
                if valid_tiles > SF_TILES_PER_CHUNK:
                    valid_tiles = cutlass.Int32(SF_TILES_PER_CHUNK)
                for tile in range(SF_TILES_PER_CHUNK):
                    if tile < valid_tiles:
                        src = cute.make_tensor(
                            sOUT_tile.iterator + tile * SF_TILE_BYTES,
                            cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                        )
                        dst = cute.make_tensor(
                            output_scales.iterator + out_base + tile * SF_TILE_BYTES,
                            cute.make_layout((SF_TILE_BYTES,), stride=(1,)),
                        )
                        cute.copy(bulk_store_atom, src, dst)
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        @cute.jit
        def __call__(
            self,
            input_scales: cute.Tensor,
            input_offsets: cute.Tensor,
            output_scales: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            padded_rows: cutlass.Int32,
            num_row_blocks: cutlass.Int32,
            max_active_col_chunks: cutlass.Int32,
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
                padded_rows,
                bulk_store_atom,
            ).launch(
                grid=(max_active_col_chunks, num_row_blocks, 1),
                block=(THREADS_PER_BLOCK, 1, 1),
                cluster=(1, 1, 1),
                smem=SharedStorage.size_in_bytes(),
                stream=stream,
            )

    kernel = MXBlockRearrange2dKGroups()

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
        padded_rows=128,
        num_row_blocks=1,
        max_active_col_chunks=1,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )


def _mx_block_rearrange_2d_k_groups_cutedsl_impl(
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
    padded_rows = ceil_div(rows, 128) * 128
    padded_cols = cols + num_groups * 4
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
    compiled = _compile_mx_block_rearrange_2d_k_groups_cutedsl(
        num_groups,
        chunk_width,
        cols % 16 == 0,
    )
    import cuda.bindings.driver as cuda

    stream = cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))
    compiled(
        scales_tensor.view(torch.uint8),
        input_offsets,
        output.view(-1),
        int(rows),
        int(cols),
        int(padded_rows),
        int(ceil_div(rows, 128)),
        int(ceil_div(cols, chunk_width) + num_groups),
        stream,
    )
    return output.view(scales_tensor.dtype)


@torch.library.custom_op(
    "torchao::mx_block_rearrange_2d_k_groups_cutedsl",
    mutates_args=(),
)
def _mx_block_rearrange_2d_k_groups_cutedsl_custom_op(
    scales_tensor: torch.Tensor,
    input_offsets: torch.Tensor,
    chunk_width: int,
) -> torch.Tensor:
    return _mx_block_rearrange_2d_k_groups_cutedsl_impl(
        scales_tensor,
        input_offsets,
        None if chunk_width == 0 else chunk_width,
    )


@_mx_block_rearrange_2d_k_groups_cutedsl_custom_op.register_fake
def _fake_mx_block_rearrange_2d_k_groups_cutedsl_custom_op(
    scales_tensor: torch.Tensor,
    input_offsets: torch.Tensor,
    chunk_width: int,
) -> torch.Tensor:
    assert scales_tensor.ndim == 2
    assert scales_tensor.element_size() == 1
    assert input_offsets.ndim == 1
    rows, cols = scales_tensor.shape
    num_groups = input_offsets.shape[0]
    padded_rows = ceil_div(rows, 128) * 128
    padded_cols = cols + num_groups * 4
    return scales_tensor.new_empty((padded_rows, padded_cols))


def mx_block_rearrange_2d_k_groups_cutedsl(
    scales_tensor: torch.Tensor,
    input_offsets: torch.Tensor,
    chunk_width: int | None = None,
) -> torch.Tensor:
    return _mx_block_rearrange_2d_k_groups_cutedsl_custom_op(
        scales_tensor,
        input_offsets,
        0 if chunk_width is None else chunk_width,
    )
