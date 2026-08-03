# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch


@functools.cache
def _compile_to_sparse_semi_structured_cutedsl(
    float8_dtype: str,
    has_padding: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    THREADS_PER_BLOCK = 256
    float8_element_type = (
        cutlass.Float8E5M2 if float8_dtype == "e5m2" else cutlass.Float8E4M3FN
    )
    HAS_PADDING = has_padding

    class ToSparseSemiStructured:
        @cute.jit
        def _load_nonzero(
            self,
            weight: cute.Tensor,
            offset: cutlass.Int32,
        ):
            value = (
                cute.recast_ptr(weight.iterator + offset, dtype=cutlass.Uint8)
                .load()
                .to(cutlass.Uint8)
            )
            is_nonzero = (value & cutlass.Uint8(0x7F)) != cutlass.Uint8(0)
            return value, is_nonzero

        @cute.jit
        def _encode_chunk(
            self,
            weight: cute.Tensor,
            row_base: cutlass.Int32,
            chunk_col: cutlass.Int32,
            output: cute.Tensor,
            output_base: cutlass.Int32,
        ):
            value0, nz0 = self._load_nonzero(weight, row_base + chunk_col)
            value1, nz1 = self._load_nonzero(weight, row_base + chunk_col + 1)
            value2, nz2 = self._load_nonzero(weight, row_base + chunk_col + 2)
            value3, nz3 = self._load_nonzero(weight, row_base + chunk_col + 3)

            out0 = cutlass.Uint8(0)
            out1 = cutlass.Uint8(0)
            idx0 = cutlass.Uint8(0)
            idx1 = cutlass.Uint8(0)
            count = cutlass.Int32(0)

            if nz0 != 0:
                out0 = value0
                idx0 = cutlass.Uint8(0)
                count = count + 1
            if nz1 != 0:
                if count == cutlass.Int32(0):
                    out0 = value1
                    idx0 = cutlass.Uint8(1)
                else:
                    out1 = value1
                    idx1 = cutlass.Uint8(1)
                count = count + 1
            if nz2 != 0:
                if count == cutlass.Int32(0):
                    out0 = value2
                    idx0 = cutlass.Uint8(2)
                else:
                    out1 = value2
                    idx1 = cutlass.Uint8(2)
                count = count + 1
            if nz3 != 0:
                if count == cutlass.Int32(0):
                    out0 = value3
                    idx0 = cutlass.Uint8(3)
                else:
                    out1 = value3
                    idx1 = cutlass.Uint8(3)
                count = count + 1

            if count == cutlass.Int32(1):
                if idx0 == cutlass.Uint8(3):
                    out1 = out0
                    out0 = cutlass.Uint8(0)
                    idx0 = cutlass.Uint8(0)
                    idx1 = cutlass.Uint8(3)
                else:
                    out1 = cutlass.Uint8(0)
                    idx1 = cutlass.Uint8(3)

            cute.recast_ptr(output.iterator + output_base, dtype=cutlass.Uint8).store(
                out0
            )
            cute.recast_ptr(
                output.iterator + output_base + 1, dtype=cutlass.Uint8
            ).store(out1)

            return idx0 | (idx1 << 2)

        @cute.kernel
        def kernel(
            self,
            weight: cute.Tensor,
            output: cute.Tensor,
            metadata: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            compressed_cols: cutlass.Int32,
            metadata_cols: cutlass.Int32,
            metadata_rows: cutlass.Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block_idx, _, _ = cute.arch.block_idx()
            linear = cutlass.Int32(block_idx * THREADS_PER_BLOCK + tidx)
            logical_metadata_cols = cols // 8
            total_metadata = rows * logical_metadata_cols
            output_padding_cols = compressed_cols - cols // 2
            total_output_padding = rows * output_padding_cols
            total_metadata_col_padding = rows * (metadata_cols - logical_metadata_cols)
            total_metadata_row_padding = (metadata_rows - rows) * metadata_cols

            if cutlass.const_expr(HAS_PADDING):
                if linear < total_output_padding:
                    output_pad_row = linear // output_padding_cols
                    output_pad_col = linear - output_pad_row * output_padding_cols
                    output_offset = (
                        output_pad_row * compressed_cols + cols // 2 + output_pad_col
                    )
                    output_value = cutlass.Uint8(0)
                    cute.recast_ptr(
                        output.iterator + output_offset, dtype=cutlass.Uint8
                    ).store(output_value)
                metadata_col_pad_linear = linear - total_output_padding
                if (
                    metadata_col_pad_linear >= 0
                    and metadata_col_pad_linear < total_metadata_col_padding
                ):
                    metadata_pad_row = metadata_col_pad_linear // (
                        metadata_cols - logical_metadata_cols
                    )
                    metadata_pad_col = (
                        metadata_col_pad_linear
                        - metadata_pad_row * (metadata_cols - logical_metadata_cols)
                        + logical_metadata_cols
                    )
                    metadata_pad_row_tile = metadata_pad_row // 64
                    metadata_pad_row_in_tile = (
                        metadata_pad_row - metadata_pad_row_tile * 64
                    )
                    metadata_pad_col_tile = metadata_pad_col // 16
                    metadata_pad_col_in_tile = (
                        metadata_pad_col - metadata_pad_col_tile * 16
                    )
                    metadata_pad_offset = (
                        metadata_pad_col_tile * metadata_rows * 16
                        + metadata_pad_row_tile * 64 * 16
                        + metadata_pad_row_in_tile * 16
                        + metadata_pad_col_in_tile
                    )
                    metadata_value = cutlass.Uint8(0)
                    cute.recast_ptr(
                        metadata.iterator + metadata_pad_offset, dtype=cutlass.Uint8
                    ).store(metadata_value)
                metadata_row_pad_linear = (
                    linear - total_output_padding - total_metadata_col_padding
                )
                if (
                    metadata_row_pad_linear >= 0
                    and metadata_row_pad_linear < total_metadata_row_padding
                ):
                    metadata_pad_row = metadata_row_pad_linear // metadata_cols + rows
                    metadata_pad_col = (
                        metadata_row_pad_linear
                        - (metadata_pad_row - rows) * metadata_cols
                    )
                    metadata_pad_row_tile = metadata_pad_row // 64
                    metadata_pad_row_in_tile = (
                        metadata_pad_row - metadata_pad_row_tile * 64
                    )
                    metadata_pad_col_tile = metadata_pad_col // 16
                    metadata_pad_col_in_tile = (
                        metadata_pad_col - metadata_pad_col_tile * 16
                    )
                    metadata_pad_offset = (
                        metadata_pad_col_tile * metadata_rows * 16
                        + metadata_pad_row_tile * 64 * 16
                        + metadata_pad_row_in_tile * 16
                        + metadata_pad_col_in_tile
                    )
                    metadata_value = cutlass.Uint8(0)
                    cute.recast_ptr(
                        metadata.iterator + metadata_pad_offset, dtype=cutlass.Uint8
                    ).store(metadata_value)
            if linear < total_metadata:
                row = linear // logical_metadata_cols
                metadata_col = linear - row * logical_metadata_cols
                metadata_value = cutlass.Uint8(0)
                row_tile = row // 64
                row_in_tile = row - row_tile * 64
                col_tile = metadata_col // 16
                col_in_tile = metadata_col - col_tile * 16
                metadata_offset = (
                    col_tile * metadata_rows * 16
                    + row_tile * 64 * 16
                    + row_in_tile * 16
                    + col_in_tile
                )
                if row < rows:
                    row_base = row * cols
                    compressed_row_base = row * compressed_cols
                    chunk_col = metadata_col * 8
                    output_base = compressed_row_base + metadata_col * 4
                    lo = self._encode_chunk(
                        weight,
                        row_base,
                        chunk_col,
                        output,
                        output_base,
                    )
                    hi = self._encode_chunk(
                        weight,
                        row_base,
                        chunk_col + 4,
                        output,
                        output_base + 2,
                    )
                    metadata_value = (lo | (hi << 4)).to(cutlass.Uint8)
                cute.recast_ptr(
                    metadata.iterator + metadata_offset, dtype=cutlass.Uint8
                ).store(metadata_value)

        @cute.jit
        def __call__(
            self,
            weight: cute.Tensor,
            output: cute.Tensor,
            metadata: cute.Tensor,
            rows: cutlass.Int32,
            cols: cutlass.Int32,
            compressed_cols: cutlass.Int32,
            metadata_cols: cutlass.Int32,
            metadata_rows: cutlass.Int32,
            num_blocks: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            grid = (
                num_blocks,
                1,
                1,
            )
            self.kernel(
                weight,
                output,
                metadata,
                rows,
                cols,
                compressed_cols,
                metadata_cols,
                metadata_rows,
            ).launch(
                grid=grid,
                block=[THREADS_PER_BLOCK, 1, 1],
                stream=stream,
            )

    return cute.compile(
        ToSparseSemiStructured(),
        weight=make_fake_tensor(
            float8_element_type,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        output=make_fake_tensor(
            float8_element_type,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        metadata=make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
        ),
        rows=0,
        cols=0,
        compressed_cols=16,
        metadata_cols=16,
        metadata_rows=64,
        num_blocks=1,
        stream=make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def to_sparse_semi_structured_cutedsl(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert weight.dim() == 2, f"Expected weight to be 2D, got {weight.dim()}D"
    assert weight.is_cuda, "Expected weight to be CUDA"
    assert weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
        "Expected FP8 weight"
    )
    assert weight.stride(1) == 1, "Expected weight in row-major layout"
    rows, cols = weight.shape
    assert cols % 8 == 0, "Expected number of columns to be divisible by 8"

    compressed_cols = (((cols + 31) // 32) * 32) // 2
    output = torch.empty(
        (rows, compressed_cols),
        dtype=weight.dtype,
        device=weight.device,
    )
    metadata_rows = ((rows + 63) // 64) * 64
    metadata_cols = (((cols + 127) // 128) * 128) // 8
    metadata = torch.empty(
        (metadata_rows, metadata_cols),
        dtype=torch.uint8,
        device=weight.device,
    )
    float8_dtype = "e5m2" if weight.dtype == torch.float8_e5m2 else "e4m3"
    has_padding = (
        compressed_cols != cols // 2
        or metadata_rows != rows
        or metadata_cols != cols // 8
    )
    compiled = _compile_to_sparse_semi_structured_cutedsl(float8_dtype, has_padding)
    logical_metadata_cols = cols // 8
    total_metadata = rows * logical_metadata_cols
    total_padding = (
        rows * (compressed_cols - cols // 2)
        + rows * (metadata_cols - logical_metadata_cols)
        + (metadata_rows - rows) * metadata_cols
    )
    total_work = max(total_metadata, total_padding) if has_padding else total_metadata
    num_blocks = (total_work + 255) // 256
    compiled(
        weight=weight,
        output=output,
        metadata=metadata,
        rows=rows,
        cols=cols,
        compressed_cols=compressed_cols,
        metadata_cols=metadata_cols,
        metadata_rows=metadata_rows,
        num_blocks=num_blocks,
        stream=torch.cuda.current_stream(),
    )
    return output, metadata
