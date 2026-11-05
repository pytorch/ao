# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from threading import Lock
from typing import Callable, Optional, Tuple

import torch

from torchao.utils import ceil_div, round_up

# The blockwise fp8 GEMM is a Triton kernel. Triton is imported and the kernel is
# built lazily (and cached) on first use so this module imports fine on machines
# without Triton, and the (one-time) Triton import cost is only paid when the
# kernel is actually called.
_gemm_op: Optional[Callable] = None
_gemm_op_lock = Lock()


def _get_blockwise_fp8_gemm_op() -> Optional[Callable]:
    global _gemm_op
    op = _gemm_op
    if op is None:
        with _gemm_op_lock:
            op = _gemm_op
            if op is None:
                op = _build_blockwise_fp8_gemm_op()
                _gemm_op = op
    return op


def _build_blockwise_fp8_gemm_op() -> Optional[Callable]:
    from torch.utils._triton import has_triton

    if not has_triton():
        return None

    import triton
    import triton.language as tl
    from triton import Config

    # Original implementation at https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

    fp8_gemm_configs = [
        Config(
            {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
            num_stages=num_stages,
            num_warps=8,
        )
        for block_m in [16, 32, 64, 128]
        for block_n in [32, 64, 128]
        for num_stages in [3, 4, 5, 6]
    ]

    @triton.autotune(
        configs=fp8_gemm_configs, key=["N", "K", "M_BUCKET", "BLOCK_SIZE_K"]
    )
    @triton.jit
    def blockwise_fp8_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_s_ptr,
        b_s_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        M_BUCKET: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        k = tl.cdiv(K, BLOCK_SIZE_K)
        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
        a_s_ptrs = a_s_ptr + offs_m * k
        b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(k):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
            a_s = tl.load(a_s_ptrs)
            b_s = tl.load(b_s_ptrs)
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
            a_s_ptrs += 1
            b_s_ptrs += 1

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)

    @torch.library.custom_op("ao::blockwise_fp8_gemm", mutates_args=())
    def _blockwise_fp8_gemm_op(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        assert a.is_contiguous()
        assert b.is_contiguous()
        assert a_s.is_contiguous()
        assert b_s.is_contiguous()
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)
        M_BUCKET = math.ceil(math.log2(M))
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        blockwise_fp8_gemm_kernel[grid](
            a, b, c, a_s, b_s, M, N, K, M_BUCKET, BLOCK_SIZE_K=block_size
        )
        return c

    @_blockwise_fp8_gemm_op.register_fake
    def _(a, a_s, b, b_s, block_size=128):
        N = b.size(0)
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
        return c

    return _blockwise_fp8_gemm_op


def _blockwise_fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    op = _get_blockwise_fp8_gemm_op()
    if op is None:
        raise AssertionError("unsupported without triton")
    return op(a, a_s, b, b_s, block_size)


@functools.cache
def _cutlass_dtype(dtype: torch.dtype):
    import cutlass

    return {
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[dtype]


def _current_raw_stream(device_index: int) -> int:
    # The raw cudaStream_t handle of the current stream. The TVM-FFI entry
    # points accept it directly, and it is a few microseconds per call cheaper
    # than building a torch.cuda.Stream object with torch.cuda.current_stream().
    return torch._C._cuda_getCurrentRawStream(device_index)


# FP8 2:4 sparse compression (dense -> compressed data + metadata), replacing
# the legacy to_sparse_semi_structured_cutlass_sm9x_f8. The compressed data and
# metadata layouts are the ones defined by CUTLASS
# (include/cutlass/transform/kernel/sm90_sparse_gemm_compressor.hpp, with the
# metadata layout from Sm90GemmSparseConfig in
# include/cutlass/gemm/collective/builders/sm90_sparse_config.inl), which the
# legacy CUTLASS sparse GEMM consumes; the output is bitwise identical to the
# legacy op, padding included. Deviation from CUTLASS's compressor: one thread
# encodes one 8-column metadata chunk directly from global memory, and the same
# launch zeroes the padding, instead of CUTLASS's tiled shared-memory
# implementation.
_CONVERSION_THREADS = 256


def _sparse_semi_structured_shapes(rows: int, cols: int) -> Tuple[int, int, int]:
    """Returns (compressed_cols, metadata_rows, metadata_cols) for a weight."""
    return round_up(cols, 32) // 2, round_up(rows, 64), round_up(cols, 128) // 8


@functools.cache
def _compile_to_sparse_semi_structured_cutedsl(
    float8_dtype: torch.dtype,
    has_padding: bool,
    wide_offsets: bool,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    THREADS_PER_BLOCK = _CONVERSION_THREADS
    float8_element_type = _cutlass_dtype(float8_dtype)
    HAS_PADDING = has_padding
    # Weight and output element offsets reach 2^31 only for weights of 2 GiB
    # or more; 64-bit offsets cost a few percent, so only those get them.
    OFFSET = cutlass.Int64 if wide_offsets else cutlass.Int32

    class ToSparseSemiStructured:
        @cute.jit
        def _load_nonzero(
            self,
            weight: cute.Tensor,
            offset: OFFSET,
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
            row_base: OFFSET,
            chunk_col: cutlass.Int32,
            output: cute.Tensor,
            output_base: OFFSET,
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

            # `linear` indexes two independent jobs, so the grid is sized as
            # max(total_metadata, total_padding) and one launch does both:
            #
            #   1. zeroing the padding, split into three consecutive ranges --
            #      [0, total_output_padding)                    output cols beyond cols//2
            #      [.., + total_metadata_col_padding)           metadata cols beyond cols//8
            #      [.., + total_metadata_row_padding)           metadata rows beyond `rows`
            #      Each range divides `linear` by its own row width to recover a
            #      (row, col) pair, then adds the base offset of the padded region.
            #
            #   2. writing the real metadata, for `linear < total_metadata`.
            #
            # The two jobs never alias: (1) only touches indices at or past
            # cols//2 (output) and cols//8 or `rows` (metadata), while (2) stays
            # strictly below both, so no ordering between them is needed.
            if cutlass.const_expr(HAS_PADDING):
                if linear < total_output_padding:
                    output_pad_row = linear // output_padding_cols
                    output_pad_col = linear - output_pad_row * output_padding_cols
                    output_offset = (
                        OFFSET(output_pad_row) * compressed_cols
                        + cols // 2
                        + output_pad_col
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
                    row_base = OFFSET(row) * cols
                    compressed_row_base = OFFSET(row) * compressed_cols
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


@torch.library.custom_op("torchao::to_sparse_semi_structured_cutedsl", mutates_args=())
def _to_sparse_semi_structured_cutedsl(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert weight.dim() == 2, f"Expected weight to be 2D, got {weight.dim()}D"
    assert weight.is_cuda, "Expected weight to be CUDA"
    assert weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
        "Expected FP8 weight"
    )
    assert weight.stride(1) == 1, "Expected weight in row-major layout"
    rows, cols = weight.shape
    assert weight.numel() == 0 or weight.stride(0) == cols, (
        "Expected weight rows to be densely packed"
    )
    assert cols % 8 == 0, "Expected number of columns to be divisible by 8"

    compressed_cols, metadata_rows, metadata_cols = _sparse_semi_structured_shapes(
        rows, cols
    )
    output = torch.empty(
        (rows, compressed_cols),
        dtype=weight.dtype,
        device=weight.device,
    )
    metadata = torch.empty(
        (metadata_rows, metadata_cols),
        dtype=torch.uint8,
        device=weight.device,
    )
    has_padding = (
        compressed_cols != cols // 2
        or metadata_rows != rows
        or metadata_cols != cols // 8
    )
    logical_metadata_cols = cols // 8
    total_metadata = rows * logical_metadata_cols
    total_padding = (
        rows * (compressed_cols - cols // 2)
        + rows * (metadata_cols - logical_metadata_cols)
        + (metadata_rows - rows) * metadata_cols
    )
    total_work = max(total_metadata, total_padding) if has_padding else total_metadata
    if total_work == 0:
        # Only an empty weight gets here, and both outputs are empty too; a
        # zero-sized grid would be rejected by the launch.
        return output, metadata
    num_blocks = ceil_div(total_work, _CONVERSION_THREADS)
    compiled = _compile_to_sparse_semi_structured_cutedsl(
        weight.dtype, has_padding, rows * cols >= 2**31
    )
    # The --enable-tvm-ffi runtime is positional-only, so this must match
    # ToSparseSemiStructured.__call__'s parameter order exactly.
    compiled(
        weight,
        output,
        metadata,
        rows,
        cols,
        compressed_cols,
        metadata_cols,
        metadata_rows,
        num_blocks,
        _current_raw_stream(weight.get_device()),
    )
    return output, metadata


@_to_sparse_semi_structured_cutedsl.register_fake
def _(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    compressed_cols, metadata_rows, metadata_cols = _sparse_semi_structured_shapes(
        rows, cols
    )
    return (
        weight.new_empty((rows, compressed_cols)),
        weight.new_empty((metadata_rows, metadata_cols), dtype=torch.uint8),
    )
