# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from threading import Lock
from typing import Callable, NamedTuple, Optional, Tuple

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


# PTX type names of the FP8 dtypes, for the sparse WGMMA opcode.
_FP8_PTX_NAMES = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


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
# sparse GEMM below and the legacy CUTLASS GEMM both consume; the output is
# bitwise identical to the legacy op, padding included. Deviation from
# CUTLASS's compressor: one thread encodes one 8-column metadata chunk directly
# from global memory, and the same launch zeroes the padding, instead of
# CUTLASS's tiled shared-memory implementation.
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


def _sparse_wgmma_opcode(input_dtype: str, weight_dtype: str, n: int) -> str:
    if n not in (32, 64, 128):
        raise AssertionError(f"Unsupported sparse WGMMA N: {n}")
    if input_dtype not in ("e4m3", "e5m2") or weight_dtype not in ("e4m3", "e5m2"):
        raise AssertionError(
            f"Unsupported sparse WGMMA dtypes: {weight_dtype}, {input_dtype}"
        )
    return (
        f"wgmma.mma_async.sp.sync.aligned.m64n{n}k64.f32.{weight_dtype}.{input_dtype}"
    )


# FP8 2:4 sparse GEMM with rowwise scales and optional bias, replacing the
# legacy rowwise_scaled_linear_sparse_cutlass_f8f8. It follows CUTLASS's SM90
# sparse warp-specialized mainloop
# (include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized_fp8.hpp)
# in a cooperative arrangement
# (include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp):
# one producer warpgroup issues TMA loads into a multi-stage mbarrier ring and
# the consumer warpgroups run wgmma.mma_async.sp on slices of the CTA's weight
# rows. The weight is the sparse operand, so weight rows take the WGMMA M mode
# and input rows take N. Deviations from CUTLASS, and from the legacy kernel
# (KernelTmaWarpSpecializedPingpongFP8FastAccum, persistent):
# - cooperative instead of ping-pong, and a non-persistent grid; ping-pong was
#   measured slower for this kernel;
# - the sparse WGMMA is issued as inline PTX with hand-packed shared memory
#   descriptors;
# - metadata goes through shared memory with bulk copies when the CTA's weight
#   rows are covered by the padded metadata, and is read from global memory
#   otherwise, with the row window clamped for ragged N;
# - the CTA tile is 128 or 256 weight rows by 128 input rows, chosen from the
#   problem size by _select_config, with a 1x2 cluster multicasting the weight
#   tile when M is a multiple of 256; for M <= 64 it is legacy's own decode
#   tile instead (_DECODE_CONFIG);
# - the rowwise scales and bias are applied to the accumulators in registers,
#   then the tile is staged through shared memory for a TMA store, instead of
#   CUTLASS's epilogue visitor tree broadcast nodes;
# - tiles are rasterized in groups of _RASTER_GROUP_N weight-row blocks.
_WGMMA_WEIGHT_ROWS = 64
_WGMMA_LOGICAL_K = 64
_WGMMA_COMPRESSED_K = _WGMMA_LOGICAL_K // 2
_WGMMA_THREADS = 128
_WARPS_PER_WARPGROUP = 4
_MMA_PER_WG_CHOICES = (1, 2)
_MMA_PER_WG_DEFAULT = 2
_PRODUCER_REGS = 32
_CONSUMER_REGS = 232
_GMMA_LAYOUT_TYPE_B128 = 1
_GMMA_LAYOUT_TYPE_B64 = 2
_GMMA_LAYOUT_TYPE_B32 = 3
_GMMA_DESC_UNIT_BYTES = 16
_GMMA_DESC_ROWS_PER_GROUP = 8
_GMMA_DESC_LEADING_OFFSET = 1
_A_GMMA_DESC_STRIDE_OFFSET = (
    _GMMA_DESC_ROWS_PER_GROUP * _WGMMA_COMPRESSED_K // _GMMA_DESC_UNIT_BYTES
)
_B_GMMA_DESC_STRIDE_OFFSET = (
    _GMMA_DESC_ROWS_PER_GROUP * _WGMMA_LOGICAL_K // _GMMA_DESC_UNIT_BYTES
)

_META_UINT32_PER_ROW = 4
_RASTER_GROUP_N = 8
_EPI_TILE_N = 64
_EPI_STAGES = 2
_MAX_SMEM_BYTES = 232448


class _LinearConfig(NamedTuple):
    # Input rows of a CTA tile, i.e. the WGMMA N.
    input_rows: int
    consumer_warpgroups: int
    stages: int
    # K tiles of _WGMMA_LOGICAL_K per pipeline stage.
    k_group: int
    # Stop the K loop after the last real group of K tiles, instead of running
    # whole waves of `stages` groups; the extra groups only add zeros, but cost
    # up to stages - 1 groups of work, which matters when stages is large.
    exact_k_loop: bool
    # 128-byte swizzled A and B buffers spanning a whole stage's K, as CUTLASS
    # uses, so one TMA copy per operand fills a stage, instead of one per K tile
    # into 32- and 64-byte swizzled buffers.
    swizzle128: bool


# Two consumer warpgroups splitting 128 or 256 weight rows, on 128 input rows.
_DEFAULT_CONFIG = _LinearConfig(
    input_rows=128,
    consumer_warpgroups=2,
    stages=4,
    k_group=2,
    exact_k_loop=False,
    swizzle128=False,
)
# Legacy's configuration for M <= 64, read from its kernel name in ncu:
# TileShape 64x32x256, 12 stages, no cluster. That is one consumer warpgroup
# on 64 weight rows and 32 input rows, with K step 256, and the deep pipeline
# is what keeps enough weight bytes in flight when there are few CTAs.
_DECODE_CONFIG = _LinearConfig(
    input_rows=32,
    consumer_warpgroups=1,
    stages=12,
    k_group=4,
    exact_k_loop=True,
    swizzle128=True,
)
_DECODE_MAX_M = 64


def _cta_weight_rows(mma_per_wg: int, config: _LinearConfig) -> int:
    return _WGMMA_WEIGHT_ROWS * mma_per_wg * config.consumer_warpgroups


def _select_config(m: int, n: int, num_sms: int) -> Tuple[_LinearConfig, int]:
    """Returns the configuration and the MMAs per consumer warpgroup."""
    if m <= _DECODE_MAX_M:
        # Its shared memory allows one CTA per SM, and with a non-persistent
        # grid every further wave refills the 12-stage pipeline from cold.
        # Measured on H100 against the default configuration: with one block
        # of input rows (M <= 32) it was faster at every shape tried, up to 6.8
        # waves; with two, where each weight tile is read twice, up to 2.4
        # waves, and mostly slower from 3.4.
        decode_ctas = ceil_div(n, _cta_weight_rows(1, _DECODE_CONFIG)) * ceil_div(
            m, _DECODE_CONFIG.input_rows
        )
        if m <= _DECODE_CONFIG.input_rows or decode_ctas <= 3 * num_sms:
            return _DECODE_CONFIG, 1
    config = _DEFAULT_CONFIG
    wide = _cta_weight_rows(_MMA_PER_WG_DEFAULT, config)
    ctas = ceil_div(n, wide) * ceil_div(m, config.input_rows)
    limit = num_sms // 4 if m <= config.input_rows // 2 else num_sms // 2
    return config, (1 if ctas <= limit else _MMA_PER_WG_DEFAULT)


@functools.cache
def _compile_rowwise_scaled_linear_sparse_cutedsl(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    scale_dtype: torch.dtype,
    output_dtype: torch.dtype,
    has_bias: bool,
    m_exact: bool,
    n_exact: bool,
    cluster_mcast: bool,
    meta_block_safe: bool,
    mma_per_wg: int,
    config: _LinearConfig,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.cutlass_dsl import T, dsl_user_op

    if scale_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise AssertionError(f"Unsupported scale dtype: {scale_dtype}")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise AssertionError(f"Unsupported output dtype: {output_dtype}")
    if mma_per_wg not in _MMA_PER_WG_CHOICES:
        raise AssertionError(f"Unsupported MMAs per warpgroup: {mma_per_wg}")

    _WGMMA_INPUT_ROWS = config.input_rows
    _CONSUMER_WARPGROUPS = config.consumer_warpgroups
    _STAGES = config.stages
    _K_GROUP = config.k_group
    _WGMMA_ACC_REGS = _WGMMA_WEIGHT_ROWS * _WGMMA_INPUT_ROWS // _WGMMA_THREADS
    _CTA_THREADS = _WGMMA_THREADS * (_CONSUMER_WARPGROUPS + 1)
    _PRODUCER_WARP = _CONSUMER_WARPGROUPS * _WARPS_PER_WARPGROUP
    _BUFFERS = _STAGES * _K_GROUP
    # K tiles per A or B shared-memory buffer.
    _K_TILES_PER_BUF = _K_GROUP if config.swizzle128 else 1
    _BUFS_PER_STAGE = _K_GROUP // _K_TILES_PER_BUF
    _N_BUFS = _STAGES * _BUFS_PER_STAGE
    _A_BUF_K = _WGMMA_COMPRESSED_K * _K_TILES_PER_BUF
    _B_BUF_K = _WGMMA_LOGICAL_K * _K_TILES_PER_BUF
    _B_STAGE_BYTES = _WGMMA_INPUT_ROWS * _WGMMA_LOGICAL_K
    _META_HALVES_PER_GROUP = max(1, _K_GROUP // 2)
    _META_SMEM_STAGED = _K_GROUP % 2 == 0
    _EPI_TILE_M = min(64, _WGMMA_INPUT_ROWS)
    _EPI_M_SUBTILES = _WGMMA_INPUT_ROWS // _EPI_TILE_M
    _EPI_SLOTS = _CONSUMER_WARPGROUPS * _EPI_STAGES
    _EPI_VID2_PER_SUBTILE = _WGMMA_ACC_REGS // 4 // _EPI_M_SUBTILES
    _EXACT_K_LOOP = config.exact_k_loop

    _MMA_PER_WG = mma_per_wg
    _WG_WEIGHT_ROWS = _WGMMA_WEIGHT_ROWS * _MMA_PER_WG
    _CTA_WEIGHT_ROWS = _WG_WEIGHT_ROWS * _CONSUMER_WARPGROUPS
    _TOTAL_ACC_REGS = _WGMMA_ACC_REGS * _MMA_PER_WG
    _A_STAGE_BYTES = _CTA_WEIGHT_ROWS * _WGMMA_COMPRESSED_K
    _A_BUF_DESC_UNITS = _CTA_WEIGHT_ROWS * _A_BUF_K // _GMMA_DESC_UNIT_BYTES
    _B_BUF_DESC_UNITS = _WGMMA_INPUT_ROWS * _B_BUF_K // _GMMA_DESC_UNIT_BYTES
    if config.swizzle128:
        if _A_BUF_K > 128:
            raise AssertionError("An A buffer must fit one 128-byte swizzle atom")
        _A_LAYOUT_TYPE = _GMMA_LAYOUT_TYPE_B128
        _B_LAYOUT_TYPE = _GMMA_LAYOUT_TYPE_B128
        _A_DESC_STRIDE = _GMMA_DESC_ROWS_PER_GROUP * 128 // _GMMA_DESC_UNIT_BYTES
        _B_DESC_STRIDE = _A_DESC_STRIDE
    else:
        _A_LAYOUT_TYPE = _GMMA_LAYOUT_TYPE_B32
        _B_LAYOUT_TYPE = _GMMA_LAYOUT_TYPE_B64
        _A_DESC_STRIDE = _A_GMMA_DESC_STRIDE_OFFSET
        _B_DESC_STRIDE = _B_GMMA_DESC_STRIDE_OFFSET

    def _a_desc_units(k_tile):
        # An A buffer holds _K_TILES_PER_BUF K tiles side by side in each row.
        buf, i = divmod(k_tile, _K_TILES_PER_BUF)
        return (
            buf * _A_BUF_DESC_UNITS + i * _WGMMA_COMPRESSED_K // _GMMA_DESC_UNIT_BYTES
        )

    def _b_desc_units(k_tile):
        # Past 128 bytes of K, a B buffer continues in the next swizzle atom,
        # which follows all of the buffer's rows.
        buf, i = divmod(k_tile, _K_TILES_PER_BUF)
        atom, inner = divmod(i * _WGMMA_LOGICAL_K, 128)
        return (
            buf * _B_BUF_DESC_UNITS
            + (atom * _WGMMA_INPUT_ROWS * 128 + inner) // _GMMA_DESC_UNIT_BYTES
        )

    _TMA_STAGE_BYTES = _A_STAGE_BYTES + _B_STAGE_BYTES
    _TMA_GROUP_BYTES = _K_GROUP * _TMA_STAGE_BYTES
    _META_STAGE_UINT32 = _CTA_WEIGHT_ROWS * _META_UINT32_PER_ROW
    _META_STAGE_BYTES = _META_STAGE_UINT32 * 4
    _META_GROUP_BYTES = _META_HALVES_PER_GROUP * _META_STAGE_BYTES
    _META_SMEM_UINT32 = _STAGES * _META_HALVES_PER_GROUP * _META_STAGE_UINT32
    _EPI_N_SUBTILES = _WG_WEIGHT_ROWS // _EPI_TILE_N

    opcode = _sparse_wgmma_opcode(
        _FP8_PTX_NAMES[input_dtype], _FP8_PTX_NAMES[weight_dtype], _WGMMA_INPUT_ROWS
    )
    _use_a_mcast = cluster_mcast
    _meta_staged = _META_SMEM_STAGED and meta_block_safe
    if _EXACT_K_LOOP and not _meta_staged:
        # The global-memory metadata path prefetches one wave ahead, which
        # assumes whole waves.
        raise AssertionError("The exact K loop needs staged metadata")

    INPUT_DTYPE = _cutlass_dtype(input_dtype)
    WEIGHT_DTYPE = _cutlass_dtype(weight_dtype)
    SCALE_DTYPE = _cutlass_dtype(scale_dtype)
    OUTPUT_DTYPE = _cutlass_dtype(output_dtype)

    @dsl_user_op
    def _pack_gmma_smem_desc_k(
        ptr: cute.Pointer,
        byte_offset,
        leading: int,
        stride: int,
        layout_type: int,
        *,
        loc=None,
        ip=None,
    ):
        address = ptr.toint(loc=loc, ip=ip).to(cutlass.Uint64) + cutlass.Uint64(
            byte_offset
        ).to(cutlass.Uint64)
        start = (address >> cutlass.Uint64(4)) & cutlass.Uint64(0x3FFF)
        leading_bits = cutlass.Uint64(leading & 0x3FFF) << cutlass.Uint64(16)
        stride_bits = cutlass.Uint64(stride & 0x3FFF) << cutlass.Uint64(32)
        layout_bits = cutlass.Uint64(layout_type) << cutlass.Uint64(62)
        return (start | leading_bits | stride_bits | layout_bits).to(cutlass.Int64)

    @dsl_user_op
    def _wgmma_sp(
        desc_a,
        desc_b,
        meta: cutlass.Uint32,
        acc,
        *,
        loc=None,
        ip=None,
    ):
        regs = ", ".join(f"${i}" for i in range(_WGMMA_ACC_REGS))
        desc_a_arg = 2 * _WGMMA_ACC_REGS
        asm = (
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            f"{opcode} "
            "{" + regs + "}, "
            f"${desc_a_arg}, ${desc_a_arg + 1}, ${desc_a_arg + 2}, 0, p, 1, 1;\n"
            "}\n"
        )
        constraints = ",".join(
            ["=f"] * _WGMMA_ACC_REGS
            + [str(i) for i in range(_WGMMA_ACC_REGS)]
            + ["l", "l", "r"]
        )
        result = llvm.inline_asm(
            ir.Type.parse(
                "!llvm.struct<(" + ",".join(["f32"] * _WGMMA_ACC_REGS) + ")>"
            ),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip) for a in acc]
            + [
                cutlass.Int64(desc_a).ir_value(loc=loc, ip=ip),
                cutlass.Int64(desc_b).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(meta).ir_value(loc=loc, ip=ip),
            ],
            asm,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return tuple(
            cutlass.Float32(llvm.extractvalue(T.f32(), result, [i]))
            for i in range(_WGMMA_ACC_REGS)
        )

    def _acc_only_asm(body, n_acc):
        @dsl_user_op
        def op(acc, *, loc=None, ip=None):
            constraints = ",".join(["=f"] * n_acc + [str(i) for i in range(n_acc)])
            result = llvm.inline_asm(
                ir.Type.parse("!llvm.struct<(" + ",".join(["f32"] * n_acc) + ")>"),
                [cutlass.Float32(a).ir_value(loc=loc, ip=ip) for a in acc],
                "{\n" + body + "}\n",
                constraints,
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
            return tuple(
                cutlass.Float32(llvm.extractvalue(T.f32(), result, [i]))
                for i in range(n_acc)
            )

        return op

    @dsl_user_op
    def _opaque_i64(value, *, loc=None, ip=None):
        # A copy the compiler cannot see through, so values derived from it
        # inside a loop are not hoisted out of it.
        return cutlass.Int64(
            llvm.inline_asm(
                T.i64(),
                [cutlass.Int64(value).ir_value(loc=loc, ip=ip)],
                "mov.b64 $0, $1;",
                "=l,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    _wgmma_fence = _acc_only_asm("wgmma.fence.sync.aligned;\n", _TOTAL_ACC_REGS)
    _wgmma_commit_wait = _acc_only_asm(
        "wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 1;\n",
        _TOTAL_ACC_REGS,
    )
    _wgmma_drain = _acc_only_asm(
        "wgmma.wait_group.sync.aligned 0;\n",
        _TOTAL_ACC_REGS,
    )

    @cute.struct
    class SharedStorage:
        sa: cute.struct.Align[
            cute.struct.MemRange[WEIGHT_DTYPE, _BUFFERS * _A_STAGE_BYTES],
            1024,
        ]
        sb: cute.struct.Align[
            cute.struct.MemRange[INPUT_DTYPE, _BUFFERS * _B_STAGE_BYTES],
            1024,
        ]
        smeta: cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint32, _META_SMEM_UINT32],
            128,
        ]
        sepi: cute.struct.Align[
            cute.struct.MemRange[OUTPUT_DTYPE, _EPI_SLOTS * _EPI_TILE_M * _EPI_TILE_N],
            1024,
        ]
        mbar: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, _STAGES],
            8,
        ]
        mbar_empty: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int64, _STAGES],
            8,
        ]

    if SharedStorage.size_in_bytes() > _MAX_SMEM_BYTES:
        raise AssertionError(
            f"{config} needs {SharedStorage.size_in_bytes()} bytes of shared memory"
        )

    def _make_smem_layouts():
        kinds = cute.nvgpu.warpgroup.SmemLayoutAtomKind
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            kinds.K_SW128 if config.swizzle128 else kinds.K_SW32,
            WEIGHT_DTYPE,
        )
        b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            kinds.K_SW128 if config.swizzle128 else kinds.K_SW64,
            INPUT_DTYPE,
        )
        a_staged = cute.tile_to_shape(
            a_atom,
            (_CTA_WEIGHT_ROWS, _A_BUF_K, _N_BUFS),
            order=(0, 1, 2),
        )
        b_staged = cute.tile_to_shape(
            b_atom,
            (_WGMMA_INPUT_ROWS, _B_BUF_K, _N_BUFS),
            order=(0, 1, 2),
        )
        return a_staged, b_staged

    def _make_epi_smem_layout():
        c_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            OUTPUT_DTYPE,
        )
        return cute.tile_to_shape(
            c_atom,
            (_EPI_TILE_M, _EPI_TILE_N),
            order=(0, 1),
        )

    class RowwiseScaledLinearSparseWgmma:
        def __init__(self):
            self.shared_storage = SharedStorage

        @cute.kernel
        def kernel(
            self,
            tma_atom_a: cute.CopyAtom,
            tma_tensor_a: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            tma_tensor_b: cute.Tensor,
            tma_atom_c: cute.CopyAtom,
            tma_tensor_c: cute.Tensor,
            input_scale: cute.Tensor,
            weight_meta: cute.Tensor,
            weight_scale: cute.Tensor,
            bias: cute.Tensor,
            output: cute.Tensor,
            m_size: cutlass.Int32,
            n_size: cutlass.Int32,
            k_size: cutlass.Int32,
            metadata_rows: cutlass.Int32,
            raster_group_n: cutlass.Int32,
        ):
            import cutlass.utils as utils

            tidx, _, _ = cute.arch.thread_idx()
            raw_x, raw_y, _ = cute.arch.block_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            warpgroup = tidx // _WGMMA_THREADS
            tid_wg = tidx - warpgroup * _WGMMA_THREADS
            a_mcast_mask = cutlass.Int16(0)
            a_tma_crd = cutlass.Int32(0)
            a_tma_layout = cute.make_layout(1)
            cta_rank = cutlass.Int32(0)
            peer_cta_rank = cutlass.Int32(0)
            if cutlass.const_expr(cluster_mcast):
                cta_layout_mnk = cute.make_layout((1, 2, 1))
                cta_layout_vmnk = cute.make_layout((1, 1, 2, 1))
                cidx, cidy, _ = cute.arch.cluster_idx()
                raw_x = cidx
                raw_y = cidy
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank)
                cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank)
                peer_cta_rank = cta_rank ^ 1
                if cutlass.const_expr(_use_a_mcast):
                    a_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask(
                        cta_layout_vmnk, cluster_coord_vmnk, 2
                    )
                    a_tma_layout = cute.make_layout(
                        cute.slice_(cta_layout_mnk, (0, None, 0)).shape
                    )
                    a_tma_crd = cluster_coord_mnk[1]

            num_n_blocks = ceil_div(n_size, _CTA_WEIGHT_ROWS)
            num_m_blocks = ceil_div(m_size, _WGMMA_INPUT_ROWS)
            if cutlass.const_expr(cluster_mcast):
                num_m_units = num_m_blocks // 2
            else:
                num_m_units = num_m_blocks
            linear_tile = raw_y * num_n_blocks + raw_x
            tiles_per_group = raster_group_n * num_m_units
            raster_group = linear_tile // tiles_per_group
            first_n = raster_group * raster_group_n
            group_n = num_n_blocks - first_n
            if group_n > raster_group_n:
                group_n = raster_group_n
            tile_in_group = linear_tile - raster_group * tiles_per_group
            n_block = first_n + tile_in_group % group_n
            m_unit = tile_in_group // group_n
            if cutlass.const_expr(cluster_mcast):
                m_block = m_unit * 2 + cta_rank
            else:
                m_block = m_unit

            n_base = n_block * _CTA_WEIGHT_ROWS
            smem_allocator = utils.SmemAllocator()
            storage = smem_allocator.allocate(self.shared_storage)
            a_staged, b_staged = _make_smem_layouts()
            sa = storage.sa.get_tensor(a_staged)
            sb = storage.sb.get_tensor(b_staged)
            smeta = storage.smeta.get_tensor(cute.make_layout(_META_SMEM_UINT32))
            epi_layout = _make_epi_smem_layout()
            sepi_ptr = storage.sepi.data_ptr()
            mbar = storage.mbar.data_ptr()
            mbar_empty = storage.mbar_empty.data_ptr()
            bulk_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyBulkG2SOp(),
                cutlass.Uint32,
            )

            if warp_idx == 0:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(_STAGES):
                        cute.arch.mbarrier_init(mbar + stage, 1)
                        empty_arrivals = _CONSUMER_WARPGROUPS
                        if cutlass.const_expr(cluster_mcast):
                            empty_arrivals = _CONSUMER_WARPGROUPS * 2
                        cute.arch.mbarrier_init(mbar_empty + stage, empty_arrivals)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(cluster_mcast):
                cute.arch.cluster_arrive_relaxed()

            meta_stage_base = n_base
            if cutlass.const_expr(_meta_staged and not n_exact):
                meta_stage_base = cutlass.min(n_base, metadata_rows - _CTA_WEIGHT_ROWS)
            meta_shift = n_base - meta_stage_base

            k_tiles = ceil_div(k_size, _WGMMA_LOGICAL_K)
            meta_last_half = (k_tiles - 1) // 2
            num_groups = ceil_div(k_tiles, _K_GROUP)
            num_waves = ceil_div(num_groups, _STAGES)
            # One past the last group the producer loads and the consumers run.
            group_limit = num_waves * _STAGES
            if cutlass.const_expr(_EXACT_K_LOOP):
                group_limit = num_groups
            meta_row_stride = metadata_rows * 4
            meta_u32 = cute.recast_ptr(weight_meta.iterator, dtype=cutlass.Uint32)

            def issue_group(
                group,
                stage,
                stage_const,
                bar,
                atom_a,
                ga,
                sa_,
                atom_b,
                gb,
                sb_,
                atom_m,
                msrc,
                mstride,
                smeta_,
                last_half,
                staged,
                a_mask,
                use_a_mcast,
                mma_per_wg,
                tma_group_bytes,
                meta_stage_uint32,
                meta_group_bytes,
                bufs_per_stage,
                meta_halves,
            ):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar + stage,
                        tma_group_bytes + (meta_group_bytes if staged else 0),
                    )
                for sub in cutlass.range_constexpr(bufs_per_stage):
                    buf = stage * bufs_per_stage + sub
                    kt = group * bufs_per_stage + sub
                    a_dst = sa_[stage_const * bufs_per_stage + sub]
                    if cutlass.const_expr(use_a_mcast):
                        cute.copy(
                            atom_a.with_(
                                mcast_mask=a_mask,
                                tma_bar_ptr=bar + stage,
                            ),
                            ga[None, kt],
                            a_dst,
                        )
                    else:
                        cute.copy(
                            atom_a,
                            ga[None, kt],
                            a_dst,
                            tma_bar_ptr=bar + stage,
                        )
                    cute.copy(
                        atom_b,
                        gb[None, kt],
                        sb_[None, buf],
                        tma_bar_ptr=bar + stage,
                    )
                if cutlass.const_expr(staged):
                    # Bulk-copy atoms require the caller to elect one issuing thread.
                    with cute.arch.elect_one():
                        for i in cutlass.range_constexpr(meta_halves):
                            half = group * meta_halves + i
                            if half > last_half:
                                half = last_half
                            slot = stage * meta_halves + i
                            cute.copy(
                                atom_m,
                                cute.make_tensor(
                                    msrc + half * mstride,
                                    cute.make_layout(meta_stage_uint32),
                                ),
                                cute.make_tensor(
                                    smeta_.iterator + slot * meta_stage_uint32,
                                    cute.make_layout(meta_stage_uint32),
                                ),
                                mbar_ptr=bar + stage,
                            )

            def group_metas_smem(
                stage,
                smeta_,
                lanes,
                mma_per_wg,
                meta_stage_uint32,
                k_group,
                meta_halves,
            ):
                out = ()
                for sub in cutlass.range_constexpr(k_group):
                    slot = stage * meta_halves + sub // 2
                    for j in cutlass.range_constexpr(mma_per_wg):
                        out = out + (
                            smeta_[slot * meta_stage_uint32 + lanes[j] + 2 * (sub % 2)],
                        )
                return out

            def group_metas(group, bases, row_strides, k_tiles_, mma_per_wg, k_group):
                out = ()
                for sub in cutlass.range_constexpr(k_group):
                    kt = group * k_group + sub
                    if kt >= k_tiles_:
                        kt = k_tiles_ - 1
                    half = kt // 2
                    for j in cutlass.range_constexpr(mma_per_wg):
                        out = out + (
                            (
                                bases[j] + (half * row_strides[j] + 2 * (kt - half * 2))
                            ).load(),
                        )
                return out

            if cutlass.const_expr(cluster_mcast):
                cute.arch.cluster_wait()
            else:
                cute.arch.sync_threads()

            if warpgroup == _CONSUMER_WARPGROUPS:
                # With 256 threads every warp already gets 255 registers, and
                # setmaxnreg.inc to _CONSUMER_REGS would be a decrease, which
                # setmaxnreg does not allow; so only rebalance with 384.
                if cutlass.const_expr(_CONSUMER_WARPGROUPS > 1):
                    cute.arch.setmaxregister_decrease(_PRODUCER_REGS)
                if warp_idx == _PRODUCER_WARP:
                    ga = cute.local_tile(
                        tma_tensor_a,
                        (_CTA_WEIGHT_ROWS, _A_BUF_K),
                        (n_block, None),
                    )
                    gb = cute.local_tile(
                        tma_tensor_b,
                        (_WGMMA_INPUT_ROWS, _B_BUF_K),
                        (m_block, None),
                    )
                    _, tma_ga = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_a,
                        a_tma_crd,
                        a_tma_layout,
                        cute.group_modes(sa, 0, 2),
                        cute.group_modes(ga, 0, 2),
                    )
                    tma_sa = ()
                    for b in cutlass.range_constexpr(_N_BUFS):
                        sa_buf, _ = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_a,
                            a_tma_crd,
                            a_tma_layout,
                            cute.group_modes(cute.slice_(sa, (None, None, b)), 0, 2),
                            cute.group_modes(ga, 0, 2),
                        )
                        tma_sa = tma_sa + (sa_buf,)
                    tma_sb, tma_gb = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_b,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sb, 0, 2),
                        cute.group_modes(gb, 0, 2),
                    )
                    meta_src = meta_u32 + meta_stage_base * _META_UINT32_PER_ROW
                    for st in cutlass.range_constexpr(_STAGES):
                        prologue_issue = True
                        if cutlass.const_expr(_EXACT_K_LOOP):
                            # A short K can have fewer groups than stages.
                            prologue_issue = cutlass.Int32(st) < group_limit
                        if prologue_issue:
                            issue_group(
                                cutlass.Int32(st),
                                cutlass.Int32(st),
                                st,
                                mbar,
                                tma_atom_a,
                                tma_ga,
                                tma_sa,
                                tma_atom_b,
                                tma_gb,
                                tma_sb,
                                bulk_atom,
                                meta_src,
                                meta_row_stride,
                                smeta,
                                meta_last_half,
                                _meta_staged,
                                a_mcast_mask,
                                _use_a_mcast,
                                _MMA_PER_WG,
                                _TMA_GROUP_BYTES,
                                _META_STAGE_UINT32,
                                _META_GROUP_BYTES,
                                _BUFS_PER_STAGE,
                                _META_HALVES_PER_GROUP,
                            )
                    wave = cutlass.Int32(0)
                    while wave < num_waves:
                        phase = wave & 1
                        base_group = wave * _STAGES
                        for st in cutlass.range_constexpr(_STAGES):
                            prefetch_group = base_group + st + _STAGES
                            if prefetch_group < group_limit:
                                cute.arch.mbarrier_wait(mbar_empty + st, phase)
                                issue_group(
                                    prefetch_group,
                                    cutlass.Int32(st),
                                    st,
                                    mbar,
                                    tma_atom_a,
                                    tma_ga,
                                    tma_sa,
                                    tma_atom_b,
                                    tma_gb,
                                    tma_sb,
                                    bulk_atom,
                                    meta_src,
                                    meta_row_stride,
                                    smeta,
                                    meta_last_half,
                                    _meta_staged,
                                    a_mcast_mask,
                                    _use_a_mcast,
                                    _MMA_PER_WG,
                                    _TMA_GROUP_BYTES,
                                    _META_STAGE_UINT32,
                                    _META_GROUP_BYTES,
                                    _BUFS_PER_STAGE,
                                    _META_HALVES_PER_GROUP,
                                )
                        wave = wave + 1
            else:
                if cutlass.const_expr(_CONSUMER_WARPGROUPS > 1):
                    cute.arch.setmaxregister_increase(_CONSUMER_REGS)
                wg_base = n_base + warpgroup * _WG_WEIGHT_ROWS
                m_base = m_block * _WGMMA_INPUT_ROWS
                desc_a_bases = tuple(
                    _pack_gmma_smem_desc_k(
                        sa.iterator,
                        (warpgroup * _MMA_PER_WG + j) * _WGMMA_WEIGHT_ROWS * _A_BUF_K,
                        _GMMA_DESC_LEADING_OFFSET,
                        _A_DESC_STRIDE,
                        _A_LAYOUT_TYPE,
                    )
                    for j in range(_MMA_PER_WG)
                )
                desc_b_base = _pack_gmma_smem_desc_k(
                    sb.iterator,
                    0,
                    _GMMA_DESC_LEADING_OFFSET,
                    _B_DESC_STRIDE,
                    _B_LAYOUT_TYPE,
                )
                meta_lane_row = (
                    (tid_wg // 4) % 8 + 8 * (tid_wg % 2) + 16 * (tid_wg // 32)
                )
                meta_k_half = (tid_wg // 2) % 2
                if cutlass.const_expr(_meta_staged):
                    meta_smem_lanes = ()
                    for j in cutlass.range_constexpr(_MMA_PER_WG):
                        slot_row = (
                            (warpgroup * _MMA_PER_WG + j) * _WGMMA_WEIGHT_ROWS
                            + meta_lane_row
                            + meta_shift
                        )
                        if cutlass.const_expr(not n_exact):
                            if slot_row >= _CTA_WEIGHT_ROWS:
                                slot_row = _CTA_WEIGHT_ROWS - 1
                        meta_smem_lanes = meta_smem_lanes + (
                            slot_row * _META_UINT32_PER_ROW + meta_k_half,
                        )
                else:
                    meta_ptrs = ()
                    meta_strides = ()
                    for j in cutlass.range_constexpr(_MMA_PER_WG):
                        meta_row = wg_base + j * _WGMMA_WEIGHT_ROWS + meta_lane_row
                        meta_row_tile = meta_row // 64
                        meta_row_in_tile = meta_row - meta_row_tile * 64
                        meta_base = (
                            meta_row_tile * 256 + meta_row_in_tile * 4 + meta_k_half
                        )
                        meta_stride = meta_row_stride
                        if cutlass.const_expr(not n_exact):
                            if meta_row >= metadata_rows:
                                meta_base = cutlass.Int32(0)
                                meta_stride = cutlass.Int32(0)
                        meta_ptrs = meta_ptrs + (meta_u32 + meta_base,)
                        meta_strides = meta_strides + (meta_stride,)

                acc_col = m_base + 2 * (tid_wg % 4)
                w_scaled = ()
                w_bias = ()
                for j in cutlass.range_constexpr(_MMA_PER_WG):
                    acc_row = (
                        wg_base
                        + j * _WGMMA_WEIGHT_ROWS
                        + 16 * (tid_wg // 32)
                        + (tid_wg // 4) % 8
                    )
                    for vid1 in cutlass.range_constexpr(2):
                        weight_row = acc_row + 8 * vid1
                        row_safe = weight_row
                        if cutlass.const_expr(not n_exact):
                            if weight_row >= n_size:
                                row_safe = n_size - 1
                        w_scaled = w_scaled + (
                            (weight_scale.iterator + row_safe)
                            .load()
                            .to(cutlass.Float32),
                        )
                        if cutlass.const_expr(has_bias):
                            w_bias = w_bias + (
                                (bias.iterator + row_safe).load().to(cutlass.Float32),
                            )
                in_scaled = ()
                for vid2 in cutlass.range_constexpr(_WGMMA_ACC_REGS // 4):
                    for vid0 in cutlass.range_constexpr(2):
                        input_row = acc_col + vid0 + 8 * vid2
                        col_safe = input_row
                        if cutlass.const_expr(not m_exact):
                            if input_row >= m_size:
                                col_safe = m_size - 1
                        in_scaled = in_scaled + (
                            (input_scale.iterator + col_safe)
                            .load()
                            .to(cutlass.Float32),
                        )
                acc = tuple(cutlass.Float32(0.0) for _ in range(_TOTAL_ACC_REGS))
                metas = ()
                if cutlass.const_expr(not _meta_staged):
                    for st in cutlass.range_constexpr(_STAGES):
                        metas = metas + group_metas(
                            cutlass.Int32(st),
                            meta_ptrs,
                            meta_strides,
                            k_tiles,
                            _MMA_PER_WG,
                            _K_GROUP,
                        )
                wave = cutlass.Int32(0)
                while wave < num_waves:
                    phase = wave & 1
                    base_group = wave * _STAGES
                    next_metas = ()
                    wave_desc_a = desc_a_bases
                    wave_desc_b = desc_b_base
                    if cutlass.const_expr(_BUFFERS > 16):
                        # Each buffer's descriptor is a constant offset from the
                        # base. Hoisted out of the loop, that is two 64-bit
                        # registers per buffer, more than there are with many
                        # buffers, so derive them per wave from opaque copies.
                        wave_desc_a = tuple(_opaque_i64(d) for d in desc_a_bases)
                        wave_desc_b = _opaque_i64(desc_b_base)
                    n_meta = _K_GROUP * _MMA_PER_WG
                    for st in cutlass.range_constexpr(_STAGES):
                        group = base_group + st
                        if cutlass.const_expr(not _meta_staged):
                            prefetched_meta = metas[st * n_meta : (st + 1) * n_meta]
                            next_metas = next_metas + group_metas(
                                group + _STAGES,
                                meta_ptrs,
                                meta_strides,
                                k_tiles,
                                _MMA_PER_WG,
                                _K_GROUP,
                            )
                        consume = True
                        if cutlass.const_expr(_EXACT_K_LOOP):
                            consume = group < group_limit
                        if consume:
                            cute.arch.mbarrier_wait(mbar + st, phase)
                            cute.arch.fence_view_async_shared()
                            if cutlass.const_expr(_meta_staged):
                                cur_meta = group_metas_smem(
                                    st,
                                    smeta,
                                    meta_smem_lanes,
                                    _MMA_PER_WG,
                                    _META_STAGE_UINT32,
                                    _K_GROUP,
                                    _META_HALVES_PER_GROUP,
                                )
                            else:
                                cur_meta = prefetched_meta
                            acc = _wgmma_fence(acc)
                            for sub in cutlass.range_constexpr(_K_GROUP):
                                buf = st * _K_GROUP + sub
                                for j in cutlass.range_constexpr(_MMA_PER_WG):
                                    lo = j * _WGMMA_ACC_REGS
                                    hi = lo + _WGMMA_ACC_REGS
                                    part = _wgmma_sp(
                                        wave_desc_a[j] + _a_desc_units(buf),
                                        wave_desc_b + _b_desc_units(buf),
                                        cur_meta[sub * _MMA_PER_WG + j],
                                        acc[lo:hi],
                                    )
                                    acc = acc[:lo] + part + acc[hi:]
                            acc = _wgmma_commit_wait(acc)
                            prev_st = (st - 1) % _STAGES
                            released = cutlass.const_expr(st > 0) or wave > 0
                            if tid_wg == 0:
                                if released:
                                    cute.arch.mbarrier_arrive(mbar_empty + prev_st)
                                    if cutlass.const_expr(cluster_mcast):
                                        cute.arch.mbarrier_arrive(
                                            mbar_empty + prev_st,
                                            peer_cta_rank_in_cluster=peer_cta_rank,
                                        )
                    if cutlass.const_expr(not _meta_staged):
                        metas = next_metas
                    wave = wave + 1
                acc = _wgmma_drain(acc)

                epi_row = 2 * (tid_wg % 4)
                epi_col = 16 * (tid_wg // 32) + (tid_wg // 4) % 8
                epi_bar = 1 + warpgroup
                epi_warp = tid_wg // 32
                for s_i in cutlass.range_constexpr(_EPI_M_SUBTILES):
                    for t in cutlass.range_constexpr(_EPI_N_SUBTILES):
                        slot = warpgroup * _EPI_STAGES + (
                            (s_i * _EPI_N_SUBTILES + t) % _EPI_STAGES
                        )
                        sc = cute.make_tensor(
                            sepi_ptr + slot * (_EPI_TILE_M * _EPI_TILE_N),
                            epi_layout,
                        )
                        for v in cutlass.range_constexpr(_EPI_VID2_PER_SUBTILE):
                            vid2 = s_i * _EPI_VID2_PER_SUBTILE + v
                            for vid0 in cutlass.range_constexpr(2):
                                in_s = in_scaled[vid2 * 2 + vid0]
                                for vid1 in cutlass.range_constexpr(2):
                                    idx = t * 2 + vid1
                                    vid = vid0 + 2 * vid1 + 4 * vid2
                                    scaled = (
                                        acc[t * _WGMMA_ACC_REGS + vid]
                                        * w_scaled[idx]
                                        * in_s
                                    )
                                    if cutlass.const_expr(has_bias):
                                        scaled = scaled + w_bias[idx]
                                    sc[
                                        epi_row + (vid0 ^ (v & 1)) + 8 * v,
                                        epi_col + 8 * (vid1 ^ vid0),
                                    ] = scaled.to(OUTPUT_DTYPE)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.barrier(
                            barrier_id=epi_bar,
                            number_of_threads=_WGMMA_THREADS,
                        )
                        if epi_warp == 0:
                            gc = cute.local_tile(
                                tma_tensor_c,
                                (_EPI_TILE_M, _EPI_TILE_N),
                                (
                                    m_block * _EPI_M_SUBTILES + s_i,
                                    n_block * (_CTA_WEIGHT_ROWS // _EPI_TILE_N)
                                    + warpgroup * _EPI_N_SUBTILES
                                    + t,
                                ),
                            )
                            tma_sc, tma_gc = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_c,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sc, 0, 2),
                                cute.group_modes(gc, 0, 2),
                            )
                            cute.copy(tma_atom_c, tma_sc, tma_gc)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(
                                _EPI_STAGES - 1, read=True
                            )
                        cute.arch.barrier(
                            barrier_id=epi_bar,
                            number_of_threads=_WGMMA_THREADS,
                        )
                if epi_warp == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

        @cute.jit
        def __call__(
            self,
            input: cute.Tensor,
            input_scale: cute.Tensor,
            weight: cute.Tensor,
            weight_meta: cute.Tensor,
            weight_scale: cute.Tensor,
            bias: cute.Tensor,
            output: cute.Tensor,
            m_size: cutlass.Int32,
            n_size: cutlass.Int32,
            k_size: cutlass.Int32,
            metadata_rows: cutlass.Int32,
            num_n_blocks: cutlass.Int32,
            num_m_blocks: cutlass.Int32,
            raster_group_n: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            a_staged, b_staged = _make_smem_layouts()
            tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
                (
                    cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
                    if cutlass.const_expr(_use_a_mcast)
                    else cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
                ),
                weight,
                cute.slice_(a_staged, (None, None, 0)),
                (_CTA_WEIGHT_ROWS, _A_BUF_K),
                num_multicast=2 if cutlass.const_expr(_use_a_mcast) else 1,
            )
            tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                input,
                cute.slice_(b_staged, (None, None, 0)),
                (_WGMMA_INPUT_ROWS, _B_BUF_K),
            )
            tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
                output,
                _make_epi_smem_layout(),
                (_EPI_TILE_M, _EPI_TILE_N),
            )
            if cutlass.const_expr(cluster_mcast):
                self.kernel(
                    tma_atom_a,
                    tma_tensor_a,
                    tma_atom_b,
                    tma_tensor_b,
                    tma_atom_c,
                    tma_tensor_c,
                    input_scale,
                    weight_meta,
                    weight_scale,
                    bias,
                    output,
                    m_size,
                    n_size,
                    k_size,
                    metadata_rows,
                    raster_group_n,
                ).launch(
                    grid=(num_n_blocks, num_m_blocks, 1),
                    block=[_CTA_THREADS, 1, 1],
                    cluster=(1, 2, 1),
                    smem=self.shared_storage.size_in_bytes(),
                    stream=stream,
                )
            else:
                self.kernel(
                    tma_atom_a,
                    tma_tensor_a,
                    tma_atom_b,
                    tma_tensor_b,
                    tma_atom_c,
                    tma_tensor_c,
                    input_scale,
                    weight_meta,
                    weight_scale,
                    bias,
                    output,
                    m_size,
                    n_size,
                    k_size,
                    metadata_rows,
                    raster_group_n,
                ).launch(
                    grid=(num_n_blocks, num_m_blocks, 1),
                    block=[_CTA_THREADS, 1, 1],
                    smem=self.shared_storage.size_in_bytes(),
                    stream=stream,
                )

    return cute.compile(
        RowwiseScaledLinearSparseWgmma(),
        input=make_fake_tensor(
            INPUT_DTYPE,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        input_scale=make_fake_tensor(
            SCALE_DTYPE,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        ),
        weight=make_fake_tensor(
            WEIGHT_DTYPE,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        weight_meta=make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        weight_scale=make_fake_tensor(
            SCALE_DTYPE,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        ),
        bias=make_fake_tensor(
            OUTPUT_DTYPE,
            (cute.sym_int(),),
            stride=(1,),
            assumed_align=16,
        ),
        output=make_fake_tensor(
            OUTPUT_DTYPE,
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=16,
        ),
        m_size=_WGMMA_INPUT_ROWS,
        n_size=_CTA_WEIGHT_ROWS,
        k_size=_WGMMA_LOGICAL_K,
        metadata_rows=64,
        raster_group_n=1,
        num_n_blocks=1,
        num_m_blocks=1,
        stream=make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op(
    "torchao::rowwise_scaled_linear_sparse_cutedsl", mutates_args=()
)
def _rowwise_scaled_linear_sparse_cutedsl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_meta: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert input.dim() >= 2, f"Expected input to be 2D or higher, got {input.dim()}D"
    assert input.is_cuda, "Expected input to be CUDA"
    assert input.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "Expected FP8 input"
    assert weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
        "Expected FP8 weight"
    )
    assert input_scale.dtype == weight_scale.dtype, "Scale dtypes must match"
    assert input_scale.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        "Expected fp16, bf16, or fp32 scales"
    )
    output_dtype = out_dtype if out_dtype is not None else input_scale.dtype
    assert output_dtype in (torch.float16, torch.bfloat16), (
        "Expected fp16 or bf16 output"
    )
    if bias is not None:
        # Matches the legacy CUTLASS kernel's own requirement (it runtime-checks
        # bias.scalar_type() == Y.scalar_type()), not a CuTeDSL-specific limitation.
        assert bias.dtype == output_dtype, "Bias dtype must match output dtype"
        assert bias.dim() == 1, "Expected bias to be 1D"

    k = input.shape[-1]
    # An empty K would leave the kernel's K loop with nothing to wait on while
    # the producer's prologue loads are still in flight, so it is declined.
    assert k > 0, "Input K must be positive"
    assert k % 32 == 0, "Input K must be divisible by 32"
    assert weight.dim() == 2, f"Expected weight to be 2D, got {weight.dim()}D"
    n, weight_cols = weight.shape
    assert n % 8 == 0, "Weight rows must be divisible by 8"
    assert k == 2 * weight_cols, "Input K must match compressed weight columns"
    assert weight_cols % 8 == 0, "Compressed weight columns must be divisible by 8"
    assert input.stride(-1) == 1, "Expected input in row-major layout"
    # An empty input has no meaningful batch strides: torch gives a contiguous
    # (2, 0, k) tensor stride(0) == k, not 0.
    if input.numel() > 0:
        expected_stride = input.stride(-2)
        for i in range(input.dim() - 3, -1, -1):
            expected_stride *= input.size(i + 1)
            assert input.stride(i) == expected_stride, (
                "Expected input in row-major layout"
            )
    assert weight.stride(1) == 1, "Expected weight in row-major layout"
    assert weight_meta.dtype == torch.uint8, "Expected uint8 weight metadata"
    assert weight_meta.dim() == 2, "Expected weight metadata to be 2D"
    assert weight_meta.stride(1) == 1, "Expected metadata in row-major layout"
    expected_meta_rows = max(round_up(n, 64), 64)
    expected_meta_cols = max(round_up(weight_cols // 4, 16), 16)
    assert weight_meta.shape == (expected_meta_rows, expected_meta_cols), (
        "Unexpected weight metadata shape"
    )
    assert input_scale.shape == input.shape[:-1], "Unexpected input scale shape"
    assert input_scale.is_contiguous(), "Expected contiguous input scale"
    assert weight_scale.numel() == n, "Unexpected weight scale shape"
    assert weight_scale.is_contiguous(), "Expected contiguous weight scale"
    if bias is not None:
        assert bias.numel() == n, "Unexpected bias shape"
        assert bias.stride(0) == 1, "Expected contiguous bias"

    input_2d = input.reshape(-1, k)
    input_scale_1d = input_scale.flatten()
    weight_scale_1d = weight_scale.flatten()
    output = torch.empty(
        (input_2d.shape[0], n),
        dtype=output_dtype,
        device=input.device,
    )
    # The kernel never dereferences the bias pointer when has_bias is False, so
    # any live allocation will do; `output` is reused to avoid a second one. Its
    # shape (m*n) deliberately does not match a real bias (n).
    bias_arg = bias if bias is not None else output.reshape(-1)
    m = input_2d.shape[0]
    if m == 0 or n == 0:
        # Nothing to compute, and a zero-sized grid would be rejected by the
        # launch.
        return output.reshape(*input.shape[:-1], n)
    device_index = input.get_device()
    config, mma_per_wg = _select_config(m, n, _num_sms(device_index))
    cta_weight_rows = _cta_weight_rows(mma_per_wg, config)
    compiled = _compile_rowwise_scaled_linear_sparse_cutedsl(
        input.dtype,
        weight.dtype,
        input_scale.dtype,
        output_dtype,
        bias is not None,
        m % config.input_rows == 0,
        n % cta_weight_rows == 0,
        # The decode configuration mirrors legacy's, which has no cluster.
        config == _DEFAULT_CONFIG and m % (2 * config.input_rows) == 0,
        weight_meta.shape[0] >= cta_weight_rows,
        mma_per_wg,
        config,
    )
    num_n_blocks = ceil_div(n, cta_weight_rows)
    num_m_blocks = ceil_div(m, config.input_rows)
    raster_group_n = _RASTER_GROUP_N if num_n_blocks > 32 else num_n_blocks
    # The --enable-tvm-ffi runtime is positional-only, so this must match
    # the kernel's __call__ parameter order exactly (not the order above).
    compiled(
        input_2d,
        input_scale_1d,
        weight,
        weight_meta,
        weight_scale_1d,
        bias_arg,
        output,
        m,
        n,
        k,
        weight_meta.shape[0],
        num_n_blocks,
        num_m_blocks,
        raster_group_n,
        _current_raw_stream(device_index),
    )
    return output.reshape(*input.shape[:-1], n)


@_rowwise_scaled_linear_sparse_cutedsl.register_fake
def _(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_meta: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    dtype = out_dtype if out_dtype is not None else input_scale.dtype
    return input.new_empty((*input.shape[:-1], weight.shape[0]), dtype=dtype)
