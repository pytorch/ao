# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
BF16 to MXFP8 1x32 Blockwise Quantization Kernel with MMA Layout Scale Output

Self-contained version adapted from
fbcode/ads_mkl/ops/cute_dsl/quack/mxfp8/quantize_rowwise.py.

This module provides rowwise MXFP8 quantization kernels that output scales directly
in MMA atom-tiled layout expected by SM100 blockscaled GEMM, eliminating the need
for a separate scale conversion kernel.
"""

from dataclasses import dataclass
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Uint8
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack, make_ptr

from torchao.prototype.moe_training.kernels.mxfp8.cutedsl import copy_utils
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl.utils import (
    abs_max_bf16x2,
    bitcast_bf16_to_u16,
    CompileCache,
    compute_mma_scale_offset,
    cvt_bf16x8_to_f32x8,
    E2M1_MAX_NORM_RCP,
    E4M3_MAX_NORM_RCP,
    E5M2_MAX_NORM_RCP,
    E8M0_NEUTRAL_SCALE,
    float_to_fp4_e2m1,
    float_to_fp8_e4m3,
    float_to_fp8_e5m2,
    fused_amax_to_e8m0_scale,
    fused_amax_to_e8m0_scale_stochastic,
    get_cuda_stream,
    load_u32_global,
    MMA_ATOM_K,
    MMA_ATOM_M_TOTAL,
    mul_cvt_8x_e2m1,
    mul_cvt_8x_e4m3,
    mul_cvt_8x_e5m2,
    MXFP4_BLOCK_SIZE,
    MXFP8_BLOCK_SIZE,
    pack_2xbf16_to_u32,
    pack_2xu32_to_u64,
    pack_bf16x2,
    store_u128_global,
    store_u64_global,
    store_u8_global,
    TORCH_TO_CUTE_DTYPE,
)


@dataclass
class MxFP8QuantizeRowwiseConfig:
    """Configuration for rowwise MXFP8 quantization kernel with MMA layout output.

    Thread Layout: 4 warps x 32 threads = 128 threads per block
    - Each warp handles 1 row
    - Each thread handles 2 x 32-element MXFP8 blocks
    - Block covers 4 rows x 2048 columns
    """

    chunk_elems: int = 32
    chunks_per_thread: int = 2
    threads_per_warp: int = 32
    warps_per_block: int = 4
    block_dim_m: int = 4
    block_dim_n: int = 2048
    num_threads: int = 128
    num_units_per_chunk: int = 4
    smem_rowwise_scale_bytes: int = 0


_ROWWISE_CHUNK_ELEMS: int = 32
_ROWWISE_CHUNKS_PER_THREAD: int = 2
_ROWWISE_BLOCK_DIM_M: int = 4
_ROWWISE_BLOCK_DIM_N: int = 2048


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division."""
    return (a + b - 1) // b


class MXFP8QuantizeRowwiseMMALayout:
    """Rowwise MXFP8 quantization with direct MMA layout scale output.

    This kernel writes scales directly in the MMA atom-tiled layout expected by
    SM100 blockscaled GEMM, eliminating the need for a separate scale conversion
    kernel.

    MMA Scale Layout: (32, 4, rest_m, 4, rest_k)
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        M: int,
        N: int,
        M_orig: int,
        N_orig: int,
        fp8_format: str = "e4m3",
    ) -> None:
        self.dtype = dtype
        self.M = M
        self.N = N
        self.M_orig = M_orig
        self.N_orig = N_orig
        self.fp8_format = fp8_format

        self.config = MxFP8QuantizeRowwiseConfig()

        self.max_norm_rcp: float = (
            E4M3_MAX_NORM_RCP if fp8_format == "e4m3" else E5M2_MAX_NORM_RCP
        )

        self.num_scale_cols = N_orig // MXFP8_BLOCK_SIZE
        self.rest_m = _ceil_div(M_orig, MMA_ATOM_M_TOTAL)
        self.rest_k = _ceil_div(self.num_scale_cols, MMA_ATOM_K)
        self.mma_scale_size = 32 * 4 * self.rest_m * 4 * self.rest_k

    def _smem_size_in_bytes(self) -> int:
        return 0

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScalesMMA: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        config = self.config
        grid_x = cute.ceil_div(self.M, config.block_dim_m)
        grid_y = cute.ceil_div(self.N, config.block_dim_n)

        self.kernel(
            mInput,
            mOutput,
            mScalesMMA,
            cutlass.Int32(self.M),
            cutlass.Int32(self.N),
            cutlass.Int32(self.M_orig),
            cutlass.Int32(self.num_scale_cols),
            cutlass.Int32(self.rest_m),
            cutlass.Int32(self.rest_k),
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[config.num_threads, 1, 1],
            cluster=[1, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mScalesMMA: cute.Tensor,
        M: cutlass.Int32,
        N: cutlass.Int32,
        M_orig: cutlass.Int32,
        num_scale_cols: cutlass.Int32,
        rest_m: cutlass.Int32,
        rest_k: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx_x, bidx_y, _ = cute.arch.block_idx()

        block_row: cutlass.Int64 = cutlass.Int64(bidx_x) * cutlass.Int64(
            _ROWWISE_BLOCK_DIM_M
        )
        block_col: cutlass.Int64 = cutlass.Int64(bidx_y) * cutlass.Int64(
            _ROWWISE_BLOCK_DIM_N
        )

        warp_idx: cutlass.Int32 = tidx // cutlass.Int32(32)
        lane_idx: cutlass.Int32 = tidx % cutlass.Int32(32)

        thread_row: cutlass.Int64 = block_row + cutlass.Int64(warp_idx)

        gOutput_base_ptr: cutlass.Int64 = mOutput.iterator.toint()
        gScaleMMA_base_ptr: cutlass.Int64 = mScalesMMA.iterator.toint()

        max_norm_rcp: Float32 = Float32(self.max_norm_rcp)

        padded_m: cutlass.Int32 = rest_m * cutlass.Int32(MMA_ATOM_M_TOTAL)
        padded_k: cutlass.Int32 = rest_k * cutlass.Int32(MMA_ATOM_K)

        for chunk_idx in cutlass.range_constexpr(_ROWWISE_CHUNKS_PER_THREAD):
            chunk_stride: int = 32 * _ROWWISE_CHUNK_ELEMS
            chunk_col_start: cutlass.Int32 = (
                block_col
                + cutlass.Int32(chunk_idx * chunk_stride)
                + lane_idx * cutlass.Int32(_ROWWISE_CHUNK_ELEMS)
            )

            rInput_bf16 = cute.make_fragment((32,), BFloat16)
            full_chunk: bool = (
                chunk_col_start + _ROWWISE_CHUNK_ELEMS
            ) <= N and thread_row < M

            if full_chunk:
                gInput_row = mInput[thread_row, None]
                chunk_tile_idx: cutlass.Int32 = (
                    block_col // cutlass.Int32(_ROWWISE_CHUNK_ELEMS)
                    + cutlass.Int32(chunk_idx * 32)
                    + lane_idx
                )
                gInput_chunk = cute.local_tile(gInput_row, (32,), (chunk_tile_idx,))

                tiled_copy_load = copy_utils.tiled_copy_1d(
                    BFloat16, num_threads=1, num_copy_elems=8
                )
                thr_copy_load = tiled_copy_load.get_slice(0)
                src_part_load = thr_copy_load.partition_S(gInput_chunk)
                dst_part_load = thr_copy_load.partition_D(rInput_bf16)
                cute.copy(thr_copy_load, src_part_load, dst_part_load)
            else:
                for i in cutlass.range_constexpr(32):
                    col = chunk_col_start + i
                    if col < N:
                        rInput_bf16[i] = mInput[thread_row, col]
                    else:
                        rInput_bf16[i] = BFloat16(0.0)

            amax_packed: cutlass.Uint32 = cutlass.Uint32(0)
            for i in cutlass.range_constexpr(16):
                val0_u16: cutlass.Uint16 = bitcast_bf16_to_u16(rInput_bf16[i * 2])
                val1_u16: cutlass.Uint16 = bitcast_bf16_to_u16(rInput_bf16[i * 2 + 1])
                val_packed: cutlass.Uint32 = pack_bf16x2(val0_u16, val1_u16)
                amax_packed = abs_max_bf16x2(amax_packed, val_packed)

            e8m0_scale, inv_scale = fused_amax_to_e8m0_scale(amax_packed, max_norm_rcp)

            gOutput_linear_offset: cutlass.Int64 = cutlass.Int64(
                thread_row
            ) * cutlass.Int64(N) + cutlass.Int64(chunk_col_start)

            if full_chunk:
                for batch_idx in cutlass.range_constexpr(2):
                    base: int = batch_idx * 16

                    bf16_pair_01: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 0], rInput_bf16[base + 1]
                    )
                    bf16_pair_23: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 2], rInput_bf16[base + 3]
                    )
                    bf16_pair_45: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 4], rInput_bf16[base + 5]
                    )
                    bf16_pair_67: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 6], rInput_bf16[base + 7]
                    )
                    v0, v1, v2, v3, v4, v5, v6, v7 = cvt_bf16x8_to_f32x8(
                        bf16_pair_01, bf16_pair_23, bf16_pair_45, bf16_pair_67
                    )

                    bf16_pair_89: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 8], rInput_bf16[base + 9]
                    )
                    bf16_pair_ab: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 10], rInput_bf16[base + 11]
                    )
                    bf16_pair_cd: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 12], rInput_bf16[base + 13]
                    )
                    bf16_pair_ef: cutlass.Uint32 = pack_2xbf16_to_u32(
                        rInput_bf16[base + 14], rInput_bf16[base + 15]
                    )
                    v8, v9, v10, v11, v12, v13, v14, v15 = cvt_bf16x8_to_f32x8(
                        bf16_pair_89, bf16_pair_ab, bf16_pair_cd, bf16_pair_ef
                    )

                    if cutlass.const_expr(self.fp8_format == "e4m3"):
                        packed_lo: cutlass.Int64 = mul_cvt_8x_e4m3(
                            v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                        )
                        packed_hi: cutlass.Int64 = mul_cvt_8x_e4m3(
                            v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
                        )
                    else:
                        packed_lo: cutlass.Int64 = mul_cvt_8x_e5m2(
                            v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                        )
                        packed_hi: cutlass.Int64 = mul_cvt_8x_e5m2(
                            v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
                        )

                    store_ptr: cutlass.Int64 = (
                        gOutput_base_ptr + gOutput_linear_offset + cutlass.Int64(base)
                    )
                    store_u128_global(store_ptr, packed_lo, packed_hi)
            else:
                for i in cutlass.range_constexpr(_ROWWISE_CHUNK_ELEMS):
                    global_col: cutlass.Int32 = chunk_col_start + cutlass.Int32(i)
                    if global_col < N:
                        val: Float32 = rInput_bf16[i].to(Float32)
                        scaled_val: Float32 = val * inv_scale

                        if cutlass.const_expr(self.fp8_format == "e4m3"):
                            fp8_val: Uint8 = float_to_fp8_e4m3(scaled_val)
                        else:
                            fp8_val: Uint8 = float_to_fp8_e5m2(scaled_val)

                        mOutput[thread_row, global_col] = fp8_val

            global_scale_col: cutlass.Int32 = chunk_col_start // cutlass.Int32(
                _ROWWISE_CHUNK_ELEMS
            )

            if thread_row < M_orig and global_scale_col < num_scale_cols:
                mma_offset: cutlass.Int64 = compute_mma_scale_offset(
                    thread_row, global_scale_col, rest_m, rest_k
                )
                mma_ptr: cutlass.Int64 = gScaleMMA_base_ptr + mma_offset
                store_u8_global(mma_ptr, e8m0_scale)
            elif thread_row < padded_m and global_scale_col < padded_k:
                mma_offset: cutlass.Int64 = compute_mma_scale_offset(
                    thread_row, global_scale_col, rest_m, rest_k
                )
                mma_ptr: cutlass.Int64 = gScaleMMA_base_ptr + mma_offset
                store_u8_global(mma_ptr, Uint8(E8M0_NEUTRAL_SCALE))


_mxfp8_quantize_rowwise_mma_cache = CompileCache()


# Track which CUDA devices have been initialized to avoid per-call overhead
_initialized_devices: set = set()


def _ensure_cuda_context(device: torch.device) -> None:
    """Ensure CUDA context is initialized for the given device (once per device)."""
    device_key = (device.type, device.index)
    if device_key in _initialized_devices:
        return

    torch.cuda.init()
    torch.cuda.set_device(device)
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)

    _initialized_devices.add(device_key)


def _launch_rowwise_kernel_mma_layout(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    scales_mma_tensor: torch.Tensor,
    fp8_format: str,
    M_orig: int,
    N_orig: int,
) -> None:
    """Launch rowwise MXFP8 quantization kernel with MMA layout scale output."""
    M, N = input_tensor.shape
    input_dtype = TORCH_TO_CUTE_DTYPE[input_tensor.dtype]

    device = input_tensor.device
    _ensure_cuda_context(device)

    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()
    scales_mma_tensor = scales_mma_tensor.contiguous()

    op = MXFP8QuantizeRowwiseMMALayout(
        dtype=input_dtype,
        M=M,
        N=N,
        M_orig=M_orig,
        N_orig=N_orig,
        fp8_format=fp8_format,
    )

    mInput = from_dlpack(input_tensor.detach(), assumed_align=16)
    mOutput = from_dlpack(output_tensor.view(torch.uint8).detach(), assumed_align=16)
    mScalesMMA = from_dlpack(scales_mma_tensor.detach(), assumed_align=16)

    stream = get_cuda_stream()

    device_idx = device.index if device.type == "cuda" else -1
    compile_key = (
        "rowwise_mma_layout",
        input_dtype,
        M,
        N,
        M_orig,
        N_orig,
        fp8_format,
        device_idx,
    )

    if compile_key not in _mxfp8_quantize_rowwise_mma_cache:
        _mxfp8_quantize_rowwise_mma_cache[compile_key] = cute.compile(
            op,
            mInput,
            mOutput,
            mScalesMMA,
            stream,
            options="--enable-tvm-ffi",
        )

    _mxfp8_quantize_rowwise_mma_cache[compile_key](
        mInput,
        mOutput,
        mScalesMMA,
        stream,
    )


def _launch_rowwise_kernel_mma_layout_with_stream(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    scales_mma_tensor: torch.Tensor,
    fp8_format: str,
    M_orig: int,
    N_orig: int,
    stream: cuda.CUstream,
) -> None:
    """Launch rowwise kernel with MMA layout on a specific CUDA stream."""
    M, N = input_tensor.shape
    input_dtype = TORCH_TO_CUTE_DTYPE[input_tensor.dtype]

    device = input_tensor.device
    _ensure_cuda_context(device)

    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()
    scales_mma_tensor = scales_mma_tensor.contiguous()

    op = MXFP8QuantizeRowwiseMMALayout(
        dtype=input_dtype,
        M=M,
        N=N,
        M_orig=M_orig,
        N_orig=N_orig,
        fp8_format=fp8_format,
    )

    mInput = from_dlpack(input_tensor.detach(), assumed_align=16)
    mOutput = from_dlpack(output_tensor.view(torch.uint8).detach(), assumed_align=16)
    mScalesMMA = from_dlpack(scales_mma_tensor.detach(), assumed_align=16)

    device_idx = device.index if device.type == "cuda" else -1
    compile_key = (
        "rowwise_mma_layout",
        input_dtype,
        M,
        N,
        M_orig,
        N_orig,
        fp8_format,
        device_idx,
    )

    if compile_key not in _mxfp8_quantize_rowwise_mma_cache:
        _mxfp8_quantize_rowwise_mma_cache[compile_key] = cute.compile(
            op,
            mInput,
            mOutput,
            mScalesMMA,
            stream,
            options="--enable-tvm-ffi",
        )

    _mxfp8_quantize_rowwise_mma_cache[compile_key](
        mInput,
        mOutput,
        mScalesMMA,
        stream,
    )


# =============================================================================
# PyTorch Interface Functions
# =============================================================================


def mxfp8_quantize_rowwise_mma_layout(
    input: torch.Tensor,
    fp8_format: str = "e4m3",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to MXFP8 format with scales directly in MMA layout.

    Args:
        input: BF16/FP16 tensor of shape (M, N)
        fp8_format: "e4m3" or "e5m2"

    Returns:
        Tuple of (fp8_output, scales_mma)
        - fp8_output: FP8 tensor of shape (M, N)
        - scales_mma: E8M0 scales in MMA layout (flattened 1D)
    """
    M_orig, N_orig = input.shape
    assert N_orig % MXFP8_BLOCK_SIZE == 0, f"N must be divisible by {MXFP8_BLOCK_SIZE}"
    assert input.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"Input dtype must be bfloat16 or float16 for MXFP8 quantization, got {input.dtype}"

    output_dtype = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    BLOCK_DIM_M = 4
    M_padded = _ceil_div(M_orig, BLOCK_DIM_M) * BLOCK_DIM_M

    BLOCK_DIM_N = _ROWWISE_BLOCK_DIM_N
    N_padded = _ceil_div(N_orig, BLOCK_DIM_N) * BLOCK_DIM_N

    if M_padded != M_orig or N_padded != N_orig:
        input_padded = torch.zeros(
            (M_padded, N_padded), dtype=input.dtype, device=input.device
        )
        input_padded[:M_orig, :N_orig] = input
        input = input_padded
        M = M_padded
        N = N_padded
    else:
        M = M_orig
        N = N_orig

    num_scale_cols = N_orig // MXFP8_BLOCK_SIZE
    rest_m = _ceil_div(M_orig, MMA_ATOM_M_TOTAL)
    rest_k = _ceil_div(num_scale_cols, MMA_ATOM_K)
    mma_scale_size = 32 * 4 * rest_m * 4 * rest_k

    row_output = torch.empty((M, N), dtype=output_dtype, device=input.device)
    scales_mma = torch.full(
        (mma_scale_size,),
        E8M0_NEUTRAL_SCALE,
        dtype=torch.uint8,
        device=input.device,
    )

    _launch_rowwise_kernel_mma_layout(
        input, row_output, scales_mma, fp8_format, M_orig, N_orig
    )

    return row_output[:M_orig, :N_orig], scales_mma


def mxfp8_quantize_rowwise_dual_mma_layout(
    A: torch.Tensor,
    B: torch.Tensor,
    fp8_format: str = "e4m3",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize matrices A and B concurrently with scales directly in MMA layout.

    Args:
        A: BF16/FP16 tensor of shape (M_A, K)
        B: BF16/FP16 tensor of shape (M_B, K)
        fp8_format: "e4m3" or "e5m2"

    Returns:
        Tuple of (A_fp8, A_scales_mma, B_fp8, B_scales_mma)
    """
    assert A.dim() == 2 and B.dim() == 2
    M_A, N_A = A.shape
    M_B, N_B = B.shape
    assert N_A % MXFP8_BLOCK_SIZE == 0
    assert N_B % MXFP8_BLOCK_SIZE == 0

    output_dtype = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    num_scale_cols_A = N_A // MXFP8_BLOCK_SIZE
    rest_m_A = _ceil_div(M_A, MMA_ATOM_M_TOTAL)
    rest_k_A = _ceil_div(num_scale_cols_A, MMA_ATOM_K)
    mma_scale_size_A = 32 * 4 * rest_m_A * 4 * rest_k_A

    num_scale_cols_B = N_B // MXFP8_BLOCK_SIZE
    rest_m_B = _ceil_div(M_B, MMA_ATOM_M_TOTAL)
    rest_k_B = _ceil_div(num_scale_cols_B, MMA_ATOM_K)
    mma_scale_size_B = 32 * 4 * rest_m_B * 4 * rest_k_B

    A_output = torch.empty((M_A, N_A), dtype=output_dtype, device=A.device)
    A_scales_mma = torch.full(
        (mma_scale_size_A,),
        E8M0_NEUTRAL_SCALE,
        dtype=torch.uint8,
        device=A.device,
    )

    B_output = torch.empty((M_B, N_B), dtype=output_dtype, device=B.device)
    B_scales_mma = torch.full(
        (mma_scale_size_B,),
        E8M0_NEUTRAL_SCALE,
        dtype=torch.uint8,
        device=B.device,
    )

    stream_A = torch.cuda.Stream()
    stream_B = torch.cuda.Stream()

    def get_stream_handle(torch_stream: torch.cuda.Stream) -> cuda.CUstream:
        return cuda.CUstream(torch_stream.cuda_stream)

    cuda_stream_A = get_stream_handle(stream_A)
    cuda_stream_B = get_stream_handle(stream_B)

    _launch_rowwise_kernel_mma_layout_with_stream(
        A, A_output, A_scales_mma, fp8_format, M_A, N_A, cuda_stream_A
    )
    _launch_rowwise_kernel_mma_layout_with_stream(
        B, B_output, B_scales_mma, fp8_format, M_B, N_B, cuda_stream_B
    )

    stream_A.synchronize()
    stream_B.synchronize()

    return A_output, A_scales_mma, B_output, B_scales_mma


# =============================================================================
# Custom Op Interface (torch.library)
# =============================================================================


@torch.library.custom_op("torchao::cutedsl_mxfp8_quantize_rowwise", mutates_args=())
def cutedsl_mxfp8_quantize_rowwise(
    input: torch.Tensor,
    fp8_format: str = "e4m3",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom op for MXFP8 rowwise quantization with MMA layout scale output."""
    return mxfp8_quantize_rowwise_mma_layout(input, fp8_format)


@cutedsl_mxfp8_quantize_rowwise.register_fake
def _(
    input: torch.Tensor,
    fp8_format: str = "e4m3",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile shape inference."""
    M, N = input.shape
    output_dtype = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    num_scale_cols = N // MXFP8_BLOCK_SIZE
    rest_m = _ceil_div(M, MMA_ATOM_M_TOTAL)
    rest_k = _ceil_div(num_scale_cols, MMA_ATOM_K)
    mma_scale_size = 32 * 4 * rest_m * 4 * rest_k

    fp8_output = torch.empty((M, N), dtype=output_dtype, device=input.device)
    scales_mma = torch.empty((mma_scale_size,), dtype=torch.uint8, device=input.device)
    return fp8_output, scales_mma


@torch.library.custom_op(
    "torchao::cutedsl_mxfp8_quantize_rowwise_dual", mutates_args=()
)
def cutedsl_mxfp8_quantize_rowwise_dual(
    A: torch.Tensor,
    B: torch.Tensor,
    fp8_format: str = "e4m3",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom op for concurrent MXFP8 quantization of matrices A and B."""
    return mxfp8_quantize_rowwise_dual_mma_layout(A, B, fp8_format)


@cutedsl_mxfp8_quantize_rowwise_dual.register_fake
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    fp8_format: str = "e4m3",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile shape inference."""
    M_A, N_A = A.shape
    M_B, N_B = B.shape
    output_dtype = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    num_scale_cols_A = N_A // MXFP8_BLOCK_SIZE
    rest_m_A = _ceil_div(M_A, MMA_ATOM_M_TOTAL)
    rest_k_A = _ceil_div(num_scale_cols_A, MMA_ATOM_K)
    mma_scale_size_A = 32 * 4 * rest_m_A * 4 * rest_k_A

    num_scale_cols_B = N_B // MXFP8_BLOCK_SIZE
    rest_m_B = _ceil_div(M_B, MMA_ATOM_M_TOTAL)
    rest_k_B = _ceil_div(num_scale_cols_B, MMA_ATOM_K)
    mma_scale_size_B = 32 * 4 * rest_m_B * 4 * rest_k_B

    A_fp8 = torch.empty((M_A, N_A), dtype=output_dtype, device=A.device)
    A_scales_mma = torch.empty((mma_scale_size_A,), dtype=torch.uint8, device=A.device)
    B_fp8 = torch.empty((M_B, N_B), dtype=output_dtype, device=B.device)
    B_scales_mma = torch.empty((mma_scale_size_B,), dtype=torch.uint8, device=B.device)
    return A_fp8, A_scales_mma, B_fp8, B_scales_mma


# =============================================================================
# In-Place Custom Op (for fair benchmarking without allocation overhead)
# =============================================================================


def _get_mma_scale_size(M: int, N: int) -> int:
    """Calculate the MMA scale tensor size for given dimensions."""
    num_scale_cols = N // MXFP8_BLOCK_SIZE
    rest_m = _ceil_div(M, MMA_ATOM_M_TOTAL)
    rest_k = _ceil_div(num_scale_cols, MMA_ATOM_K)
    return 32 * 4 * rest_m * 4 * rest_k


def get_output_shapes(M: int, N: int, fp8_format: str = "e4m3") -> dict:
    """Get the shapes and dtypes needed to pre-allocate output tensors."""
    BLOCK_DIM_M = 4
    BLOCK_DIM_N = _ROWWISE_BLOCK_DIM_N

    M_padded = _ceil_div(M, BLOCK_DIM_M) * BLOCK_DIM_M
    N_padded = _ceil_div(N, BLOCK_DIM_N) * BLOCK_DIM_N

    mma_scale_size = _get_mma_scale_size(M, N)
    fp8_dtype = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    return {
        "fp8_shape": (M_padded, N_padded),
        "scales_shape": (mma_scale_size,),
        "fp8_dtype": fp8_dtype,
        "M_padded": M_padded,
        "N_padded": N_padded,
    }


@torch.library.custom_op(
    "torchao::cutedsl_mxfp8_quantize_rowwise_inplace",
    mutates_args=("output_fp8", "output_scales"),
)
def cutedsl_mxfp8_quantize_rowwise_inplace(
    input: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scales: torch.Tensor,
    fp8_format: str = "e4m3",
) -> None:
    """In-place MXFP8 quantization custom op for fair benchmarking."""
    M_orig, N_orig = input.shape

    BLOCK_DIM_M = 4
    BLOCK_DIM_N = _ROWWISE_BLOCK_DIM_N
    M_padded = _ceil_div(M_orig, BLOCK_DIM_M) * BLOCK_DIM_M
    N_padded = _ceil_div(N_orig, BLOCK_DIM_N) * BLOCK_DIM_N

    if M_padded != M_orig or N_padded != N_orig:
        input_padded = torch.zeros(
            (M_padded, N_padded), dtype=input.dtype, device=input.device
        )
        input_padded[:M_orig, :N_orig] = input
        input = input_padded

    _launch_rowwise_kernel_mma_layout(
        input, output_fp8, output_scales, fp8_format, M_orig, N_orig
    )


@cutedsl_mxfp8_quantize_rowwise_inplace.register_fake
def _(
    input: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scales: torch.Tensor,
    fp8_format: str = "e4m3",
) -> None:
    """Fake implementation for torch.compile - no-op since outputs are pre-allocated."""
    pass


# =============================================================================
# Pointer-Based API (eliminates from_dlpack overhead per call)
# =============================================================================

_mxfp8_quantize_rowwise_ptr_cache = CompileCache()


@cute.jit
def _mxfp8_rowwise_e4m3_ptr_wrapper(
    input_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    scales_ptr: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    M_orig: cutlass.Int32,
    num_scale_cols: cutlass.Int32,
    rest_m: cutlass.Int32,
    rest_k: cutlass.Int32,
    mma_scale_size: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    """JIT wrapper that creates cute.Tensors from raw pointers.

    This bypasses the from_dlpack overhead by accepting raw CUDA pointers
    and creating cute.Tensor objects inside the JIT function.
    """
    input_layout = cute.make_ordered_layout((M, N), order=(0, 1))
    mInput = cute.make_tensor(input_ptr, layout=input_layout)

    output_layout = cute.make_ordered_layout((M, N), order=(0, 1))
    mOutput = cute.make_tensor(output_ptr, layout=output_layout)

    scales_layout = cute.make_layout((mma_scale_size,))
    mScalesMMA = cute.make_tensor(scales_ptr, layout=scales_layout)

    config = MxFP8QuantizeRowwiseConfig()
    max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)

    padded_m: cutlass.Int32 = rest_m * cutlass.Int32(MMA_ATOM_M_TOTAL)
    padded_k: cutlass.Int32 = rest_k * cutlass.Int32(MMA_ATOM_K)

    grid_x = cute.ceil_div(M, config.block_dim_m)
    grid_y = cute.ceil_div(N, config.block_dim_n)

    _mxfp8_rowwise_e4m3_ptr_kernel(
        mInput,
        mOutput,
        mScalesMMA,
        M,
        N,
        M_orig,
        num_scale_cols,
        rest_m,
        rest_k,
        max_norm_rcp,
        padded_m,
        padded_k,
    ).launch(
        grid=[grid_x, grid_y, 1],
        block=[config.num_threads, 1, 1],
        cluster=[1, 1, 1],
        smem=0,
        stream=stream,
    )


@cute.kernel
def _mxfp8_rowwise_e4m3_ptr_kernel(
    mInput: cute.Tensor,
    mOutput: cute.Tensor,
    mScalesMMA: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    M_orig: cutlass.Int32,
    num_scale_cols: cutlass.Int32,
    rest_m: cutlass.Int32,
    rest_k: cutlass.Int32,
    max_norm_rcp: Float32,
    padded_m: cutlass.Int32,
    padded_k: cutlass.Int32,
) -> None:
    """Rowwise MXFP8 E4M3 quantization kernel (pointer-based version)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx_x, bidx_y, _ = cute.arch.block_idx()

    block_row: cutlass.Int64 = cutlass.Int64(bidx_x) * cutlass.Int64(
        _ROWWISE_BLOCK_DIM_M
    )
    block_col: cutlass.Int64 = cutlass.Int64(bidx_y) * cutlass.Int64(
        _ROWWISE_BLOCK_DIM_N
    )

    warp_idx: cutlass.Int32 = tidx // cutlass.Int32(32)
    lane_idx: cutlass.Int32 = tidx % cutlass.Int32(32)

    thread_row: cutlass.Int64 = block_row + cutlass.Int64(warp_idx)

    gOutput_base_ptr: cutlass.Int64 = mOutput.iterator.toint()
    gScaleMMA_base_ptr: cutlass.Int64 = mScalesMMA.iterator.toint()

    for chunk_idx in cutlass.range_constexpr(_ROWWISE_CHUNKS_PER_THREAD):
        chunk_stride: int = 32 * _ROWWISE_CHUNK_ELEMS
        chunk_col_start: cutlass.Int32 = (
            block_col
            + cutlass.Int32(chunk_idx * chunk_stride)
            + lane_idx * cutlass.Int32(_ROWWISE_CHUNK_ELEMS)
        )

        rInput_bf16 = cute.make_fragment((32,), BFloat16)
        full_chunk: bool = (
            chunk_col_start + _ROWWISE_CHUNK_ELEMS
        ) <= N and thread_row < M

        if full_chunk:
            gInput_row = mInput[thread_row, None]
            chunk_tile_idx: cutlass.Int32 = (
                block_col // cutlass.Int32(_ROWWISE_CHUNK_ELEMS)
                + cutlass.Int32(chunk_idx * 32)
                + lane_idx
            )
            gInput_chunk = cute.local_tile(gInput_row, (32,), (chunk_tile_idx,))

            tiled_copy_load = copy_utils.tiled_copy_1d(
                BFloat16, num_threads=1, num_copy_elems=8
            )
            thr_copy_load = tiled_copy_load.get_slice(0)
            src_part_load = thr_copy_load.partition_S(gInput_chunk)
            dst_part_load = thr_copy_load.partition_D(rInput_bf16)
            cute.copy(thr_copy_load, src_part_load, dst_part_load)
        else:
            for i in cutlass.range_constexpr(32):
                col = chunk_col_start + i
                if col < N:
                    rInput_bf16[i] = mInput[thread_row, col]
                else:
                    rInput_bf16[i] = BFloat16(0.0)

        amax_packed: cutlass.Uint32 = cutlass.Uint32(0)
        for i in cutlass.range_constexpr(16):
            val0_u16: cutlass.Uint16 = bitcast_bf16_to_u16(rInput_bf16[i * 2])
            val1_u16: cutlass.Uint16 = bitcast_bf16_to_u16(rInput_bf16[i * 2 + 1])
            val_packed: cutlass.Uint32 = pack_bf16x2(val0_u16, val1_u16)
            amax_packed = abs_max_bf16x2(amax_packed, val_packed)

        e8m0_scale, inv_scale = fused_amax_to_e8m0_scale(amax_packed, max_norm_rcp)

        gOutput_linear_offset: cutlass.Int64 = cutlass.Int64(
            thread_row
        ) * cutlass.Int64(N) + cutlass.Int64(chunk_col_start)

        if full_chunk:
            for batch_idx in cutlass.range_constexpr(2):
                base: int = batch_idx * 16

                bf16_pair_01: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 0], rInput_bf16[base + 1]
                )
                bf16_pair_23: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 2], rInput_bf16[base + 3]
                )
                bf16_pair_45: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 4], rInput_bf16[base + 5]
                )
                bf16_pair_67: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 6], rInput_bf16[base + 7]
                )
                v0, v1, v2, v3, v4, v5, v6, v7 = cvt_bf16x8_to_f32x8(
                    bf16_pair_01, bf16_pair_23, bf16_pair_45, bf16_pair_67
                )

                bf16_pair_89: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 8], rInput_bf16[base + 9]
                )
                bf16_pair_ab: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 10], rInput_bf16[base + 11]
                )
                bf16_pair_cd: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 12], rInput_bf16[base + 13]
                )
                bf16_pair_ef: cutlass.Uint32 = pack_2xbf16_to_u32(
                    rInput_bf16[base + 14], rInput_bf16[base + 15]
                )
                v8, v9, v10, v11, v12, v13, v14, v15 = cvt_bf16x8_to_f32x8(
                    bf16_pair_89, bf16_pair_ab, bf16_pair_cd, bf16_pair_ef
                )

                packed_lo: cutlass.Int64 = mul_cvt_8x_e4m3(
                    v0, v1, v2, v3, v4, v5, v6, v7, inv_scale
                )
                packed_hi: cutlass.Int64 = mul_cvt_8x_e4m3(
                    v8, v9, v10, v11, v12, v13, v14, v15, inv_scale
                )

                store_ptr: cutlass.Int64 = (
                    gOutput_base_ptr + gOutput_linear_offset + cutlass.Int64(base)
                )
                store_u128_global(store_ptr, packed_lo, packed_hi)
        else:
            for i in cutlass.range_constexpr(_ROWWISE_CHUNK_ELEMS):
                global_col: cutlass.Int32 = chunk_col_start + cutlass.Int32(i)
                if global_col < N:
                    val: Float32 = rInput_bf16[i].to(Float32)
                    scaled_val: Float32 = val * inv_scale
                    fp8_val: Uint8 = float_to_fp8_e4m3(scaled_val)
                    mOutput[thread_row, global_col] = fp8_val

        global_scale_col: cutlass.Int32 = chunk_col_start // cutlass.Int32(
            _ROWWISE_CHUNK_ELEMS
        )

        if thread_row < M_orig and global_scale_col < num_scale_cols:
            mma_offset: cutlass.Int64 = compute_mma_scale_offset(
                thread_row, global_scale_col, rest_m, rest_k
            )
            mma_ptr: cutlass.Int64 = gScaleMMA_base_ptr + mma_offset
            store_u8_global(mma_ptr, e8m0_scale)
        elif thread_row < padded_m and global_scale_col < padded_k:
            mma_offset: cutlass.Int64 = compute_mma_scale_offset(
                thread_row, global_scale_col, rest_m, rest_k
            )
            mma_ptr: cutlass.Int64 = gScaleMMA_base_ptr + mma_offset
            store_u8_global(mma_ptr, Uint8(E8M0_NEUTRAL_SCALE))


def mxfp8_quantize_rowwise_ptr(
    input: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scales: torch.Tensor,
) -> None:
    """
    Pointer-based MXFP8 rowwise quantization — eliminates from_dlpack overhead.

    This function takes PRE-ALLOCATED output tensors and quantizes the input
    directly using raw CUDA pointers, bypassing the from_dlpack conversion
    overhead that occurs with the standard API.

    Use ``get_output_shapes()`` to determine correct output sizes.

    Args:
        input: BF16 tensor of shape (M, N) where N is divisible by 32.
               Must already be padded if padding is needed.
        output_fp8: Pre-allocated FP8 E4M3 tensor of shape (M_padded, N_padded)
        output_scales: Pre-allocated uint8 tensor of shape (mma_scale_size,)

    Example::

        shapes = get_output_shapes(M, N)
        output_fp8 = torch.empty(shapes["fp8_shape"], dtype=shapes["fp8_dtype"], device="cuda")
        output_scales = torch.full(shapes["scales_shape"], 127, dtype=torch.uint8, device="cuda")

        # Pad input if needed
        x_padded = pad_input(x, shapes)

        # First call triggers JIT compilation (slow)
        mxfp8_quantize_rowwise_ptr(x_padded, output_fp8, output_scales)

        # Subsequent calls are fast — no from_dlpack, no allocation
        mxfp8_quantize_rowwise_ptr(x_padded, output_fp8, output_scales)
    """
    M_orig, N_orig = input.shape
    assert N_orig % MXFP8_BLOCK_SIZE == 0, f"N must be divisible by {MXFP8_BLOCK_SIZE}"
    assert input.dtype in (
        torch.bfloat16,
    ), f"Input must be bfloat16, got {input.dtype}"

    device = input.device
    _ensure_cuda_context(device)

    # Compute MMA scale dimensions from original (unpadded) shape
    # For the pointer API, input is expected to already be padded,
    # so M_orig/N_orig here are the padded dimensions.
    # The caller must pass the original M via the output_scales shape.
    M, N = input.shape
    num_scale_cols = N // MXFP8_BLOCK_SIZE
    rest_m = _ceil_div(M, MMA_ATOM_M_TOTAL)
    rest_k = _ceil_div(num_scale_cols, MMA_ATOM_K)
    mma_scale_size = 32 * 4 * rest_m * 4 * rest_k

    input = input.contiguous()
    output_fp8 = output_fp8.contiguous()
    output_scales = output_scales.contiguous()

    input_cute_dtype = TORCH_TO_CUTE_DTYPE[input.dtype]

    input_ptr = make_ptr(
        input_cute_dtype, input.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    output_ptr = make_ptr(
        Uint8, output_fp8.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    scales_ptr = make_ptr(
        Uint8, output_scales.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )

    stream = get_cuda_stream()

    device_idx = device.index if device.type == "cuda" else -1
    compile_key = (
        "mxfp8_rowwise_e4m3_ptr",
        input_cute_dtype,
        M,
        N,
        device_idx,
    )

    if compile_key not in _mxfp8_quantize_rowwise_ptr_cache:
        _mxfp8_quantize_rowwise_ptr_cache[compile_key] = cute.compile(
            _mxfp8_rowwise_e4m3_ptr_wrapper,
            input_ptr,
            output_ptr,
            scales_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(M),
            cutlass.Int32(num_scale_cols),
            cutlass.Int32(rest_m),
            cutlass.Int32(rest_k),
            cutlass.Int32(mma_scale_size),
            stream,
            options="--enable-tvm-ffi",
        )

    _mxfp8_quantize_rowwise_ptr_cache[compile_key](
        input_ptr,
        output_ptr,
        scales_ptr,
        cutlass.Int32(M),
        cutlass.Int32(N),
        cutlass.Int32(M),
        cutlass.Int32(num_scale_cols),
        cutlass.Int32(rest_m),
        cutlass.Int32(rest_k),
        cutlass.Int32(mma_scale_size),
        stream,
    )
