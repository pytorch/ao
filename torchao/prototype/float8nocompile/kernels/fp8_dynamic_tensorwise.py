# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for scaling high precision tensors to float8.
"""

from enum import Enum

import torch
import triton
import triton.language as tl

from torchao.float8.float8_tensor import Float8Tensor, GemmInputRole, LinearMMConfig

EPS = 1e-12

FP8_DTYPE_MAP = {
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


kernel_configs_1D = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]

kernel_configs_2D = [
    triton.Config({"BLOCK_SIZE_ROWS": 32, "BLOCK_SIZE_COLS": 32}, num_warps=1),
    triton.Config({"BLOCK_SIZE_ROWS": 64, "BLOCK_SIZE_COLS": 64}, num_warps=8),
    triton.Config({"BLOCK_SIZE_ROWS": 128, "BLOCK_SIZE_COLS": 128}, num_warps=4),
]


class KernelAlgorithm(Enum):
    """Enum for FP8 conversion strategy."""

    # use atomic max to compute global amax between blocks
    ATOMIC_MAX = "atomic_max"

    # reduce shared buffer containing local block amaxes to find global amax
    REDUCTION = "reduction"


@triton.autotune(configs=kernel_configs_1D, key=["num_elements"])
@triton.jit
def _to_fp8_row_major(
    input_ptr,
    out_ptr,
    scale_ptr,
    num_elements: int,
    fp8_dtype_min: float,
    fp8_dtype_max: float,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    block_id = tl.program_id(axis=0)

    # load scaling factor
    scale = tl.load(scale_ptr).to(tl.float32)

    # load block of input tensor
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)
    tl.store(out_ptr + block_offs, fp8_vals, mask=mask)


@triton.autotune(
    configs=kernel_configs_2D,
    key=["num_elements"],
)
@triton.jit
def _to_fp8_row_major_t(
    input_ptr,
    out_ptr,
    scale_ptr,
    num_elements: int,
    fp8_dtype_min: float,
    fp8_dtype_max: float,
    input_num_rows: int,
    input_num_cols: int,
    output_num_rows: int,
    output_num_cols: int,
    input_stride_row: int,
    input_stride_col: int,
    output_stride_row: int,
    output_stride_col: int,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    EPS: tl.constexpr,
):
    block_row_id = tl.program_id(axis=0)
    block_col_id = tl.program_id(axis=1)

    # load scaling factor
    scale = tl.load(scale_ptr).to(tl.float32)

    # load block of input tensor
    block_row_start = block_row_id * BLOCK_SIZE_ROWS
    block_col_start = block_col_id * BLOCK_SIZE_COLS
    block_row_offs = block_row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_offs = block_col_start + tl.arange(0, BLOCK_SIZE_COLS)
    input_offs = (
        block_row_offs[:, None] * input_stride_row
        + block_col_offs[None, :] * input_stride_col
    )
    input_mask = (block_row_offs[:, None] < input_num_rows) & (
        block_col_offs[None, :] < input_num_cols
    )
    vals = tl.load(input_ptr + input_offs, mask=input_mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)

    # write back in tranposed output tensor
    out_offs = (
        block_col_offs[:, None] * output_stride_row
        + block_row_offs[None, :] * output_stride_col
    )
    out_mask = (block_row_offs[:, None] < output_num_rows) & (
        block_col_offs[None, :] < output_num_cols
    )
    tl.store(out_ptr + out_offs, fp8_vals.trans(1, 0), mask=out_mask)


@triton.autotune(
    configs=kernel_configs_2D,
    key=["num_elements"],
)
@triton.jit
def _to_fp8_col_major(
    input_ptr,
    out_ptr,
    scale_ptr,
    num_elements: int,
    fp8_dtype_min: float,
    fp8_dtype_max: float,
    num_rows: int,
    num_cols: int,
    out_stride_row: int,
    out_stride_col: int,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    EPS: tl.constexpr,
):
    block_row_id = tl.program_id(axis=0)
    block_col_id = tl.program_id(axis=1)

    # load scaling factor
    scale = tl.load(scale_ptr).to(tl.float32)

    # load block of input tensor
    block_row_start = block_row_id * BLOCK_SIZE_ROWS
    block_col_start = block_col_id * BLOCK_SIZE_COLS
    block_row_offs = block_row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_offs = block_col_start + tl.arange(0, BLOCK_SIZE_COLS)
    block_offs = block_row_offs[:, None] * num_cols + block_col_offs[None, :]
    mask = (block_row_offs[:, None] < num_rows) & (block_col_offs[None, :] < num_cols)
    vals = tl.load(input_ptr + block_offs, mask=mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)
    out_offs = block_col_offs[None, :] * num_rows + block_row_offs[:, None]
    tl.store(out_ptr + out_offs, fp8_vals, mask=mask)


@triton.autotune(
    configs=kernel_configs_2D,
    key=["num_elements"],
)
@triton.jit
def to_fp8_col_major_t(
    input_ptr,
    out_ptr,
    scale_ptr,
    num_elements: int,
    fp8_dtype_min: float,
    fp8_dtype_max: float,
    input_num_rows: int,
    input_num_cols: int,
    output_num_rows: int,
    output_num_cols: int,
    input_stride_row: int,
    input_stride_col: int,
    output_stride_row: int,
    output_stride_col: int,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    EPS: tl.constexpr,
):
    block_row_id = tl.program_id(axis=0)
    block_col_id = tl.program_id(axis=1)

    # load scaling factor
    scale = tl.load(scale_ptr).to(tl.float32)

    # load block of input tensor
    block_row_start = block_row_id * BLOCK_SIZE_ROWS
    block_col_start = block_col_id * BLOCK_SIZE_COLS
    block_row_offs = block_row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_offs = block_col_start + tl.arange(0, BLOCK_SIZE_COLS)
    input_offs = (
        block_row_offs[:, None] * input_stride_row
        + block_col_offs[None, :] * input_stride_col
    )
    input_mask = (block_row_offs[:, None] < input_num_rows) & (
        block_col_offs[None, :] < input_num_cols
    )
    vals = tl.load(input_ptr + input_offs, mask=input_mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)
    fp8_vals = fp8_vals.trans(1, 0)

    # write back in tranposed output tensor
    out_offs = (
        block_col_offs[:, None] * output_stride_row
        + block_row_offs[None, :] * output_stride_col
    )
    out_mask = (block_row_offs[:, None] < output_num_rows) & (
        block_col_offs[None, :] < output_num_cols
    )
    tl.store(out_ptr + out_offs, fp8_vals, mask=out_mask)


@triton.autotune(
    configs=kernel_configs_2D,
    key=["num_elements"],
)
@triton.jit
def _to_fp8_row_and_col_major(
    input_ptr,
    row_major_out_ptr,
    col_major_out_ptr,
    scale_ptr,
    num_elements: int,
    fp8_dtype_min: float,
    fp8_dtype_max: float,
    num_rows: int,
    num_cols: int,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    EPS: tl.constexpr,
):
    block_row_id = tl.program_id(axis=0)
    block_col_id = tl.program_id(axis=1)

    # load scaling factor
    scale = tl.load(scale_ptr).to(tl.float32)

    # load block of input tensor
    block_row_start = block_row_id * BLOCK_SIZE_ROWS
    block_col_start = block_col_id * BLOCK_SIZE_COLS
    block_row_offs = block_row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    block_col_offs = block_col_start + tl.arange(0, BLOCK_SIZE_COLS)
    block_offs = block_row_offs[:, None] * num_cols + block_col_offs[None, :]
    mask = (block_row_offs[:, None] < num_rows) & (block_col_offs[None, :] < num_cols)
    vals = tl.load(input_ptr + block_offs, mask=mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)

    # write row major output
    row_major_offs = block_row_offs[:, None] * num_cols + block_col_offs[None, :]
    tl.store(row_major_out_ptr + row_major_offs, fp8_vals, mask=mask)

    # write column major output
    col_major_offs = block_col_offs[None, :] * num_rows + block_row_offs[:, None]
    tl.store(col_major_out_ptr + col_major_offs, fp8_vals, mask=mask)


@triton.autotune(configs=kernel_configs_1D, key=["num_elements"])
@triton.jit
def _amax_atomic(
    input_ptr,
    amax_ptr,
    num_elements,
    input_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # compute local amax for each block
    block_id = tl.program_id(axis=0)
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    block_mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=block_mask).to(input_dtype)
    block_amax = tl.max(tl.abs(vals))
    tl.atomic_max(amax_ptr, block_amax)


@triton.jit
def _scale_atomic(
    amax_ptr,
    scale_out_ptr,
    fp8_dtype_max,
    EPS: tl.constexpr,
):
    # load previously computed global amax
    global_amax = tl.load(amax_ptr).to(tl.float64)

    # compute scale, must be fp32
    scale = (fp8_dtype_max / tl.clamp(global_amax, min=EPS, max=float("inf"))).to(
        tl.float32
    )

    # store scale for use in Float8Tensor constructor
    scale_off = tl.arange(0, 1)
    tl.store(scale_out_ptr + scale_off, scale)


@triton.jit
def _amax_reduction(
    input_ptr,
    block_amaxes_ptr,
    num_elements,
    input_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # compute local amax for each block
    block_id = tl.program_id(axis=0)
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    block_mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=block_mask).to(input_dtype)
    block_amax = tl.max(tl.abs(vals))
    tl.store(block_amaxes_ptr + block_id, block_amax)


@triton.jit
def _scale_reduction(
    block_amaxes_ptr,
    scale_out_ptr,
    num_elements,
    fp8_dtype_max,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # calculate global amax across all blocks
    global_amax = tl.zeros([1], dtype=tl.float64)
    num_blocks = tl.cdiv(num_elements, BLOCK_SIZE)
    for i in range(num_blocks):
        block_max = tl.load(block_amaxes_ptr + i)
        global_amax = tl.maximum(global_amax, block_max)

    # compute scale, must be fp32
    scale = (fp8_dtype_max / tl.clamp(global_amax, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    scale_off = tl.arange(0, 1)
    tl.store(scale_out_ptr + scale_off, scale)


def hp_to_fp8_row_major(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
) -> Float8Tensor:
    assert hp_tensor.is_contiguous(), "input tensor must be contiguous"

    num_elements = hp_tensor.numel()
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # compute scaling factor for tensor
    scale = _hp_tensor_to_scale(
        hp_tensor,
        tl_input_dtype,
        fp8_dtype_max,
        algo,
    )

    # perform fp8 conversion
    output_buffer = torch.empty_like(
        hp_tensor, dtype=fp8_dtype, device=hp_tensor.device
    )
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    _to_fp8_row_major[grid](
        hp_tensor,
        output_buffer,
        scale,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
        EPS=EPS,
    )

    # wrap output tensor in Float8Tensor
    fp8_tensor_row_major = Float8Tensor(
        output_buffer,
        scale,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
    return fp8_tensor_row_major


def hp_to_fp8_row_major_t(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
) -> Float8Tensor:
    assert hp_tensor.is_contiguous(), "input tensor must be contiguous"

    num_elements = hp_tensor.numel()
    input_num_rows, input_num_cols = hp_tensor.shape
    output_num_rows, output_num_cols = input_num_cols, input_num_rows
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # compute scaling factor for tensor
    scale = _hp_tensor_to_scale(
        hp_tensor,
        tl_input_dtype,
        fp8_dtype_max,
        algo,
    )

    # perform conversion
    output_buffer = torch.empty(
        (output_num_rows, output_num_cols), dtype=fp8_dtype, device=hp_tensor.device
    )
    grid = lambda meta: (
        triton.cdiv(input_num_rows, meta["BLOCK_SIZE_ROWS"]),
        triton.cdiv(input_num_cols, meta["BLOCK_SIZE_COLS"]),
    )
    _to_fp8_row_major_t[grid](
        hp_tensor,
        output_buffer,
        scale,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        input_num_rows,
        input_num_cols,
        output_num_rows,
        output_num_cols,
        hp_tensor.stride(0),
        hp_tensor.stride(1),
        output_buffer.stride(0),
        output_buffer.stride(1),
        input_dtype=tl_input_dtype,
        output_dtype=tl_output_dtype,
        EPS=EPS,
    )

    # wrap output tensor in Float8Tensor
    fp8_tensor_row_major_t = Float8Tensor(
        output_buffer,
        scale,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
    return fp8_tensor_row_major_t


def hp_to_fp8_col_major(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
) -> Float8Tensor:
    assert hp_tensor.is_contiguous(), "input tensor must be contiguous"

    num_elements = hp_tensor.numel()
    num_rows, num_cols = hp_tensor.shape
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # compute scaling factor for tensor
    scale = _hp_tensor_to_scale(
        hp_tensor,
        tl_input_dtype,
        fp8_dtype_max,
        algo,
    )

    # perform fp8 conversion
    output_buffer = torch.empty_like(
        hp_tensor, dtype=fp8_dtype, device=hp_tensor.device
    )
    grid = lambda meta: (
        triton.cdiv(num_rows, meta["BLOCK_SIZE_ROWS"]),
        triton.cdiv(num_cols, meta["BLOCK_SIZE_COLS"]),
    )
    _to_fp8_col_major[grid](
        hp_tensor,
        output_buffer,
        scale,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        num_rows,
        num_cols,
        output_buffer.stride(0),
        output_buffer.stride(1),
        input_dtype=tl_input_dtype,
        output_dtype=tl_output_dtype,
        EPS=EPS,
    )

    # for col major we need to update the strides to reflect the new memory layout
    col_major_strides = (1, num_rows)
    output_buffer = output_buffer.as_strided(output_buffer.size(), col_major_strides)

    # wrap output tensor in Float8Tensor
    fp8_tensor_col_major = Float8Tensor(
        output_buffer,
        scale,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
    return fp8_tensor_col_major


def hp_to_fp8_col_major_t(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
) -> Float8Tensor:
    assert hp_tensor.is_contiguous(), "input tensor must be contiguous"

    num_elements = hp_tensor.numel()
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # compute scaling factor for tensor
    scale = _hp_tensor_to_scale(
        hp_tensor,
        tl_input_dtype,
        fp8_dtype_max,
        algo,
    )

    # perform conversion
    output_buffer = torch.empty_like(
        hp_tensor.t(), dtype=fp8_dtype, device=hp_tensor.device
    )
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    _to_fp8_row_major[grid](
        hp_tensor,
        output_buffer,
        scale,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
        EPS=EPS,
    )

    # wrap output tensor in Float8Tensor
    fp8_tensor_col_major_t = Float8Tensor(
        output_buffer,
        scale,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
    return fp8_tensor_col_major_t


def hp_to_fp8_row_and_col_major(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
) -> Float8Tensor:
    assert hp_tensor.is_contiguous(), "input tensor must be contiguous"

    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # compute scaling factor for tensor
    scale = _hp_tensor_to_scale(
        hp_tensor,
        tl_input_dtype,
        fp8_dtype_max,
        algo,
    )

    # perform fp8 conversion
    orig_shape = hp_tensor.shape
    num_elements = hp_tensor.numel()

    # preallocate necessary output tensors
    fp8_output_row_major = torch.empty(
        orig_shape, dtype=fp8_dtype, device=hp_tensor.device
    )
    fp8_output_col_major = torch.empty(
        orig_shape, dtype=fp8_dtype, device=hp_tensor.device
    )

    # launch triton kernel to perform conversion
    num_rows, num_cols = orig_shape
    grid = lambda meta: (
        triton.cdiv(num_rows, meta["BLOCK_SIZE_ROWS"]),
        triton.cdiv(num_cols, meta["BLOCK_SIZE_COLS"]),
    )
    _to_fp8_row_and_col_major[grid](
        hp_tensor,
        fp8_output_row_major,
        fp8_output_col_major,
        scale,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        num_rows,
        num_cols,
        input_dtype=tl_input_dtype,
        output_dtype=tl_output_dtype,
        EPS=EPS,
    )

    # for col major we need to update the strides to reflect the new memory layout
    col_major_strides = (1, num_rows)
    fp8_output_col_major = fp8_output_col_major.as_strided(
        fp8_output_col_major.size(), col_major_strides
    )

    # wrap outputs in Float8Tensors
    fp8_tensor_row_major = Float8Tensor(
        fp8_output_row_major,
        scale,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
    fp8_tensor_col_major = Float8Tensor(
        fp8_output_col_major,
        scale,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
    return fp8_tensor_row_major, fp8_tensor_col_major


def _hp_tensor_to_scale(
    hp_tensor: torch.Tensor,
    tl_input_dtype: tl.core.dtype,
    fp8_dtype_max: float,
    algo: KernelAlgorithm,
) -> torch.Tensor:
    num_elements = hp_tensor.numel()
    scale_out = torch.empty((), dtype=torch.float32, device=hp_tensor.device)
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    # compute the fp8 scale using the given algorithm
    if algo == KernelAlgorithm.ATOMIC_MAX:
        global_amax = torch.zeros((1,), dtype=torch.float32, device=hp_tensor.device)
        # compute global amax to be used for scaling
        _amax_atomic[grid](
            hp_tensor,
            global_amax,
            num_elements,
            input_dtype=tl_input_dtype,
            EPS=EPS,
        )

        # compute scale for fp8 conversion
        _scale_atomic[1, 1, 1](
            global_amax,
            scale_out,
            fp8_dtype_max,
            EPS=EPS,
        )

    elif algo == KernelAlgorithm.REDUCTION:
        # max block size and num warps values determined via manual tuning
        max_block_size = 4096
        num_warps = 8
        block_size = triton.next_power_of_2(min(max_block_size, num_elements))
        amax_buffer_size = triton.cdiv(num_elements, block_size)
        block_amaxes = torch.zeros(
            (amax_buffer_size,), dtype=torch.float32, device=hp_tensor.device
        )
        # compute local amax for each block
        _amax_reduction[grid](
            hp_tensor,
            block_amaxes,
            num_elements,
            input_dtype=tl_input_dtype,
            BLOCK_SIZE=block_size,
            EPS=EPS,
            num_warps=num_warps,
        )

        # calculate global amax across all blocks and use it to compute scale
        _scale_reduction[(1, 1, 1)](
            block_amaxes,
            scale_out,
            num_elements,
            fp8_dtype_max,
            BLOCK_SIZE=block_size,
            EPS=EPS,
        )
    else:
        raise ValueError(f"Unsupported kernel algorithm: {algo}")

    return scale_out
