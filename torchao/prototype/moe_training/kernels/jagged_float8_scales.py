# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for scaling high precision tensors to float8 using "jagged"
rowwise scales (i.e., separate scales for each group/subtensor as determined by
the offsets).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

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

block_sizes = [1, 16, 32, 64]
block_sizes_iter = [32, 64, 128, 256]
num_warps = [1, 4]
num_stages = [2, 3]
kernel_configs_2D = [
    triton.Config(
        {"BLOCK_SIZE": block_size, "BLOCK_SIZE_ITER": block_size_iter},
        num_warps=warps,
        num_stages=stages,
    )
    for block_size in block_sizes
    for block_size_iter in block_sizes_iter
    for warps in num_warps
    for stages in num_stages
]


@torch.library.custom_op(
    "torchao::triton_fp8_per_group_rowwise_scales", mutates_args={}
)
def triton_fp8_per_group_rowwise_scales(
    hp_tensor: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a high precision tensor to a float8 tensor in row-major memory layout,
    using 'jagged' rowwise scales (i.e., separate scales for each group/subtensor as
    determined by the offsets).

    Args:
        - hp_tensor: 2D high precision tensor to be converted
        - offsets: end index for each group/subtensor along dim 0
        - output_dtype: desired float8 dtype for the output tensor
        - round_scales_to_power_of_2: boolean indicating if scales should be rounded
            down to the nearest power of 2.
    Returns:
        - float8 tensor
        - jagged rowwise scales (i.e., rowwise scales for each group)
    """
    assert hp_tensor.ndim == 2, "input tensor must be 2D"

    num_elements = hp_tensor.numel()
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

    fp8_dtype_min = torch.finfo(output_dtype).min
    fp8_dtype_max = torch.finfo(output_dtype).max

    m, k = hp_tensor.shape
    n_groups = offsets.numel()

    # allocate on-device buffers for output and scales
    output_buffer = torch.empty((m, k), dtype=output_dtype, device=hp_tensor.device)
    scales_buffer = torch.empty(
        (m * n_groups), dtype=torch.float32, device=hp_tensor.device
    )

    # parallelize across rows and groups (offsets)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE"]),
        offsets.numel(),
    )
    _triton_fp8_per_group_rowwise_scales_kernel[grid](
        hp_tensor,
        offsets,
        output_buffer,
        scales_buffer,
        m,
        k,
        hp_tensor.stride(0),
        hp_tensor.stride(1),
        output_buffer.stride(0),
        output_buffer.stride(1),
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
        round_scales_to_power_of_2,
        EPS=EPS,
    )
    return output_buffer, scales_buffer


@triton_fp8_per_group_rowwise_scales.register_fake
def _fake_triton_fp8_per_group_rowwise_scales_kernel(
    hp_tensor: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hp_tensor.ndim == 2, "input tensor must be 2D"
    m, k = hp_tensor.shape
    n_groups = offsets.numel()
    output = torch.empty_like(hp_tensor, dtype=output_dtype).as_strided(
        (m, k),  # shape
        (k, 1),  # stride
    )
    scales = torch.empty((m * n_groups), dtype=torch.float32, device=hp_tensor.device)
    return output, scales


# This kernel is used on grad_output.t() which has shape (K, M),
# before the calculation `grad_B = grad_output_t @ input`.
# However, in this code, we use the conventional dim names (M, K)
# so the kernel is easily interpretable in a standalone fasion.
# The tokens per expert will vary per iteration, so don't want
# to recompile on `token` dim (K, in this case) changes.
@triton.autotune(configs=kernel_configs_2D, key=["M"])
@triton.jit
def _triton_fp8_per_group_rowwise_scales_kernel(
    input_ptr,
    offsets_ptr,
    out_ptr,
    scales_ptr,
    M: int,
    K: int,
    stride_input_row: int,
    stride_input_col: int,
    stride_output_row: int,
    stride_output_col: int,
    num_elements: int,
    fp8_dtype_min: tl.constexpr,
    fp8_dtype_max: tl.constexpr,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    round_scales_to_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_ITER: tl.constexpr,
    EPS: tl.constexpr,
):
    # parallel across rows and groups (offsets)
    block_row_id = tl.program_id(axis=0)
    offset_idx = tl.program_id(axis=1)

    # determine start and end column idx for this group
    group_col_start_idx = tl.load(
        offsets_ptr + offset_idx - 1, mask=offset_idx > 0, other=0
    )
    group_col_end_idx = tl.load(offsets_ptr + offset_idx)
    block_row_offs = block_row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # compute rowwise amaxes for this group
    amax_buffer = tl.zeros((BLOCK_SIZE,), dtype=input_dtype)
    for col_start_idx in range(group_col_start_idx, group_col_end_idx, BLOCK_SIZE_ITER):
        block_col_offs = col_start_idx + tl.arange(0, BLOCK_SIZE_ITER)
        block_offs = (
            block_row_offs[:, None] * stride_input_row
            + block_col_offs[None, :] * stride_input_col
        )
        block_mask = (block_row_offs[:, None] < M) & (
            block_col_offs[None, :] < group_col_end_idx
        )
        data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
            input_dtype
        )
        # we need to cast back to input dtype since triton promotes bf16 to fp32:
        # https://github.com/triton-lang/triton/blob/981e987eed9053b952f81153bc0779c99d8c642e/python/triton/language/standard.py#L173
        amax_buffer = tl.maximum(amax_buffer, tl.max(tl.abs(data), axis=1)).to(
            input_dtype
        )

    # compute rowwise scales for this group. round scales to nearest power of 2.
    amax_buffer = amax_buffer.to(tl.float64)
    scales = (fp8_dtype_max / tl.clamp(amax_buffer, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    if round_scales_to_power_of_2:
        scales = tl.exp2(tl.floor(tl.log2(scales)))

    # store rowwise scales for each group in contiguous memory:
    # [group0_row0, group_0_row1, ..., group2_row0, group2_row1]
    scales_offs = block_row_offs + (M * offset_idx)
    scales_mask = tl.arange(0, BLOCK_SIZE) < M
    tl.store(scales_ptr + scales_offs, scales, mask=scales_mask)

    # perform float8 conversion for this group
    for col_start_idx in range(group_col_start_idx, group_col_end_idx, BLOCK_SIZE_ITER):
        block_col_offs = col_start_idx + tl.arange(0, BLOCK_SIZE_ITER)
        block_offs = (
            block_row_offs[:, None] * stride_input_row
            + block_col_offs[None, :] * stride_input_col
        )
        block_mask = (block_row_offs[:, None] < M) & (
            block_col_offs[None, :] < group_col_end_idx
        )
        data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
            input_dtype
        )
        scaled_data = data * scales[:, None]
        fp8_data = tl.clamp(scaled_data, min=fp8_dtype_min, max=fp8_dtype_max).to(
            output_dtype
        )
        out_offs = (
            block_row_offs[:, None] * stride_output_row
            + block_col_offs[None, :] * stride_output_col
        )
        tl.store(out_ptr + out_offs, fp8_data, mask=block_mask)


@torch.library.custom_op(
    "torchao::triton_fp8_per_group_colwise_scales", mutates_args={}
)
def triton_fp8_per_group_colwise_scales(
    hp_tensor: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a high precision tensor to a float8 tensor in row-major memory layout,
    using 'jagged' column-wise scales (i.e., separate scales for each group/subtensor as
    determined by the offsets).

    Args:
        - hp_tensor: 2D high precision tensor to be converted
        - offsets: end index for each group/subtensor along dim 0
        - output_dtype: desired float8 dtype for the output tensor
        - round_scales_to_power_of_2: boolean indicating if scales should be rounded
            down to the nearest power of 2.
    Returns:
        - float8 tensor
        - jagged column-wise scales (i.e., column-wise scales for each group)
    """
    assert hp_tensor.ndim == 2, "input tensor must be 2D"

    num_elements = hp_tensor.numel()
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[output_dtype]

    fp8_dtype_min = torch.finfo(output_dtype).min
    fp8_dtype_max = torch.finfo(output_dtype).max

    k, n = hp_tensor.shape
    n_groups = offsets.numel()

    # Output buffer in column major
    output_buffer = torch.empty_like(
        hp_tensor, dtype=output_dtype, device=hp_tensor.device
    ).as_strided(hp_tensor.size(), (1, k))

    scales_buffer = torch.empty(
        (n * n_groups), dtype=torch.float32, device=hp_tensor.device
    )

    # parallelize across columns and groups (offsets)
    grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE"]),
        offsets.numel(),
    )
    _triton_fp8_per_group_colwise_scales_kernel[grid](
        hp_tensor,
        offsets,
        output_buffer,
        scales_buffer,
        k,
        n,
        hp_tensor.stride(0),
        hp_tensor.stride(1),
        output_buffer.stride(0),
        output_buffer.stride(1),
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        tl_input_dtype,
        tl_output_dtype,
        round_scales_to_power_of_2,
        EPS=EPS,
    )
    return output_buffer, scales_buffer


@triton_fp8_per_group_colwise_scales.register_fake
def _fake_triton_fp8_per_group_colwise_scales(
    hp_tensor: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hp_tensor.ndim == 2, "input tensor must be 2D"
    k, n = hp_tensor.shape
    n_groups = offsets.numel()
    output_buffer = torch.empty_like(
        hp_tensor, dtype=output_dtype, device=hp_tensor.device
    ).as_strided(hp_tensor.size(), (1, k))

    scales_buffer = torch.empty(
        (n * n_groups), dtype=torch.float32, device=hp_tensor.device
    )
    return output_buffer, scales_buffer


# This kernel is used on `input` which has shape (M, K),
# before the calculation `grad_B = grad_output_t @ input`.
# The tokens per expert will vary per iteration, so don't want
# to recompile on `token` dim (M) changes.
@triton.autotune(configs=kernel_configs_2D, key=["K"])
@triton.jit
def _triton_fp8_per_group_colwise_scales_kernel(
    input_ptr,
    offsets_ptr,
    out_ptr,
    scales_ptr,
    K: int,
    N: int,
    stride_input_row: int,
    stride_input_col: int,
    stride_output_row: int,
    stride_output_col: int,
    num_elements: int,
    fp8_dtype_min: tl.constexpr,
    fp8_dtype_max: tl.constexpr,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    round_scales_to_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_ITER: tl.constexpr,
    EPS: tl.constexpr,
):
    # parallel across columns and groups (offsets)
    block_col_id = tl.program_id(axis=0)
    offset_idx = tl.program_id(axis=1)

    # determine start and end row idx for this group
    group_row_start_idx = tl.load(
        offsets_ptr + offset_idx - 1, mask=offset_idx > 0, other=0
    )
    group_row_end_idx = tl.load(offsets_ptr + offset_idx)
    block_col_offs = block_col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # compute colwise amaxes for this group
    amax_buffer = tl.zeros((BLOCK_SIZE,), dtype=input_dtype)
    for row_start_idx in range(group_row_start_idx, group_row_end_idx, BLOCK_SIZE_ITER):
        block_row_offs = row_start_idx + tl.arange(0, BLOCK_SIZE_ITER)
        block_offs = (
            block_row_offs[:, None] * stride_input_row
            + block_col_offs[None, :] * stride_input_col
        )
        block_mask = (block_row_offs[:, None] < group_row_end_idx) & (
            block_col_offs[None, :] < N
        )
        data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
            input_dtype
        )
        # we need to cast back to input dtype since triton promotes bf16 to fp32:
        # https://github.com/triton-lang/triton/blob/981e987eed9053b952f81153bc0779c99d8c642e/python/triton/language/standard.py#L173
        amax_buffer = tl.maximum(amax_buffer, tl.max(tl.abs(data), axis=0)).to(
            input_dtype
        )

    # compute rowwise scales for this group.
    amax_buffer = amax_buffer.to(tl.float64)
    scales = (fp8_dtype_max / tl.clamp(amax_buffer, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    if round_scales_to_power_of_2:
        scales = tl.exp2(tl.floor(tl.log2(scales)))

    # store colwise scales for each group in contiguous memory:
    # [group0_col0, group_0_col1, ..., group2_col0, group2_col1]
    # note: input tensor is in col-major memory layout.
    scales_offs = block_col_offs + (N * offset_idx)
    scales_mask = tl.arange(0, BLOCK_SIZE) < N
    tl.store(scales_ptr + scales_offs, scales, mask=scales_mask)

    # perform float8 conversion for this group
    for row_start_idx in range(group_row_start_idx, group_row_end_idx, BLOCK_SIZE_ITER):
        block_row_offs = row_start_idx + tl.arange(0, BLOCK_SIZE_ITER)
        block_offs = (
            block_row_offs[:, None] * stride_input_row
            + block_col_offs[None, :] * stride_input_col
        )
        block_mask = (block_row_offs[:, None] < group_row_end_idx) & (
            block_col_offs[None, :] < N
        )
        data = tl.load(input_ptr + block_offs, mask=block_mask, other=0.0).to(
            input_dtype
        )
        scaled_data = data * scales[None, :]
        fp8_data = tl.clamp(scaled_data, min=fp8_dtype_min, max=fp8_dtype_max).to(
            output_dtype
        )
        out_offs = (
            block_row_offs[:, None] * stride_output_row
            + block_col_offs[None, :] * stride_output_col
        )
        tl.store(out_ptr + out_offs, fp8_data, mask=block_mask)
