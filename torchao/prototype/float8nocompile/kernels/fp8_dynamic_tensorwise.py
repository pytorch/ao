# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton kernels for scaling high precision tensors to float8.
"""

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


@triton.jit
def _block_amax(
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
    block_amax = tl.max(tl.abs(vals), axis=0)
    tl.store(block_amaxes_ptr + block_id, block_amax)


@triton.jit
def _fp8_scale(
    block_amaxes_ptr,
    scale_out_ptr,
    num_elements,
    fp8_dtype_max,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # calculate global amax across all blocks
    global_amax = tl.zeros([1], dtype=tl.float64)
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(num_blocks):
        block_max = tl.load(block_amaxes_ptr + i)
        global_amax = tl.maximum(global_amax, block_max)

    # compute scale, must be fp32
    scale = (fp8_dtype_max / tl.clamp(global_amax, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    scale_off = tl.arange(0, 1)
    tl.store(scale_out_ptr + scale_off, scale)


@triton.jit
def _to_fp8(
    input_ptr,
    scale_ptr,
    out_ptr,
    num_elements,
    fp8_dtype_min,
    fp8_dtype_max,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # load previously computed scale
    scale = tl.load(scale_ptr)

    # load block of input tensor
    block_id = tl.program_id(axis=0)
    block_start = block_id * BLOCK_SIZE
    block_offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = block_offs < num_elements
    vals = tl.load(input_ptr + block_offs, mask=mask).to(input_dtype)

    # perform conversion
    vals = vals * scale
    fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(output_dtype)
    tl.store(out_ptr + block_offs, fp8_vals, mask=mask)


def triton_hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
) -> Float8Tensor:

    BLOCK_SIZE = 8

    num_elements = hp_tensor.numel()
    orig_shape = hp_tensor.shape
    flattened_input = hp_tensor.flatten()

    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # allocate memory for computed scale, local block maxes, and output fp8 tensor
    scale_out = torch.empty((1,), dtype=torch.float32, device=hp_tensor.device)
    block_amaxes = torch.zeros(
        (num_elements // BLOCK_SIZE,), dtype=torch.float32, device=hp_tensor.device
    )
    fp8_output = torch.empty_like(
        flattened_input, dtype=fp8_dtype, device=hp_tensor.device
    )

    # compute local amax for each block
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
    _block_amax[grid](
        flattened_input,
        block_amaxes,
        num_elements,
        input_dtype=tl_input_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=EPS,
    )

    # calculate global amax across all blocks and use it to compute scale
    _fp8_scale[(1, 1, 1)](
        block_amaxes,
        scale_out,
        num_elements,
        fp8_dtype_max,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=EPS,
    )

    # perform conversion
    _to_fp8[grid](
        flattened_input,
        scale_out,
        fp8_output,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        input_dtype=tl_input_dtype,
        output_dtype=tl_output_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        EPS=EPS,
    )

    return Float8Tensor(
        fp8_output.reshape(orig_shape),
        scale_out,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
