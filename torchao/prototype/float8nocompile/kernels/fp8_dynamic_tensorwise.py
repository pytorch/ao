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

kernel_configs = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
]


@triton.autotune(configs=kernel_configs, key=["input_size"])
@triton.jit
def _block_amax(
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
    block_amax = tl.max(tl.abs(vals), axis=0)
    tl.atomic_max(amax_ptr, block_amax)


@triton.autotune(configs=kernel_configs, key=["input_size"])
@triton.jit
def _to_fp8(
    input_ptr,
    scale_out_ptr,
    amax_ptr,
    out_ptr,
    num_elements,
    fp8_dtype_min,
    fp8_dtype_max,
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # compute scale, must be fp32
    global_amax = tl.load(amax_ptr)
    scale = (fp8_dtype_max / tl.clamp(global_amax, min=EPS, max=float("inf"))).to(
        tl.float32
    )
    # only one program needs to store the scale
    block_id = tl.program_id(axis=0)
    if block_id == 0:
        scale_offs = tl.arange(0, 1)
        tl.store(scale_out_ptr + scale_offs, scale)

    # load block of input tensor
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
    assert hp_tensor.is_contiguous(), "tensor must be contiguous"

    num_elements = hp_tensor.numel()
    orig_shape = hp_tensor.shape
    flattened_input = hp_tensor.flatten()

    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    # allocate memory for computed scale, local block maxes, and output fp8 tensor
    scale_out = torch.empty((1,), dtype=torch.float32, device=hp_tensor.device)
    global_amax = torch.zeros((1,), dtype=torch.float32, device=hp_tensor.device)
    fp8_output = torch.empty_like(
        flattened_input, dtype=fp8_dtype, device=hp_tensor.device
    )

    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    # compute global amax to be used for scaling
    _block_amax[grid](
        flattened_input,
        global_amax,
        num_elements,
        input_dtype=tl_input_dtype,
        EPS=EPS,
    )

    # perform conversion
    _to_fp8[grid](
        flattened_input,
        scale_out,
        global_amax,
        fp8_output,
        num_elements,
        fp8_dtype_min,
        fp8_dtype_max,
        input_dtype=tl_input_dtype,
        output_dtype=tl_output_dtype,
        EPS=EPS,
    )

    return Float8Tensor(
        fp8_output.reshape(orig_shape),
        scale_out,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
