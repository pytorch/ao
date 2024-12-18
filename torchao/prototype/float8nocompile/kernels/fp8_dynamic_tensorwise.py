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
def _triton_to_fp8(
    input_ptr,
    scale_out_ptr,
    tensor_out_ptr,
    fp8_dtype_min,
    fp8_dtype_max,
    n_elements,
    input_dtype: tl.constexpr,
    tensor_out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)

    # get amax
    amax = tl.zeros([1], dtype=tl.float64)
    for i in range(0, n_elements, BLOCK_SIZE):
        block_offs = (i * BLOCK_SIZE) + offs
        block_mask = block_offs < n_elements
        vals = tl.load(input_ptr + block_offs, mask=block_mask).to(input_dtype)
        amax = tl.maximum(amax, tl.max(tl.abs(vals)))
        import pdb

        pdb.set_trace()

    # compute scale, must be fp32
    scale = (fp8_dtype_max / tl.clamp(amax, min=EPS, max=float("inf"))).to(tl.float32)
    scale_offs = tl.arange(0, 1)
    tl.store(scale_out_ptr + scale_offs, scale)

    # perform conversion
    for i in range(0, n_elements, BLOCK_SIZE):
        block_offs = (i * BLOCK_SIZE) + offs
        block_mask = block_offs < n_elements
        vals = tl.load(input_ptr + block_offs, mask=block_mask)
        vals = vals * scale
        fp8_vals = tl.clamp(vals, min=fp8_dtype_min, max=fp8_dtype_max).to(
            tensor_out_dtype
        )
        tl.store(tensor_out_ptr + block_offs, fp8_vals, mask=block_mask)


def triton_hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
) -> Float8Tensor:

    num_elements = hp_tensor.numel()
    tl_input_dtype = FP8_DTYPE_MAP[hp_tensor.dtype]
    tl_output_dtype = FP8_DTYPE_MAP[fp8_dtype]
    grid = lambda meta: (triton.cdiv(hp_tensor.numel(), meta["BLOCK_SIZE"]),)

    tensor_out = torch.empty_like(hp_tensor, dtype=fp8_dtype, device=hp_tensor.device)
    scale_out = torch.empty((1,), dtype=torch.float32, device=hp_tensor.device)

    fp8_dtype_min = torch.finfo(fp8_dtype).min
    fp8_dtype_max = torch.finfo(fp8_dtype).max

    _triton_to_fp8[grid](
        hp_tensor.flatten(),
        scale_out,
        tensor_out,
        fp8_dtype_min,
        fp8_dtype_max,
        num_elements,
        input_dtype=tl_input_dtype,
        tensor_out_dtype=tl_output_dtype,
        BLOCK_SIZE=8,  # TODO: tune
        EPS=1e-12,
    )

    return Float8Tensor(
        tensor_out,
        scale_out,
        orig_dtype=hp_tensor.dtype,
        linear_mm_config=linear_mm_config,
        gemm_input_role=gemm_input_role,
    )
