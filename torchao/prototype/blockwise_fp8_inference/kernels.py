# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from threading import Lock
from typing import Callable, Optional, Tuple

import torch

# The activation quant kernel is a Triton kernel that is imported and built
# lazily (and cached) on first use, so this module imports fine on machines
# without Triton.
_act_quant_kernel: Optional[Callable] = None
_act_quant_kernel_lock = Lock()


def _get_act_quant_kernel() -> Optional[Callable]:
    global _act_quant_kernel
    kernel = _act_quant_kernel
    if kernel is None:
        with _act_quant_kernel_lock:
            kernel = _act_quant_kernel
            if kernel is None:
                kernel = _build_act_quant_kernel()
                _act_quant_kernel = kernel
    return kernel


def _build_act_quant_kernel() -> Optional[Callable]:
    from torch.utils._triton import has_triton

    if not has_triton():
        return None

    import triton
    import triton.language as tl

    # Original implementation at https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

    @triton.jit
    def _fp8_blockwise_act_quant_kernel_impl(
        x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr
    ):
        """
        Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

        Args:
            x_ptr (triton.Pointer): Pointer to the input tensor.
            y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
            s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
            BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

        Returns:
            None
        """
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs).to(tl.float32)
        s = tl.max(tl.abs(x)) / 448.0
        y = x / s
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + pid, s)

    return _fp8_blockwise_act_quant_kernel_impl


def fp8_blockwise_act_quant(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with block size being BLOCK_SIZEx1.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        dtype (torch.dtype, optional): The dtype to use for the quantized tensor. Default is `torch.float8_e4m3fn`.


    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `dtype`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    kernel = _get_act_quant_kernel()
    if kernel is None:
        raise AssertionError("unsupported without triton")

    import triton

    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s
