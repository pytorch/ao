# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Blue AutoGrad function: a2a_combine

Forward:
- Input: bf16 tensor
- All-to-all in bf16 (no quantization)
- Output: bf16 tensor

Backward:
- Input: bf16 gradient tensor
- Quantize to mxfp8 using triton_to_mxfp8_dim0()
- Inverse all-to-all on qdata and scales separately
- Output: MXTensor (wrapping grad qdata and scales)
"""

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor


class A2ACombine(torch.autograd.Function):
    """
    All-to-all combine with MXFP8 quantization in backward.

    Forward:
        - Takes bf16 input
        - Performs all-to-all in bf16 (no quantization)
        - Returns bf16 output

    Backward:
        - Takes bf16 gradient input
        - Dynamically quantizes to mxfp8
        - Performs inverse all-to-all on qdata and scales
        - Returns MXTensor wrapping the gradient qdata and scales
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group: dist.ProcessGroup,
        scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
        block_size: int = 32,
    ):
        """
        Args:
            input: bf16 input tensor
            output_splits: list of output splits for all-to-all
            input_splits: list of input splits for all-to-all
            group: process group for collective
            scaling_mode: quantization scaling mode (for backward)
            block_size: block size for mxfp8 quantization (for backward)

        Returns:
            bf16 tensor: all-to-all output
        """
        assert input.dtype in (torch.bfloat16, torch.float32), (
            f"Expected bf16 or fp32, got {input.dtype}"
        )

        # Default to WORLD group if not specified
        if group is None:
            group = dist.group.WORLD

        # All-to-all in bf16
        output = all_to_all_single(
            input,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        # Wait for async op
        output = torch.ops._c10d_functional.wait_tensor(output)

        # Save for backward
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.hp_dtype = input.dtype
        ctx.scaling_mode = scaling_mode
        ctx.block_size = block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: quantize to mxfp8 and perform inverse all-to-all.

        Args:
            grad_output: bf16 gradient tensor from upstream

        Returns:
            grad_input: MXTensor with gradient qdata and scales
            None values for other forward arguments (output_splits, input_splits, group, scaling_mode, block_size)
        """
        # Quantize grad_output to mxfp8
        scaling_mode_str = str(ctx.scaling_mode.value).lower()
        grad_data, grad_scales = triton_to_mxfp8_dim0(
            grad_output,
            inner_block_size=ctx.block_size,
            scaling_mode=scaling_mode_str,
        )

        # Inverse all-to-all on qdata (async)
        grad_input_data = all_to_all_single(
            grad_data,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )

        # Inverse all-to-all on scales (async)
        # NCCL doesn't support float8_e8m0fnu, so view as uint8
        grad_input_scales = all_to_all_single(
            grad_scales.view(torch.uint8),
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )

        # Wait for async ops
        grad_input_data = torch.ops._c10d_functional.wait_tensor(grad_input_data)
        grad_input_scales = torch.ops._c10d_functional.wait_tensor(grad_input_scales)

        # Convert scales back to float8_e8m0fnu
        grad_input_scales = grad_input_scales.view(torch.float8_e8m0fnu)

        # Wrap as MXTensor
        grad_input = MXTensor(
            grad_input_data,
            grad_input_scales,
            elem_dtype=torch.float8_e4m3fn,
            block_size=ctx.block_size,
            orig_dtype=ctx.hp_dtype,
            kernel_preference=None,
            act_quant_kwargs=None,
            is_swizzled_scales=False,
        )

        return grad_input, None, None, None, None, None


def a2a_combine(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    block_size: int = 32,
) -> torch.Tensor:
    """
    All-to-all combine with MXFP8 quantization in backward.

    Args:
        input: bf16 input tensor
        output_splits: output split sizes
        input_splits: input split sizes
        group: process group
        scaling_mode: quantization scaling mode (for backward)
        block_size: mxfp8 block size (for backward)

    Returns:
        bf16 output from all-to-all
    """
    return A2ACombine.apply(
        input,
        output_splits,
        input_splits,
        group,
        scaling_mode,
        block_size,
    )
