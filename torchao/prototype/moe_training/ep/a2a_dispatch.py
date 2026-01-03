# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pink AutoGrad function: a2a_dispatch

Forward:
- Input: bf16 tensor
- Quantize to mxfp8 using triton_to_mxfp8_dim0()
- All-to-all on qdata and scales separately
- Output: MXTensor (wrapping qdata and scales)

Backward:
- Input: bf16 gradient tensor
- Inverse all-to-all (no mxfp8, just bf16)
- Output: bf16 gradient tensor
"""

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor


class _A2ADispatch(torch.autograd.Function):
    """
    All-to-all dispatch with MXFP8 quantization in forward.

    Forward:
        - Takes bf16 input
        - Dynamically quantizes to mxfp8
        - Performs all-to-all on qdata and scales separately
        - Returns MXTensor wrapping the output qdata and scales

    Backward:
        - Takes bf16 gradient input
        - Performs inverse all-to-all (no quantization)
        - Returns bf16 gradient output
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
            input: bf16 input tensor to be dispatched
            output_splits: list of output splits for all-to-all
            input_splits: list of input splits for all-to-all
            group: process group for collective
            scaling_mode: quantization scaling mode
            block_size: block size for mxfp8 quantization

        Returns:
            MXTensor: output wrapped as MXTensor with qdata and scales
        """
        assert input.dtype in (torch.bfloat16, torch.float32), (
            f"Expected bf16 or fp32, got {input.dtype}"
        )

        # Default to WORLD group if not specified
        if group is None:
            group = dist.group.WORLD

        # Quantize input to mxfp8
        scaling_mode_str = str(scaling_mode.value).lower()
        input_data, input_scales = triton_to_mxfp8_dim0(
            input,
            inner_block_size=block_size,
            scaling_mode=scaling_mode_str,
        )

        # All-to-all on qdata (async)
        output_data = all_to_all_single(
            input_data,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        # All-to-all on scales (async)
        # NCCL doesn't support float8_e8m0fnu, so view as uint8
        output_scales = all_to_all_single(
            input_scales.view(torch.uint8),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        # Wait for async ops to complete
        output_data = torch.ops._c10d_functional.wait_tensor(output_data)
        output_scales = torch.ops._c10d_functional.wait_tensor(output_scales)

        # Convert scales back to float8_e8m0fnu
        output_scales = output_scales.view(torch.float8_e8m0fnu)

        # Wrap output as MXTensor
        mx_output = MXTensor(
            output_data,
            output_scales,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            orig_dtype=input.dtype,
            kernel_preference=None,
            act_quant_kwargs=None,
            is_swizzled_scales=False,
        )

        # Save for backward
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        ctx.hp_dtype = input.dtype

        return mx_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: inverse all-to-all in bf16 (no quantization).

        Args:
            grad_output: bf16 gradient tensor from upstream

        Returns:
            grad_input: bf16 gradient tensor
            None values for other forward arguments (output_splits, input_splits, group, scaling_mode, block_size)
        """
        # Inverse all-to-all: swap input_splits and output_splits
        grad_input = all_to_all_single(
            grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group,
        )

        # Wait for async op
        grad_input = torch.ops._c10d_functional.wait_tensor(grad_input)

        return grad_input, None, None, None, None, None


def a2a_dispatch(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup = None,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    block_size: int = 32,
) -> MXTensor:
    """
    All-to-all dispatch with MXFP8 quantization.

    Args:
        input: bf16 input tensor
        output_splits: output split sizes
        input_splits: input split sizes
        group: process group
        scaling_mode: quantization scaling mode
        block_size: mxfp8 block size

    Returns:
        MXTensor with quantized output from all-to-all
    """
    return _A2ADispatch.apply(
        input,
        output_splits,
        input_splits,
        group,
        scaling_mode,
        block_size,
    )
