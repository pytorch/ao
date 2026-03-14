# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.distributed_c10d import _resolve_process_group

from torchao.prototype.moe_training.kernels.mxfp8.comms import (
    mxfp8_on_device_all_to_all_v_mx,
)
from torchao.prototype.moe_training.utils import conditional_nostrict_trace
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import MXTensor

from .kernels import generate_permute_indices


def _resolve_group(group_name: str | None) -> dist.ProcessGroup:
    if group_name is None:
        return dist.group.WORLD
    return _resolve_process_group(group_name)


class _A2ADispatchMXFP8FwdHPBwd(torch.autograd.Function):
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


@conditional_nostrict_trace
def a2a_dispatch_mxfp8_fwd_hp_bwd(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group_name: str = None,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    block_size: int = 32,
) -> MXTensor:
    """
    All-to-all dispatch with MXFP8 quantization in forward and high-precision backward.

    Args:
        input: bf16 input tensor
        output_splits: output split sizes
        input_splits: input split sizes
        group_name: process group name
        scaling_mode: quantization scaling mode
        block_size: mxfp8 block size

    Returns:
        MXTensor with quantized output from all-to-all
    """
    group = _resolve_group(group_name)

    return _A2ADispatchMXFP8FwdHPBwd.apply(
        input,
        output_splits,
        input_splits,
        group,
        scaling_mode,
        block_size,
    )


@conditional_nostrict_trace
def a2a_dispatch_and_group_mxfp8_fwd_hp_bwd(
    input: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    num_tokens_per_expert: torch.Tensor,
    group_name: str = None,
    block_size: int = 32,
) -> tuple[torch.Size, MXTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fuse MXFP8 dispatch with the expert grouping/padding layout needed by grouped GEMM.

    This keeps the forward path in MXFP8 and returns the same metadata that
    `permute_mxfp8_fwd_hp_bwd(...)` would have produced for downstream
    `unpermute_hp_fwd_mxfp8_bwd(...)`.
    """
    assert input.dtype in (torch.bfloat16, torch.float32), (
        f"Expected bf16 or fp32, got {input.dtype}"
    )
    assert block_size == 32, "Only block_size=32 is supported"

    group = _resolve_group(group_name)
    ep_degree = dist.get_world_size(group)
    assert len(input_splits) == ep_degree, (
        f"Expected {ep_degree} input splits, got {len(input_splits)}"
    )
    assert len(output_splits) == ep_degree, (
        f"Expected {ep_degree} output splits, got {len(output_splits)}"
    )
    assert num_tokens_per_expert.numel() % ep_degree == 0, (
        "num_tokens_per_expert must contain one contiguous block of local experts per EP rank"
    )

    num_local_experts = num_tokens_per_expert.numel() // ep_degree
    expert_splits_per_rank = num_tokens_per_expert.view(ep_degree, num_local_experts)
    input_splits_tensor = expert_splits_per_rank.sum(dim=1, dtype=torch.int64)
    expected_input_splits = torch.tensor(
        input_splits, dtype=torch.int64, device=input.device
    )
    assert torch.equal(input_splits_tensor, expected_input_splits), (
        f"input_splits {input_splits} do not match num_tokens_per_expert-derived splits {input_splits_tensor.tolist()}"
    )

    max_input_rows_per_rank = torch.tensor(
        [input.shape[0]], device=input.device, dtype=torch.int64
    )
    dist.all_reduce(max_input_rows_per_rank, op=dist.ReduceOp.MAX, group=group)
    max_output_rows_per_rank = int(max_input_rows_per_rank.item()) * ep_degree + (
        block_size * num_local_experts
    )
    (
        mx_output,
        output_splits_tensor,
        output_expert_splits,
        padded_group_end_offsets,
    ) = mxfp8_on_device_all_to_all_v_mx(
        input,
        input_splits_tensor,
        max_output_rows_per_rank,
        expert_splits_per_rank,
        group,
    )

    expected_output_splits = torch.tensor(
        output_splits, dtype=torch.int64, device=input.device
    )
    assert torch.equal(output_splits_tensor, expected_output_splits), (
        f"kernel output_splits {output_splits_tensor.tolist()} do not match expected {output_splits}"
    )

    padded_rows = int(padded_group_end_offsets[-1].item())
    permuted_indices, num_tokens_per_expert_padded, group_offsets = (
        generate_permute_indices(
            output_expert_splits.reshape(-1),
            num_local_experts,
            ep_degree,
            padded_rows,
            block_size,
            use_cpu=True,
        )
    )
    permuted_indices = permuted_indices.to(device=input.device, dtype=torch.int32)
    assert torch.equal(group_offsets.to(torch.int64), padded_group_end_offsets), (
        f"group_offsets {group_offsets.tolist()} do not match padded_group_end_offsets {padded_group_end_offsets.tolist()}"
    )

    padded_input_shape = torch.Size((sum(output_splits) + 1, input.shape[1]))
    return (
        padded_input_shape,
        mx_output,
        permuted_indices,
        num_tokens_per_expert_padded,
        group_offsets,
    )
