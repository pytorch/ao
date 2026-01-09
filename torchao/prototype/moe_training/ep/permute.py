# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torchao.prototype.mx_formats.mx_tensor import MXTensor

from .kernels import generate_permute_indices


def _round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y


class _PermuteMXFP8FwdHPBwd(torch.autograd.Function):
    """
    Permute operation for MXTensor with token group alignment.

    Forward:
        - Takes MXTensor input (qdata + scales)
        - Generates permutation indices using triton kernel
        - Pads each token group to multiple of TOKEN_GROUP_ALIGN_SIZE_M
        - Permutes qdata and scales separately based on routing
        - Returns MXTensor with permuted components

    Backward:
        - Takes bf16 gradient input
        - Unpermutes using saved indices
        - Returns bf16 gradient output
    """

    @staticmethod
    def forward(
        ctx,
        mx_tensor: MXTensor,
        num_tokens_per_expert: torch.Tensor,
        ep_degree: int,
        num_local_experts: int,
        group_size_multiple_of: int = 32,
        use_triton_for_bwd: bool = True,
    ):
        """
        Args:
            mx_tensor: MXTensor input with qdata and scales
            num_tokens_per_expert: number of tokens per expert (shape: [ep_degree * num_local_experts])
            ep_degree: expert parallelism degree
            num_local_experts: number of local experts
            group_size_multiple_of: alignment size for token groups

        Returns:
            tuple: (padded_shape, permuted MXTensor, permuted_indices, updated num_tokens_per_expert with padding)
        """

        # Extract qdata and scales from MXTensor
        qdata = mx_tensor.qdata
        scales = mx_tensor.scale

        # Assume worst case where each token group needs to be padded with group_size_multiple_of tokens
        x_padded_per_expert = (
            qdata.shape[0] + num_local_experts * group_size_multiple_of
        )
        padded_max_len = _round_up(x_padded_per_expert, group_size_multiple_of)

        # Generate permuted indices and updated num_tokens_per_expert with padding
        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert_padded,
                group_offsets,
            ) = generate_permute_indices(
                num_tokens_per_expert,
                num_local_experts,
                ep_degree,
                padded_max_len,
                group_size_multiple_of,
            )

        # Append row of zeros to qdata to act as a dummy padding row
        # Every time `permuted_indices` selects this index, it will correspond to a padded token
        qdata_padded = torch.vstack((qdata, qdata.new_zeros((qdata.shape[-1],))))
        padded_input_shape = qdata_padded.shape

        # Permute qdata using indices
        qdata_permuted = qdata_padded[permuted_indices, :]

        # Permute scales to match data
        scales_padded = torch.vstack((scales, scales.new_zeros((scales.shape[-1],))))
        scales_permuted = scales_padded[permuted_indices, :]

        # Wrap back into MXTensor
        mx_output = MXTensor(
            qdata_permuted,
            scales_permuted,
            elem_dtype=mx_tensor._elem_dtype,
            block_size=mx_tensor.block_size,
            orig_dtype=mx_tensor._orig_dtype,
            kernel_preference=mx_tensor.kernel_preference,
            act_quant_kwargs=mx_tensor.act_quant_kwargs,
            is_swizzled_scales=mx_tensor._is_swizzled_scales,
        )

        # Save for backward
        ctx.save_for_backward(permuted_indices)
        ctx.padded_input_shape = padded_input_shape
        ctx.use_triton_for_bwd = use_triton_for_bwd

        return (
            padded_input_shape,
            mx_output,
            permuted_indices,
            num_tokens_per_expert_padded,
            group_offsets,
        )

    @staticmethod
    def backward(
        ctx,
        grad_padded_shape,
        grad_output,
        grad_permuted_indices,
        grad_num_tokens_per_expert_padded,
        grad_group_offsets,
    ):
        """
        Backward pass: unpermute bf16 gradients.

        Args:
            grad_padded_shape: None (padded_shape doesn't need gradients)
            grad_output: bf16 gradient tensor from upstream
            grad_permuted_indices: None (permuted_indices doesn't need gradients)
            grad_num_tokens_per_expert_padded: None (num_tokens_per_expert_padded doesn't need gradients)

        Returns:
            grad_input: bf16 gradient tensor (unpermuted)
            None values for other forward arguments
        """
        (permuted_indices,) = ctx.saved_tensors
        input_rows, input_cols = ctx.padded_input_shape
        use_triton_for_bwd = ctx.use_triton_for_bwd

        if use_triton_for_bwd:
            grad_input = _triton_permute_bwd(
                grad_output,
                permuted_indices,
                input_rows - 1,  # Remove padding row
                input_cols,
            )
        else:
            # Unpermute: scatter gradients back to original positions
            grad_input_padded = grad_output.new_zeros(ctx.padded_input_shape)
            grad_input_padded[permuted_indices, :] = grad_output

            # Remove the padding row (last row)
            grad_input = grad_input_padded[:input_rows]
        return grad_input, None, None, None, None, None


# Reference impl for testing
def _permute_bf16(
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    alignment: int,
):
    """
    BF16 permute operation used for testing and benchmarking.

    Args:
        x: BF16 input tensor
        num_tokens_per_expert: number of tokens per expert
        ep_degree: expert parallelism degree
        num_local_experts: number of local experts
        alignment: block size alignment

    Returns:
        tuple: (input_shape, permuted tensor, permuted_indices, offsets)
    """
    x_padded_per_expert = x.shape[0] + num_local_experts * alignment
    padded_max_len = _round_up(x_padded_per_expert, alignment)

    with torch.no_grad():
        (
            permuted_indices,
            num_tokens_per_expert_padded,
            group_offsets,
        ) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            alignment,
        )

    x = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert_padded, group_offsets


@torch._dynamo.nonstrict_trace
def permute_mxfp8_fwd_hp_bwd(
    mx_tensor: MXTensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    group_size_multiple_of: int = 32,
    use_mxfp8: bool = True,
    use_triton_for_bwd: bool = True,
):
    """
    Permute and pad MXTensor based on routing indices with token group alignment.

    This function:
    1. Uses a triton kernel to efficiently generate permutation indices
    2. Pads each token group to a multiple of TOKEN_GROUP_ALIGN_SIZE_M
    3. Permutes the MXTensor data accordingly

    Args:
        mx_tensor: input MXTensor
        num_tokens_per_expert: number of tokens per expert (shape: [ep_degree * num_local_experts])
        ep_degree: expert parallelism degree
        num_local_experts: number of local experts
        group_size_multiple_of: alignment size for token groups
        use_triton_for_bwd: if True, use triton kernel for backward pass

    Returns:
        tuple: (padded_shape, permuted MXTensor, permuted_indices, updated num_tokens_per_expert with padding)
    """
    return _PermuteMXFP8FwdHPBwd.apply(
        mx_tensor,
        num_tokens_per_expert,
        ep_degree,
        num_local_experts,
        group_size_multiple_of,
        use_triton_for_bwd,
    )


@triton_op("torchao::_triton_permute_bwd", mutates_args={})
def _triton_permute_bwd(
    grad_output: torch.Tensor,
    permuted_indices: torch.Tensor,
    original_rows: int,
    original_cols: int,
) -> torch.Tensor:
    """
    Backward pass for BF16 permute operation.

    Args:
        grad_output: bf16 gradient tensor from upstream
        permuted_indices: permutation indices used in the forward pass
        original_shape: original shape of the input (with extra padding row)
    Returns:
        grad_input: bf16 gradient tensor (unpermuted)
    """
    grad_rows, grad_cols = grad_output.shape
    output_buffer = grad_output.new_zeros((original_rows, original_cols))
    grid = lambda meta: (
        triton.cdiv(grad_rows, meta["BLOCK_ROWS"]),
        triton.cdiv(grad_cols, meta["BLOCK_COLS"]),
    )
    wrap_triton(_triton_permute_bwd_kernel)[grid](
        grad_output,
        permuted_indices,
        output_buffer,
        grad_rows,
        grad_cols,
        original_rows,
        original_cols,
        BLOCK_ROWS=256,
        BLOCK_COLS=256,
        PADDING_VALUE=-1,
    )
    return output_buffer


@triton.jit
def _triton_permute_bwd_kernel(
    grad_ptr,
    permuted_indices_ptr,
    output_buffer_ptr,
    grad_rows,
    grad_cols,
    original_rows,
    original_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    PADDING_VALUE: tl.constexpr = -1,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_offsets = row_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = col_pid * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    dest_rows = tl.load(
        permuted_indices_ptr + row_offsets,
        mask=row_offsets < grad_rows,
        other=PADDING_VALUE,
    )

    read_mask = (row_offsets[:, None] < grad_rows) & (col_offsets[None, :] < grad_cols)
    grad_values = tl.load(
        grad_ptr + row_offsets[:, None] * grad_cols + col_offsets[None, :],
        mask=read_mask,
        other=PADDING_VALUE,
    )

    write_mask = (dest_rows[:, None] != PADDING_VALUE) & (
        col_offsets[None, :] < original_cols
    )
    tl.store(
        output_buffer_ptr + dest_rows[:, None] * original_cols + col_offsets[None, :],
        grad_values,
        mask=write_mask,
    )
