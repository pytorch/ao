# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Green AutoGrad function: permute

Forward:
- Input: MXTensor (mxfp8 qdata + scales)
- Permute qdata and scales separately based on routing indices
- Pads each token group to a multiple of TOKEN_GROUP_ALIGN_SIZE_M
- Output: MXTensor (permuted qdata + scales)

Backward:
- Input: bf16 gradient tensor
- Unpermute using saved indices (no mxfp8)
- Output: bf16 gradient tensor
"""

import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor

from .kernels import generate_permute_indices


def _round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y."""
    x_ceil_div_y = (x + y - 1) // y
    return x_ceil_div_y * y


class _Permute(torch.autograd.Function):
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
                _offsets,
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
        input_shape = qdata_padded.shape

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
        ctx.input_shape = input_shape

        return input_shape, mx_output, permuted_indices, num_tokens_per_expert_padded

    @staticmethod
    def backward(
        ctx,
        grad_padded_shape,
        grad_output,
        grad_permuted_indices,
        grad_num_tokens_per_expert_padded,
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

        # Unpermute: create empty tensor and scatter gradients back to original positions
        grad_input_padded = grad_output.new_empty(ctx.input_shape)
        grad_input_padded[permuted_indices, :] = grad_output

        # Remove the padding row (last row)
        grad_input = grad_input_padded[:-1]

        return grad_input, None, None, None, None


def permute(
    mx_tensor: MXTensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    group_size_multiple_of: int = 32,
) -> tuple[torch.Size, MXTensor, torch.Tensor, torch.Tensor]:
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

    Returns:
        tuple: (padded_shape, permuted MXTensor, permuted_indices, updated num_tokens_per_expert with padding)
    """
    return _Permute.apply(
        mx_tensor,
        num_tokens_per_expert,
        ep_degree,
        num_local_experts,
        group_size_multiple_of,
    )
