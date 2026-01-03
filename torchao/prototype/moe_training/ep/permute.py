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
- Output: MXTensor (permuted qdata + scales)

Backward:
- Input: bf16 gradient tensor
- Unpermute using saved indices (no mxfp8)
- Output: bf16 gradient tensor
"""

import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor


class Permute(torch.autograd.Function):
    """
    Permute operation for MXTensor.

    Forward:
        - Takes MXTensor input (qdata + scales)
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
        permuted_indices: torch.Tensor,
        padded_shape: torch.Size,
    ):
        """
        Args:
            mx_tensor: MXTensor input with qdata and scales
            permuted_indices: indices for permutation
            padded_shape: shape after adding padding row

        Returns:
            MXTensor: permuted MXTensor
        """
        # Extract qdata and scales from MXTensor
        qdata = mx_tensor.qdata
        scales = mx_tensor.scale

        # Add padding row of zeros to qdata
        qdata_padded = torch.vstack((qdata, qdata.new_zeros((qdata.shape[-1],))))

        # Add padding row of zeros to scales
        scales_padded = torch.vstack((scales, scales.new_zeros((scales.shape[-1],))))

        # Permute using indices
        qdata_permuted = qdata_padded[permuted_indices, :]
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
        ctx.padded_shape = padded_shape

        return mx_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: unpermute bf16 gradients.

        Args:
            grad_output: bf16 gradient tensor from upstream

        Returns:
            grad_input: bf16 gradient tensor (unpermuted)
            None values for other forward arguments
        """
        (permuted_indices,) = ctx.saved_tensors

        # Unpermute: create empty tensor and scatter gradients back to original positions
        grad_input_padded = grad_output.new_empty(ctx.padded_shape)
        grad_input_padded[permuted_indices, :] = grad_output

        # Remove the padding row (last row)
        grad_input = grad_input_padded[:-1]

        return grad_input, None, None


def permute(
    mx_tensor: MXTensor,
    permuted_indices: torch.Tensor,
    padded_shape: torch.Size,
) -> MXTensor:
    """
    Permute MXTensor based on routing indices.

    Args:
        mx_tensor: input MXTensor
        permuted_indices: permutation indices
        padded_shape: shape after padding

    Returns:
        Permuted MXTensor
    """
    return Permute.apply(mx_tensor, permuted_indices, padded_shape)
