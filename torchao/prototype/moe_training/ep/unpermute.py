# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Purple AutoGrad function: unpermute

Forward:
- Input: bf16 tensor
- Unpermute using saved indices
- Output: bf16 tensor

Backward:
- Input: MXTensor (mxfp8 qdata + scales from downstream)
- Permute qdata and scales separately
- Output: MXTensor (permuted qdata + scales)
"""

import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor


class _Unpermute(torch.autograd.Function):
    """
    Unpermute operation.

    Forward:
        - Takes bf16 input
        - Unpermutes using saved indices
        - Returns bf16 output

    Backward:
        - Takes MXTensor gradient input (qdata + scales)
        - Permutes qdata and scales separately
        - Returns MXTensor gradient output
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        permuted_indices: torch.Tensor,
        padded_shape: torch.Size,
    ):
        """
        Args:
            input: bf16 tensor to unpermute
            permuted_indices: indices used in the forward permute
            padded_shape: shape before removing padding

        Returns:
            bf16 tensor: unpermuted output
        """
        # Unpermute: create empty tensor and scatter input back to original positions
        output_padded = input.new_empty(padded_shape)
        output_padded[permuted_indices, :] = input

        # Remove the padding row (last row)
        output = output_padded[:-1]

        # Save for backward
        ctx.save_for_backward(permuted_indices)
        ctx.input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: permute MXTensor gradients.

        Args:
            grad_output: MXTensor from upstream (qdata + scales)

        Returns:
            grad_input: MXTensor (permuted qdata + scales)
            None values for other forward arguments
        """
        (permuted_indices,) = ctx.saved_tensors

        # Check if we received an MXTensor
        if isinstance(grad_output, MXTensor):
            # Extract qdata and scales
            qdata = grad_output.qdata
            scales = grad_output.scale

            # Add padding row of zeros to qdata
            qdata_padded = torch.vstack((qdata, qdata.new_zeros((qdata.shape[-1],))))

            # Add padding row of zeros to scales
            scales_padded = torch.vstack(
                (scales, scales.new_zeros((scales.shape[-1],)))
            )

            # Permute using indices
            qdata_permuted = qdata_padded[permuted_indices, :]
            scales_permuted = scales_padded[permuted_indices, :]

            # Wrap back into MXTensor with the correct shape
            grad_input = MXTensor(
                qdata_permuted,
                scales_permuted,
                elem_dtype=grad_output._elem_dtype,
                block_size=grad_output.block_size,
                orig_dtype=grad_output._orig_dtype,
                kernel_preference=grad_output.kernel_preference,
                act_quant_kwargs=grad_output.act_quant_kwargs,
                is_swizzled_scales=grad_output._is_swizzled_scales,
            )
        else:
            # Fallback: if we receive regular tensor, permute it
            grad_output_padded = torch.vstack(
                (grad_output, grad_output.new_zeros((grad_output.shape[-1],)))
            )
            grad_input = grad_output_padded[permuted_indices, :]

        return grad_input, None, None


def unpermute(
    input: torch.Tensor,
    permuted_indices: torch.Tensor,
    padded_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermute tensor based on routing indices.

    Args:
        input: bf16 input tensor
        permuted_indices: permutation indices from forward pass
        padded_shape: shape before removing padding

    Returns:
        Unpermuted bf16 tensor
    """
    return _Unpermute.apply(input, permuted_indices, padded_shape)
