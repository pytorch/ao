# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.fp8_grouped_mm.utils import conditional_nostrict_trace
from torchao.prototype.mx_formats.mx_tensor import MXTensor


class _UnpermuteHPFwdMXFP8Bwd(torch.autograd.Function):
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
        Backward pass: permute gradients (MXTensor or bf16 tensor).

        Args:
            grad_output: MXTensor from upstream (qdata + scales) or bf16 tensor

        Returns:
            grad_input: MXTensor (permuted qdata + scales) or bf16 tensor
            None values for other forward arguments
        """
        (permuted_indices,) = ctx.saved_tensors

        # Check if we received an MXTensor or bf16 tensor
        if isinstance(grad_output, MXTensor):
            # MXTensor path: permute qdata and scales separately
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
                elem_dtype=grad_output.elem_dtype,
                block_size=grad_output.block_size,
                orig_dtype=grad_output.orig_dtype,
                kernel_preference=grad_output.kernel_preference,
                act_quant_kwargs=grad_output.act_quant_kwargs,
                is_swizzled_scales=grad_output.is_swizzled_scales,
            )
        else:
            # BF16 tensor path: permute directly
            # Add padding row of zeros
            grad_padded = torch.vstack(
                (grad_output, grad_output.new_zeros((grad_output.shape[-1],)))
            )

            # Permute using indices
            grad_input = grad_padded[permuted_indices, :]

        return grad_input, None, None


@conditional_nostrict_trace
def unpermute_hp_fwd_mxfp8_bwd(
    input: torch.Tensor,
    permuted_indices: torch.Tensor,
    padded_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermute tensor based on routing indices.

    Operates in bf16 during forward and in MXFP8 during backward.

    Args:
        input: bf16 input tensor
        permuted_indices: permutation indices from forward pass
        padded_shape: shape before removing padding

    Returns:
        Unpermuted bf16 tensor
    """
    return _UnpermuteHPFwdMXFP8Bwd.apply(input, permuted_indices, padded_shape)


# Reference impl for testing
def _unpermute_bf16(
    out: torch.Tensor,
    permuted_indices: torch.Tensor,
    input_shape: torch.Size,
) -> torch.Tensor:
    """
    BF16 unpermute operation used for benchmarking.

    This is a non-autograd version that operates on BF16 tensors directly.

    Args:
        out: BF16 output tensor from grouped matrix multiplication
        permuted_indices: indices used during permutation
        input_shape: shape to restore the tensor to after unpermutation

    Returns:
        Unpermuted BF16 tensor
    """
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out
