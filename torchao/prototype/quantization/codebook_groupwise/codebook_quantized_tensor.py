# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor

# --- C++ Op Accessor Functions ---


def get_pack_op(weight_nbit: int):
    """Gets the C++ packing function from the 'torchao' namespace."""
    op_name = f"_pack_groupwise_{weight_nbit}bit_weight_with_lut"
    if not hasattr(torch.ops.torchao, op_name):
        raise NotImplementedError(f"Packing op for {weight_nbit}-bit not found.")
    return getattr(torch.ops.torchao, op_name)


def get_linear_op(weight_nbit: int):
    """Gets the C++ fused linear function from the 'torchao' namespace."""
    op_name = f"_linear_groupwise_{weight_nbit}bit_weight_with_lut"
    if not hasattr(torch.ops.torchao, op_name):
        raise NotImplementedError(f"Linear op for {weight_nbit}-bit not found.")
    return getattr(torch.ops.torchao, op_name)


aten = torch.ops.aten


class GroupwiseLutQuantizedTensor(TorchAOBaseTensor):
    """
    Corrected version that is robust for torch.export.
    """

    tensor_data_attrs = [
        "packed_weight",
    ]
    tensor_attributes = [
        "bit_width",
        "lut_group_size",
        "scale_group_size",
        "shape",
        "dtype",
    ]

    @staticmethod
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        bit_width: int,
        lut_group_size: int,
        scale_group_size: int,
        shape: torch.Size,
        dtype: torch.dtype,
    ):
        kwargs = {
            "device": packed_weight.device,
            "dtype": dtype,
            "layout": packed_weight.layout,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        packed_weight: torch.Tensor,
        bit_width: int,
        lut_group_size: int,
        scale_group_size: int,
        shape: torch.Size,
        dtype: torch.dtype,
    ):
        self.packed_weight = packed_weight
        self.bit_width = bit_width
        self.lut_group_size = lut_group_size
        self.scale_group_size = scale_group_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, "
            f"bit_width={self.bit_width}, lut_group_size={self.lut_group_size}, "
            f"scale_group_size={self.scale_group_size}, device={self.device})"
        )

    def __tensor_flatten__(self):
        metadata = [getattr(self, attr) for attr in self.tensor_attributes]
        return self.tensor_data_attrs, metadata

    @classmethod
    def __tensor_unflatten__(cls, tensors, metadata, outer_size, outer_stride):
        return cls(
            *[tensors[name] for name in cls.tensor_data_attrs],
            *metadata,
        )

    def _apply_fn_to_data(self, fn):
        new_packed_weight = fn(self.packed_weight)
        return self.__class__(
            new_packed_weight,
            self.bit_width,
            self.lut_group_size,
            self.scale_group_size,
            self.shape,
            self.dtype,
        )

    @classmethod
    def from_packed_data(
        cls,
        int_data: torch.Tensor,
        luts: torch.Tensor,
        scales: torch.Tensor,
        bit_width: int,
        lut_group_size: int,
        scale_group_size: int,
        original_shape: torch.Size,
        bias: Optional[torch.Tensor] = None,
        target: str = "auto",
    ):
        """
        A factory function that uses the C++ packing op to create an instance
        of the GroupwiseLutQuantizedTensor.
        """
        # 1. Get the correct C++ packing operator based on the bit width
        pack_op = get_pack_op(bit_width)

        # 2. Call the C++ op to get the single packed weight tensor
        packed_weight = pack_op(
            int_data,
            luts,
            scale_group_size,
            lut_group_size,
            scales,
            bias,
            target,
        )

        # 3. Construct and return the custom tensor object
        return cls(
            packed_weight,
            bit_width,
            lut_group_size,
            scale_group_size,
            original_shape,
            int_data.dtype,
        )


implements = GroupwiseLutQuantizedTensor.implements


@implements([F.linear])
def _(func, types, args, kwargs):
    """
    Override for `torch.nn.functional.linear`. This implementation calls the
    fused C++ kernel directly, avoiding a separate dequantization step.
    """
    input_tensor, weight_tensor, _ = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )

    # Get the correct C++ operator based on the bit width
    linear_op = get_linear_op(weight_tensor.bit_width)

    # --- Input Reshaping Logic ---
    #
    # The underlying C++ kernel (`linear_op`) is designed to compute a matrix multiplication on 2D tensors ONLY.
    # It assumes a simple (m, k) matrix layout.
    # We "flatten" the high-rank input into a 2D matrix that the C++ kernel understands, and then
    # "unflatten" the 2D output back to restore the original batch dimensions.

    # Store original shape to reshape the output later
    original_shape = input_tensor.shape
    k = weight_tensor.shape[1]
    # If input rank > 2, flatten all batch dimensions into one
    if input_tensor.dim() > 2:
        input_tensor = input_tensor.reshape(-1, k)

    # The 'n' dimension is the output feature dimension from the weight
    n = weight_tensor.shape[0]

    # Call the fused C++ linear operator
    output = linear_op(
        input_tensor,
        weight_tensor.packed_weight,
        weight_tensor.scale_group_size,
        weight_tensor.lut_group_size,
        n,
        k,
    )

    # Reshape the output to match the original batch dimensions
    if len(original_shape) > 2:
        output_shape = original_shape[:-1] + (n,)
        return output.reshape(output_shape)

    return output


@implements([aten.detach.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )
