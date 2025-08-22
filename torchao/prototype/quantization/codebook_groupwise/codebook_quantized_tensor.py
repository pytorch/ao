# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.prototype.quantization.codebook_coreml.codebook_quantized_tensor import (
    CodebookQuantizedTensor,
)
from torchao.prototype.quantization.codebook_utils.codebook_utils import (
    block_shape_to_group_size,
)
from torchao.quantization.quant_primitives import _DTYPE_TO_BIT_WIDTH
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


class CodebookQuantizedPackedTensor(TorchAOBaseTensor):
    tensor_data_names = [
        "packed_weight",
    ]
    tensor_attribute_names = [
        "bit_width",
        "lut_block_size",
        "scale_block_size",
        "shape",
        "dtype",
    ]

    def __new__(
        cls, packed_weight, bit_width, lut_block_size, scale_block_size, shape, dtype
    ):
        kwargs = {
            "device": packed_weight.device,
            "dtype": dtype,
            "requires_grad": False,
        }
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        packed_weight: torch.Tensor,
        bit_width: int,
        lut_block_size: List[int],
        scale_block_size: Optional[List[int]],
        shape: torch.Size,
        dtype: torch.dtype,
    ):
        self.packed_weight = packed_weight
        self.bit_width = bit_width
        self.lut_block_size = lut_block_size
        self.scale_block_size = scale_block_size

    @classmethod
    def from_unpacked(
        cls,
        int_data: torch.Tensor,
        luts: torch.Tensor,
        scales: Optional[torch.Tensor],
        bit_width: int,
        lut_block_size: List[int],
        scale_block_size: Optional[List[int]],
        original_shape: torch.Size,
        bias: Optional[torch.Tensor] = None,
    ):
        lut_group_size = block_shape_to_group_size(lut_block_size, int_data.shape)

        if scale_block_size is not None and scales is not None:
            # Scales are present, calculate group size
            scale_group_size = block_shape_to_group_size(
                scale_block_size, int_data.shape
            )
            scales_arg = scales
        else:
            # Scales are not present, provide safe defaults
            scale_group_size = -1
            scales_arg = torch.empty(0, dtype=luts.dtype, device=luts.device)

        pack_op = get_pack_op(bit_width)
        packed_weight = pack_op(
            int_data, luts, scale_group_size, lut_group_size, scales_arg, bias
        )
        return cls(
            packed_weight,
            bit_width,
            lut_block_size,
            scale_block_size,
            original_shape,
            int_data.dtype,
        )

    @classmethod
    def from_codebook_quantized_tensor(
        cls,
        tensor: CodebookQuantizedTensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Factory method to create a packed tensor from a CodebookQuantizedTensor.

        This method takes the general components of a codebook-quantized tensor
        (codes, codebook, etc.) and uses a specialized 'pack_op' to fuse them
        into a single, efficient tensor format suitable for high-performance
        inference kernels.
        """
        lut_block_size = tensor.block_size
        lut_group_size = block_shape_to_group_size(lut_block_size, tensor.shape)

        # CoreML quantization scheme does not use scales, so they are disabled.
        scale_group_size = -1
        scales = None

        bit_width = _DTYPE_TO_BIT_WIDTH[tensor.code_dtype]
        # Retrieve the appropriate packing C++/CUDA kernel for the given bit width.
        pack_op = get_pack_op(bit_width)

        # Ensure the codebook (Look-Up Table) is in float32, as this is the
        # data type expected by the underlying packing kernel.
        codebook = tensor.codebook.to(torch.float32)

        # --- Explanation for .squeeze() ---
        # The input `tensor.codebook` is often stored in a 4D format, such as
        # [1, num_groups, 256, 1], for compatibility with generic operators like
        # the dequantize function. However, the specialized `pack_op` expects a
        # more compact 2D LUT of shape [num_groups, 256].
        # The .squeeze() operation removes the unnecessary singleton (size 1)
        # dimensions to achieve this required 2D format.
        codebook = codebook.squeeze()

        # Call the packing operator to create the final fused tensor.
        packed_weight = pack_op(
            tensor.codes, codebook, scale_group_size, lut_group_size, scales, bias, None
        )

        # Return a new instance of this class containing the final packed weight
        # and its associated quantization metadata.
        return cls(
            packed_weight, bit_width, lut_block_size, None, tensor.shape, tensor.dtype
        )


implements = CodebookQuantizedPackedTensor.implements


@implements([F.linear])
def _(func, types, args, kwargs):
    """
    Override for `torch.nn.functional.linear` specifically for the
    GroupwiseLutQuantizedTensor. This calls the fused C++ kernel.
    """
    input_tensor, weight_tensor, _ = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    linear_op = get_linear_op(weight_tensor.bit_width)
    lut_group_size = block_shape_to_group_size(
        weight_tensor.lut_block_size, weight_tensor.shape
    )
    original_shape = input_tensor.shape
    k = weight_tensor.shape[1]
    if input_tensor.dim() > 2:
        input_tensor = input_tensor.reshape(-1, k)

    n = weight_tensor.shape[0]
    output = linear_op(
        input_tensor, weight_tensor.packed_weight, -1, lut_group_size, n, k
    )

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
