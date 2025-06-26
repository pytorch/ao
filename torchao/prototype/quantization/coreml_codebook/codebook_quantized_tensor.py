# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.uintx.uintx_layout import _DTYPE_TO_BIT_WIDTH, UintxTensor
from torchao.prototype.quantization.coreml_codebook.codebook_ops import (
    choose_qparams_and_quantize_codebook,
    dequantize_codebook,
)
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten


class CodebookQuantizedTensor(TorchAOBaseTensor):
    """
    Codebook quantized tensor subclass.

    Codebook (lookup table) quantization involves partitioning the input tensor into blocks, and replacing each block
    with the index of the closest entry in a predefined codebook.

    Fields:
      codes (torch.Tensor): Tensor of indices representing blocks in the original tensor. Each index
         maps to a corresponding codebook entry.
      codebook (torch.Tensor): Tensor representing the quantization codebook, where each entry
         corresponds to a block in the original tensor. Shape is `(codebook_size, out_block_size, in_block_size)`.
      block_size (Tuple[int, ...]): Granularity of quantization, specifying the dimensions of tensor
         blocks that share the same quantization parameters.
      shape (torch.Size): Shape of the original high-precision tensor.
      dtype (torch.dtype): dtype of the original high-precision tensor.
    """

    @staticmethod
    def __new__(
        cls,
        codes: torch.Tensor,
        codebook: torch.Tensor,
        group_size: int,
        shape: torch.Size,
        dtype=None,
    ):
        kwargs = {}
        kwargs["device"] = codes.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else codes.layout
        )
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        codes: torch.Tensor,
        codebook: torch.Tensor,
        group_size: int,
        shape: torch.Size,
        dtype=None,
    ):
        self.codes = codes
        self.codebook = codebook
        self.group_size = group_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(codes={self.codes}, codebook={self.codebook}, group_size={self.group_size} "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, group_size={self.group_size}, codebook_size={self.codebook.size(0)}, device={self.device}, code_dtype={self.codes.dtype}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        if isinstance(self.codes, UintxTensor):
            codes = self.codes.get_plain()
        else:
            codes = self.codes
        if codes.dtype != torch.int32:
            # TODO: Investigate and support not casting to torch.int32 for indexing to improve performance
            codes = codes.to(torch.int32)
        return dequantize_codebook(
            codes,
            self.codebook,
            self.group_size,
            output_dtype=output_dtype,
        )

    def __tensor_flatten__(self):
        return ["codes", "codebook"], [
            self.group_size,
            self.shape,
            self.dtype,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        codes = tensor_data_dict["codes"]
        codebook = tensor_data_dict["codebook"]
        group_size, shape, dtype = tensor_attributes
        return cls(
            codes,
            codebook,
            group_size,
            shape if outer_size is None else outer_size,
            dtype=dtype,
        )

    @classmethod
    def from_float(
        cls,
        input_tensor: torch.Tensor,
        code_dtype: torch.dtype,
        group_size: int,
    ):
        """
        Creates a CodebookQuantizedTensor from a floating-point tensor by performing codebook quantization.

        Args:
            input_tensor (torch.Tensor): The input floating-point tensor to quantize.
            code_dtype (torch.dtype): The dtype of the codes.
            chunk_size (int): The chunk size to use during quantization (to control memory usage).
        """

        codebook, codes = choose_qparams_and_quantize_codebook(
            input_tensor.to(torch.float32), code_dtype, group_size
        )

        if code_dtype in _DTYPE_TO_BIT_WIDTH:
            codes = UintxTensor.from_uint8(codes, dtype=code_dtype)

        codebook = codebook.to(input_tensor.dtype)

        return cls(
            codes,
            codebook,
            group_size,
            input_tensor.shape,
            dtype=input_tensor.dtype,
        )

    def to(self, *args, **kwargs):
        # I'm not sure if this is right
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.codes.to(device),
            self.codebook.to(device),
            self.group_size,
            self.shape,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        # Apply function to only codes?
        return self.__class__(
            fn(self.codes),
            fn(self.codebook),
            self.group_size,
            self.shape,
            dtype=self.dtype,
        )


implements = CodebookQuantizedTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    weight_tensor = weight_tensor.dequantize()
    return func(input_tensor, weight_tensor, bias)


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )
