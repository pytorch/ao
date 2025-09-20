# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.prototype.quantization.codebook_coreml.codebook_ops import (
    choose_qparams_and_quantize_codebook_coreml,
    dequantize_codebook,
)
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
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
         maps to a corresponding codebook entry, torch.uint8 dtype.
      codebook (torch.Tensor): Tensor representing the quantization codebook, where each entry
         corresponds to a block in the original tensor. Shape is `(codebook_size, out_block_size, in_block_size)`.
      code_dtype (torch.dtype): The logical dtype for the codes, [torch.uint1, ..., torch.uint8]
         Note that codes is stored in torch.uint8, this is just addtional information for dequantize op
      block_size (Tuple[int, ...]): Granularity of quantization, specifying the dimensions of tensor
         blocks that share the same quantization parameters.
      shape (torch.Size): Shape of the original high-precision tensor.
      dtype (torch.dtype): dtype of the original high-precision tensor.
    """

    tensor_data_attrs = ["codes", "codebook"]
    tensor_attributes = ["code_dtype", "block_size", "shape", "dtype"]

    @staticmethod
    def __new__(
        cls,
        codes: torch.Tensor,
        codebook: torch.Tensor,
        code_dtype: torch.dtype,
        block_size: List[int],
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
        code_dtype: torch.dtype,
        block_size: List[int],
        shape: torch.Size,
        dtype=None,
    ):
        self.codes = codes
        self.codebook = codebook
        self.code_dtype = code_dtype
        self.block_size = block_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(codes={self.codes}, codebook={self.codebook}, code_dtype={self.code_dtype}, block_size={self.block_size} "
            f"shape={self.shape}, device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def _quantization_type(self):
        return f"shape={self.shape}, codebook_shape={self.codebook.shape}, code_dtype={self.code_dtype}, block_size={self.block_size}, device={self.device}"

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        codes = self.codes
        if codes.dtype != torch.int32:
            # TODO: Investigate and support not casting to torch.int32 for indexing to improve performance
            codes = codes.to(torch.int32)

        # Note: code_dtype is just for lowering pass to understand the range of values in codes
        return dequantize_codebook(
            codes,
            self.codebook,
            _DTYPE_TO_BIT_WIDTH[self.code_dtype],
            self.block_size,
            output_dtype=output_dtype,
        )

    def __tensor_flatten__(self):
        return self.tensor_data_attrs, [
            getattr(self, attr) for attr in self.tensor_attributes
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_data_attrs],
            *tensor_attributes,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            *[fn(getattr(self, attr)) for attr in self.tensor_data_attrs],
            *[getattr(self, attr) for attr in self.tensor_attributes],
        )

    @classmethod
    def from_float(
        cls,
        input_tensor: torch.Tensor,
        code_dtype: torch.dtype,
        block_size: List[int],
    ):
        """
        Creates a CodebookQuantizedTensor from a floating-point tensor by performing codebook quantization.

        Args:
            input_tensor (torch.Tensor): The input floating-point tensor to quantize.
            code_dtype (torch.dtype): The dtype of the codes, Note the codes Tensor is stored in uint8
            chunk_size (int): The chunk size to use during quantization (to control memory usage).
        """
        codebook, codes = choose_qparams_and_quantize_codebook_coreml(
            input_tensor, code_dtype, block_size
        )

        assert codes.dtype == torch.uint8, "Only support using uint8 for codes for now"

        return cls(
            codes,
            codebook,
            code_dtype,
            block_size,
            input_tensor.shape,
            dtype=input_tensor.dtype,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            *[getattr(self, attr).to(device) for attr in self.tensor_data_attrs],
            *[getattr(self, attr) for attr in self.tensor_attributes],
            **kwargs,
        )


implements = CodebookQuantizedTensor.implements
implements_torch_function = CodebookQuantizedTensor.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    weight_tensor = weight_tensor.dequantize()
    return func(input_tensor, weight_tensor, bias)


@implements([aten.embedding.default])
@implements_torch_function([torch.nn.functional.embedding])
def _(func, types, args, kwargs):
    assert len(args) == 2
    indices, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor = weight_tensor.dequantize()
    return func(indices, weight_tensor, **kwargs)


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
