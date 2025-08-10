# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
    MappingType,
    choose_qparams_affine,
    dequantize_affine,
    quantize_affine,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
    fill_defaults,
)

__all__ = [
    "IntxUnpackedTensor",
]

aten = torch.ops.aten

_FLOAT_TYPES: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]


class IntxUnpackedTensor(TorchAOBaseTensor):
    """
    intx quantization with unpacked format.  Subbyte quantized data is represented as int8.
    Quantization is represented in a decomposed way.
    This format is inteded for torch.export use cases.

    Tensor Attributes:
        _data: int data for
        scale: (K/group_size, N) for 2D Tensor, (B, N, K/group_size) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N) for 2D Tensor, (B, N, K/group_size) for 3D Tensor, where B is batch size,
               dtype is the same as the original Tensor dtype

    Non-Tensor Attributes:
        block_size: the block size for quantization, representing the granularity, for example groupwise quantization will have block_size (1, group_size)
        shape: the shape of the original Tensor
    """

    tensor_data_attrs = ["int_data", "scale", "zero_point"]
    tensor_attributes = ["bit_width", "block_size"]

    def __new__(cls, int_data, scale, zero_point, bit_width, block_size=None):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data,
        scale,
        zero_point,
        bit_width,
        block_size: Optional[Tuple[int]] = None,
    ):
        # Check plain data and infer block_size from shapes
        if block_size is None:
            assert scale.ndim == int_data.ndim
            assert zero_point.ndim == int_data.ndim
            block_size = []
            for i in range(int_data.ndim):
                assert scale.shape[i] == zero_point.shape[i]
                n_blocks = scale.shape[i]
                assert int_data.shape[i] % n_blocks == 0
                block_size.append(int_data.shape[i] // n_blocks)
            block_size = tuple(block_size)
        else:
            assert len(block_size) == int_data.ndim
            n_blocks = []
            for i in range(len(block_size)):
                assert int_data.shape[i] % block_size[i] == 0
                n_blocks.append(int_data.shape[i] // block_size[i])
            scale = scale.reshape(*n_blocks)
            zero_point = zero_point.reshape(*n_blocks)

        assert block_size is not None
        assert isinstance(block_size, tuple)
        assert bit_width >= 1 and bit_width <= 8

        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point

        self.bit_width = bit_width
        self.block_size = block_size

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

    def __repr__(self):
        repr_fields = (
            self.tensor_data_attrs
            + self.tensor_attributes
            + ["shape", "device", "dtype", "require_grad"]
        )
        inner_repr = [f"{attr}={getattr(self, attr)}" for attr in repr_fields]
        inner_repr = ", ".join(inner_repr)
        return f"{self.__class__.__name__}({inner_repr}))"

    def _quantization_type(self):
        return f"bit_width={self.bit_width}, block_size={self.block_size}, shape={self.shape}, dtype={self.dtype}, device={self.device}"

    def _has_float_zero_point(self) -> bool:
        return self.zero_point.dtype in _FLOAT_TYPES

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        dtype = kwargs.pop("dtype")
        assert dtype in _FLOAT_TYPES
        return self.__class__(
            self.int_data.to(device),
            self.scale.to(device=device, dtype=dtype),
            self.zero_point.to(device=device, dtype=dtype)
            if self._has_float_zero_point()
            else self.zero_point.to(device),
            self.bit_width,
            self.block_size,
        )

    @classmethod
    def from_float(
        cls,
        float_tensor: torch.Tensor,
        block_size: Tuple[int],
        dtype: torch.dtype,
        *,
        mapping_type: MappingType = MappingType.SYMMETRIC,
    ):
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[dtype]
        bit_width = _DTYPE_TO_BIT_WIDTH[dtype]
        scale, zero_point = choose_qparams_affine(
            float_tensor,
            mapping_type,
            block_size,
            target_dtype=torch.int8,
            quant_min=qmin,
            quant_max=qmax,
        )
        int_data = quantize_affine(
            float_tensor,
            block_size,
            scale,
            zero_point,
            output_dtype=torch.int8,
            quant_min=qmin,
            quant_max=qmax,
        )
        return IntxUnpackedTensor(
            int_data=int_data,
            scale=scale,
            zero_point=zero_point,
            bit_width=bit_width,
            block_size=block_size,
        )

    def get_plain(self):
        return self.int_data, self.scale, self.zero_point

    def dequantize(self):
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[getattr(torch, f"int{self.bit_width}")]
        return dequantize_affine(
            self.int_data,
            self.block_size,
            self.scale,
            self.zero_point,
            torch.int8,
            qmin,
            qmax,
            output_dtype=self.dtype,
        )


implements = IntxUnpackedTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


@implements([torch.nn.functional.embedding, aten.embedding.default])
def _(func, types, args, kwargs):
    assert len(args) == 2
    indices, weight_tensor = (
        args[0],
        args[1],
    )
    weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.embedding(indices, weight_tensor, **kwargs)


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


def _same_metadata(self: "IntxUnpackedTensor", src: "IntxUnpackedTensor") -> bool:
    return (
        isinstance(self, IntxUnpackedTensor)
        and isinstance(src, IntxUnpackedTensor)
        and all(
            getattr(self, attr) == getattr(src, attr) for attr in self.tensor_attributes
        )
    )


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if _same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    raise ValueError(
        f"Not supported args for copy_ due to metadata mismatch: {args[0], args[1]}"
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    assert step == 1

    # Slicing must be compatible with the block size to make sense on the quantized tensor
    # In particular both start and end must be a multiple of block_size[dim]
    # Otherwise the sliced tensor cannot be represented as a IntxUnpackedTensor
    # For example, if block_size = 4, we might have:
    #
    # int_data: i i i i | i i i i
    #    scale: s s
    #
    # If we set start = 2 and end = 8, then the int_data slice is:
    #
    # int_data_slice: i i (i i | i i i i)
    #
    # But then the block_size for the first two int_data in the slice is 2
    # and remaining blocks have size 4.  This cannot be represented
    # with the metadata we store in an IntxUnpackedTensor, which requires uniform blocking

    assert start % self.block_size[dim] == 0, (
        f"slice args are incompatible with blocking: start={start} must be divisible by block_size[dim]={self.block_size[dim]}"
    )
    start_scale = start // self.block_size[dim]

    assert end % self.block_size[dim] == 0, (
        f"slice args are incompatible with blocking: end={end} must be divisible by block_size[dim]={self.block_size[dim]}"
    )
    end_scale = end // self.block_size[dim]

    int_data = aten.slice.Tensor(self.int_data, dim, start, end, step)
    scale = aten.slice.Tensor(self.scale, dim, start_scale, end_scale, step)
    zero_point = aten.slice.Tensor(self.zero_point, dim, start_scale, end_scale, step)

    new = self.__class__(
        int_data,
        scale,
        zero_point,
        self.bit_width,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)


IntxUnpackedTensor.__module__ = "torchao.quantization"

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with IntxUnpackedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([IntxUnpackedTensor])
