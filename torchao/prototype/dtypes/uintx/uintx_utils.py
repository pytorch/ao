# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities copied from the deleted uintx_layout.py to support
autoround and codebook features that still depend on UintxTensor/UintxLayout.
These should be removed once autoround and codebook are migrated to the new
quantization design.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.uintx.bitpacking import pack, unpack
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

_DTYPE_TO_BIT_WIDTH = {
    torch.uint1: 1,
    torch.uint2: 2,
    torch.uint3: 3,
    torch.uint4: 4,
    torch.uint5: 5,
    torch.uint6: 6,
    torch.uint7: 7,
}

_BIT_WIDTH_TO_DTYPE = {v: k for k, v in _DTYPE_TO_BIT_WIDTH.items()}


class UintxTensor(TorchAOBaseTensor):
    """
    Splits int data into packed shards based on bit size.
    """

    bits_to_shard = {
        1: ["int1_shard"],
        2: ["int2_shard"],
        3: ["int2_shard", "int1_shard"],
        4: ["int4_shard"],
        5: ["int4_shard", "int1_shard"],
        6: ["int4_shard", "int2_shard"],
        7: ["int4_shard", "int2_shard", "int1_shard"],
    }

    def __new__(
        cls,
        shards: List[torch.Tensor],
        packed_shape: List[int],
        bit_width: int,
        pack_dim: int = -1,
    ):
        kwargs = {"device": shards[0].device}
        kwargs["device"] = shards[0].device
        kwargs["layout"] = shards[0].layout
        kwargs["requires_grad"] = False
        kwargs["dtype"] = torch.uint8
        return torch.Tensor._make_wrapper_subclass(cls, packed_shape, **kwargs)

    def __init__(
        self,
        shards: List[torch.Tensor],
        packed_shape: List[int],
        bit_width: int,
        pack_dim: int = -1,
    ):
        for i, attrib in enumerate(self.bits_to_shard[bit_width]):
            setattr(self, attrib, shards[i])

        self.packed_shape = packed_shape
        self.bit_width = bit_width
        self.pack_dim = pack_dim

    def get_shards(self):
        return [getattr(self, i) for i in self.__class__.bits_to_shard[self.bit_width]]

    def __repr__(self):
        return f"Int{self.bit_width}Tensor(shape = {self.packed_shape}, data = {unpack(self.get_shards(), self.bit_width, dim=self.pack_dim)})"

    def __tensor_flatten__(self):
        return self.__class__.bits_to_shard[self.bit_width], [
            self.packed_shape,
            self.bit_width,
            self.pack_dim,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        shards = list(tensor_data_dict.values())
        packed_shape, bit_width, pack_dim = tensor_attributes
        return cls(shards, packed_shape, bit_width, pack_dim)

    def get_plain(self):
        return unpack(self.get_shards(), self.bit_width, dim=self.pack_dim)

    def apply_transformation(self, fn):
        og = self.get_plain()
        new = fn(og)
        dtype = _BIT_WIDTH_TO_DTYPE[self.bit_width]
        return self.from_uint8(new, dtype, self.pack_dim)

    def apply_fn_to_shards(self, fn):
        new_shards = [fn(shard) for shard in self.get_shards()]
        return self.__class__(
            new_shards, self.packed_shape, self.bit_width, self.pack_dim
        )

    @classmethod
    def from_uint8(cls, int_data: torch.Tensor, dtype: torch.dtype, pack_dim: int = -1):
        assert dtype in _DTYPE_TO_BIT_WIDTH.keys(), (
            "Expected dtype to be one of {_DTYPE_TO_BIT_WIDTH.keys()}"
        )
        bit_width = _DTYPE_TO_BIT_WIDTH[dtype]
        shards = pack(int_data, bit_width, dim=pack_dim)
        shape = list(int_data.shape)
        shape[pack_dim] = shape[pack_dim] * bit_width // 8
        return cls(shards, int_data.shape, bit_width, pack_dim)

    def _get_to_kwargs(self, *args, **kwargs):
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        memory_format = (
            memory_format if memory_format is not None else torch.preserve_format
        )
        kwargs = {
            "device": device,
            "dtype": dtype,
            "memory_format": memory_format,
        }
        return kwargs

    def to(self, *args, **kwargs):
        if "copy" in kwargs:
            return super().to(*args, **kwargs)
        kwargs = self._get_to_kwargs(*args, **kwargs)
        if "device" in kwargs:
            return self.__class__(
                list(shard.to(kwargs["device"]) for shard in self.get_shards()),
                self.packed_shape,
                self.bit_width,
                self.pack_dim,
            )
        return super().to(*args, **kwargs)


implements = UintxTensor.implements


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].apply_fn_to_shards(torch.detach)
    )


@implements(aten.view.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].apply_transformation(lambda x: x.view(*args[1:]))
    )


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(func, args, kwargs, args[0])


@implements(aten.sub.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].apply_transformation(lambda x: (x - args[1]).to(torch.uint8)),
    )


@implements(aten.mul.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].apply_transformation(lambda x: (x * args[1]).to(torch.uint8)),
    )


to_uintx = UintxTensor.from_uint8


@dataclass(frozen=True)
class UintxLayout(Layout):
    dtype: torch.dtype
    pack_dim: int = -1

    def post_process(
        self,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return to_uintx(input, self.dtype, self.pack_dim), scale, zero_point


# TODO: migrate autoround to not use UintxLayout/AffineQuantizedTensor,
# then remove UintxAQTTensorImpl and the @register_layout registration below.
from torchao.dtypes.affine_quantized_tensor import register_layout


@register_layout(UintxLayout)
class UintxAQTTensorImpl(AQTTensorImpl):
    """Minimal AQTTensorImpl for UintxLayout, inlined from the deleted PlainAQTTensorImpl."""

    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        if self.zero_point is None:
            return ["int_data", "scale"], [self._layout]
        return ["int_data", "scale", "zero_point"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = (
            tensor_data_dict["int_data"],
            tensor_data_dict["scale"],
            tensor_data_dict.get("zero_point", None),
        )
        (_layout,) = tensor_attributes
        return cls(int_data, scale, zero_point, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"])
            if self.zero_point is not None
            else None,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point) if self.zero_point is not None else None,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        elif func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        raise NotImplementedError(
            f"UintxAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.int_data.get_plain(), self.scale, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):
        assert isinstance(_layout, UintxLayout)
        return cls(int_data, scale, zero_point, _layout)
