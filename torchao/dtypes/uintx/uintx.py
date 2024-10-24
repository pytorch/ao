from typing import Tuple, List
from dataclasses import dataclass
import torch

from torch.utils._python_dispatch import return_and_correct_aliasing
from .bitpacking import pack, unpack
from torchao.dtypes.utils import (
    Layout,
)
from torchao.utils import TorchAOBaseTensor
from torchao.dtypes.affine_quantized_tensor import PlainAQTTensorImpl, register_layout
from torchao.utils import TORCH_VERSION_AT_LEAST_2_3
from torchao.utils import fill_defaults

aten = torch.ops.aten

# Note: Uintx does not work for torch 2.3 and below
_DTYPE_TO_BIT_WIDTH = {}
_BIT_WIDTH_TO_DTYPE = {}

if TORCH_VERSION_AT_LEAST_2_3:
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
else:
    print("uintx feature requires torch 2.3+, please upgrade pytorch")


class UintxTensor(TorchAOBaseTensor):
    """
    Splits int data into packed shards based on bit size
    fields:
      int4_shard (torch.Tensor): 4 bit packed shard
      int2_shard (torch.Tensor): 2 bit packed shard
      int1_shard (torch.Tensor): 1 bit packed shard
      bit_width (int): number of bits for each element
      pack_dim: (int) dimension to pack along
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
        original_size: int,
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
        original_size: int,
        packed_shape: List[int],
        bit_width: int,
        pack_dim: int = -1,
    ):
        for i, attrib in enumerate(self.bits_to_shard[bit_width]):
            setattr(self, attrib, shards[i])

        self.packed_shape = packed_shape
        self.bit_width = bit_width
        self.pack_dim = pack_dim
        self.original_size = original_size

    def get_shards(self):
        return [getattr(self,i) for i in self.__class__.bits_to_shard[self.bit_width]]

    def __repr__(self):
        return f"Int{self.bit_width}Tensor(shape = {self.packed_shape}, data = {unpack(self.get_shards(), self.bit_width, self.original_size, dim = self.pack_dim)})"

    def __tensor_flatten__(self):
        return self.__class__.bits_to_shard[self.bit_width], [self.packed_shape, self.bit_width, self.pack_dim]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        shards =  list(tensor_data_dict.values())
        packed_shape, bit_width, pack_dim = tensor_attributes
        return cls(shards, packed_shape, bit_width, pack_dim)

    def get_plain(self):
        return unpack(self.get_shards(), original_size=self.original_size, elem_size=self.bit_width, dim = self.pack_dim)

    # temporary until kernels on packed tensors are created
    def apply_transformation(self, fn):
        og = self.get_plain()
        new = fn(og)
        if new.dim() == 0:
            new = new.unsqueeze(0)
        dtype = _BIT_WIDTH_TO_DTYPE[self.bit_width]
        return self.from_uint8(new, dtype, self.pack_dim)

    # temporary until kernels on packed tensors are created
    def apply_fn_to_shards(self, fn):
        new_shards = [fn(shard) for shard in self.get_shards()]
        return self.__class__(new_shards, self.original_size, self.packed_shape, self.bit_width, self.pack_dim)

    @classmethod
    def from_uint8(cls, int_data: torch.Tensor, dtype: torch.dtype, pack_dim: int = -1):
        assert dtype in _DTYPE_TO_BIT_WIDTH.keys(), "Expected dtype to be one of {_DTYPE_TO_BIT_WIDTH.keys()}"
        bit_width = _DTYPE_TO_BIT_WIDTH[dtype]
        shards, original_size = pack(int_data, bit_width, dim=pack_dim)
        shape = list(int_data.shape)
        shape[pack_dim] = shape[pack_dim] * bit_width // 8
        return cls(shards, original_size, int_data.shape, bit_width, pack_dim)


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
    
    def __setitem__(self, index, value):
        if isinstance(value, UintxTensor):
            value = value.get_plain()
        elif isinstance(value, torch.Tensor):
            if value.dtype != torch.uint8:
                raise ValueError(f"Expected uint8 tensor, got {value.dtype}")
        else:
            value = torch.tensor(value, dtype=torch.uint8, device=self.device)

        plain_data = self.get_plain()
        plain_data[index] = value
        new_packed = self.from_uint8(plain_data, _BIT_WIDTH_TO_DTYPE[self.bit_width], self.pack_dim)

        # Update the shards
        for shard_name in self.bits_to_shard[self.bit_width]:
            setattr(self, shard_name, getattr(new_packed, shard_name))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.Tensor.__setitem__:
            self, index, value = args
            self.__setitem__(index, value)
            return None
        return super().__torch_function__(func, types, args, kwargs)

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
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]
    )

@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]
    )

@implements(aten.sub.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].apply_transformation(lambda x: (x - args[1]).to(torch.uint8))
    )

@implements(aten.mul.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].apply_transformation(lambda x: (x * args[1]).to(torch.uint8))
    )

@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    # assert step == 1
    if end >= self.shape[dim]:
        end = self.shape[dim]
    shape = list(self.shape)
    shape[dim] = end - start
    
    def slice_fn(x):
        return torch.ops.aten.slice.Tensor(x, dim, start, end, step)

    return return_and_correct_aliasing(
        func, args, kwargs, self.apply_transformation(slice_fn)
    )

@implements(aten.select.int)
def _(func, types, args, kwargs):
    self, dim, index = args
    
    def select_fn(x):
        selected = x.select(dim, index)
        return selected.unsqueeze(dim)
    
    return return_and_correct_aliasing(
        func, args, kwargs, self.apply_transformation(select_fn)
    )

@implements(aten.set_)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].apply_transformation(lambda x: x.set_(args[1]))
    )

# quantization api integrations
to_uintx = UintxTensor.from_uint8

@dataclass(frozen=True)
class UintxLayout(Layout):
    dtype: torch.dtype
    pack_dim: int = -1

    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        return to_uintx(input, self.dtype, self.pack_dim)

@register_layout(UintxLayout)
class UintxAQTTensorImpl(PlainAQTTensorImpl):

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
