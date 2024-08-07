import functools
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import torch
from torch._dynamo.comptime import comptime

from torch.utils._python_dispatch import return_and_correct_aliasing
from .bitpacking import pack, unpack, numbits
from torchao.dtypes.utils import (
    LayoutType,
    _implements,
    _register_layout_cls,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
)
from torchao.dtypes.affine_quantized_tensor import PlainAQTLayout, register_layout_cls


aten = torch.ops.aten

class UintxTensor(torch.Tensor):
    """
    Splits int data into packed shards based on bit size
    fields:
      int4_shard (torch.Tensor): 4 bit packed shard
      int2_shard (torch.Tensor): 2 bit packed shard
      int1_shard (torch.Tensor): 1 bit packed shard
      bit_size (int): element size in bits
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
        packed_shape: List[int],
        bit_size: int,
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
        bit_size: int,
        pack_dim: int = -1,
    ):
        for i, attrib in enumerate(self.bits_to_shard[bit_size]):
            setattr(self, attrib, shards[i])
            
        self.packed_shape = packed_shape
        self.bit_size = bit_size    
        self.pack_dim = pack_dim
    
    def get_shards(self):
        return [getattr(self,i) for i in self.__class__.bits_to_shard[self.bit_size]]
    
    def __repr__(self):
        return f"Int{self.bit_size}Tensor(shape = {self.packed_shape}, data = {unpack(self.get_shards(), self.bit_size, dim = self.pack_dim)})"
    
    def __tensor_flatten__(self):
        return self.__class__.bits_to_shard[self.bit_size], [self.packed_shape, self.bit_size, self.pack_dim]
    
    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        shards =  list(tensor_data_dict.values())
        packed_shape, bit_size, pack_dim = tensor_attributes
        return cls(shards, packed_shape, bit_size, pack_dim)

    implements = classmethod(_implements)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)
    __torch_function__ = classmethod(_dispatch__torch_function__)

    def get_plain(self):
        return unpack(self.get_shards(), self.bit_size, dim = self.pack_dim)
    
    # temporary until kernels on packed tensors are created
    def apply_transformation(self, fn):
        og = self.get_plain()
        new = fn(og)
        return self.from_uint8(new, self.bit_size, self.pack_dim)
    
    # temporary until kernels on packed tensors are created
    def apply_fn_to_shards(self, fn):
        new_shards = [fn(shard) for shard in self.get_shards()]
        return self.__class__(new_shards, self.packed_shape, self.bit_size, self.pack_dim)
    
    @classmethod
    def from_uint8(cls, int_data: torch.Tensor, bit_size, pack_dim: int = -1):
        shards = pack(int_data, bit_size, dim=pack_dim)
        shape = list(int_data.shape)
        shape[pack_dim] = shape[pack_dim] * bit_size // 8
        return cls(shards, int_data.shape, bit_size, pack_dim)


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
# quantization api integrations
to_uintx = UintxTensor.from_uint8

@dataclass(frozen=True)
class UintxLayoutType(LayoutType):
    bit_size: int
    pack_dim: int = -1
    
    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        return to_uintx(input, self.bit_size, self.pack_dim)

@register_layout_cls(UintxLayoutType)
class UintxAQTLayout(PlainAQTLayout):
    
    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.int_data.get_plain(), self.scale, self.zero_point
    
    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, UintxLayoutType)
        return cls(int_data, scale, zero_point, layout_type)
    

def uintx_affine_weight_only(bit_size, group_size=64, pack_dim=-1):
    """
    Applies uintx weight-only asymmetric per-group quantization to linear layers, using uintx quantization where 
    x is the number of bits specified by the `nbits` argument
    """
    from torchao.quantization.quant_primitives import (
            MappingType,
            ZeroPointDomain,
            choose_qparams_affine,
            quantize_affine,
            dequantize_affine,
        )
    from torchao.dtypes import to_affine_quantized
    from torchao.quantization.quant_api import _get_linear_subclass_inserter
    def apply_uintx_weight_only_quant(weight):
        
        layout_type = UintxLayoutType(bit_size=bit_size, pack_dim=pack_dim) 
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        quant_min = 0
        quant_max = 2**bit_size - 1
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT
        
        return to_affine_quantized(
            weight, mapping_type, block_size, torch.uint8, 
            quant_min = quant_min, quant_max = quant_max, 
            eps = eps, zero_point_dtype=zero_point_dtype,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type,
        )
    
    return _get_linear_subclass_inserter(apply_uintx_weight_only_quant)