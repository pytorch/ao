import functools
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import torch
from torch._dynamo.comptime import comptime

from torch.utils._python_dispatch import return_and_correct_aliasing
from .bitpacking import pack, unpack, numbits
from torchao.dtypes.utils import (
    _implements,
    _ATEN_OP_OR_TORCH_FN_TABLE,
    _register_layout_cls,
    _get_layout_tensor_constructor,
)


aten = torch.ops.aten
def implements(aten_ops_or_torch_fn):
    return _implements(IntxTensor, aten_ops_or_torch_fn)

def get_layout_tensor_constructor(layout_type_class: type(LayoutType)):
    return _get_layout_tensor_constructor(AffineQuantizedTensor, layout_type_class)

@dataclass(frozen=True)
class IntxLayoutType(LayoutType):
    bit_size: int
    pack_dim: int = -1
    
    def post_process(self, input: torch.Tensor) -> torch.Tensor:
        from torchao.prototype.intx import to_intx
        return to_intx(input, self.bit_size, self.pack_dim)

class IntxTensor(torch.Tensor):
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
        shards = [shard.to(torch.uint8) for shard in shards]
        self.shard = shards
        for i, atrib in enumerate(self.bits_to_shard[bit_size]):
            setattr(self, atrib, shards[i])
            
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
        shards = [i for i in tensor_data_dict.values()]
        packed_shape, bit_size, pack_dim = tensor_attributes
        return cls(shards, packed_shape, bit_size, pack_dim)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(
            f"IntxTensor dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self):
        return unpack(self.get_shards(), self.bit_size, dim = self.pack_dim)
    
    # temporary until kernels on packed tensors are created
    def apply_transformation(self, fn):
        og = self.get_plain()
        new = fn(og)
        return self.from_int(new, self.bit_size, self.pack_dim)
    
    @classmethod
    def from_int(cls, int_data: torch.Tensor, bit_size, pack_dim: int = -1):
        shards = pack(int_data, bit_size, dim=pack_dim)
        shape = list(int_data.shape)
        shape[pack_dim] = shape[pack_dim] * bit_size // 8
        return cls(shards, int_data.shape, bit_size, pack_dim)
    
    
@implements([aten.to.device])
def to(self, *args, **kwargs):
    return self.__class__(
        [shard.to(kwargs["device"]) for shard in self.get_shards()],
        self.packed_shape,
        self.bit_size,
        self.pack_dim
    )
@implements([aten.to.dtype])
def to(self, *args, **kwargs):
    return self.__class__(
        [shard.to(kwargs["dtype"]) for shard in self.get_shards()],
        self.packed_shape,
        self.bit_size,
        self.pack_dim
    )
@implements([aten.t.default])
def t(self,*args, **kwargs):
    tensor = args[0]
    new = tensor.__class__(
        [shard.t() for shard in tensor.get_shards()],
        tensor.packed_shape[::-1],
        tensor.bit_size,
        len(tensor.packed_shape) - tensor.pack_dim -1,
    )
    return return_and_correct_aliasing(func, args, kwargs, new)

@implements([aten.detach.default])
def detach(func, *args, **kwargs):
    tensor = args[0]
    new = tensor.__class__(
        [shard.detach() for shard in tensor.get_shards()],
        tensor.packed_shape,
        tensor.bit_size,
        tensor.pack_dim
    )
    return return_and_correct_aliasing(
        func, args, kwargs, new
    )
    
@implements([aten.view.default])
def view(func, *args, **kwargs):
    new = args[0].apply_transformation(lambda x: x.view(*args[1]))
    return return_and_correct_aliasing(
        func, args, kwargs, new
    )

@implements([aten.sub.Tensor, aten.sub_.Tensor])
def sub(func, *args, **kwargs):
    if func == aten.sub_.Tensor:
        return return_and_correct_aliasing(
            func, args, kwargs, args[0].apply_transformation(lambda x: x.view(*args[1:]))
        )
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].get_plain().to(torch.int32) - args[1]
    )

@implements([aten.mul_.Tensor, aten.mul.Tensor])
def mul(func, *args, **kwargs):
    if func == aten.mul_.Tensor:
        new = args[0].apply_transformation(lambda x: (x.to(torch.int8)*args[1:]).to(torch.uint8))
        return return_and_correct_aliasing(
            func, args, kwargs, new
        )
    return return_and_correct_aliasing(
        func, args, kwargs, args[0].get_plain() * args[1]
    )
    
@implements([aten._to_copy.default])
def _to_copy(func, *args, **kwargs):
    tensor = args[0]
    new = tensor.__class__(
        [shard.clone() for shard in tensor.get_shards()],
        tensor.packed_shape,
        tensor.bit_size,
        tensor.pack_dim
    )
    return return_and_correct_aliasing(
        func, args, kwargs, new
    )

# quantization api integrations
to_intx = IntxTensor.from_int

def intx_affine_weight_only(bit_size, group_size=64, pack_dim=-1):
    """
    Applies intx weight-only asymmetric per-group quantization to linear layers, using intx quantization where 
    x is the number of bits specified by the `nbits` argument
    """
    
    def apply_intx_weight_only_quant(weight):
        from torchao.quantization.quant_primitives import (
            MappingType,
            ZeroPointDomain,
            choose_qparams_affine,
            quantize_affine,
            dequantize_affine,
        )
        layout_type = IntxLayoutType(bit_size=bit_size, pack_dim=pack_dim) 
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        quant_min = 0
        quant_max = 2**bit_size - 1
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int32
        zero_point_domain = ZeroPointDomain.INT
        
        return to_affine_quantized(
            weight, mapping_type, block_size, torch.uint8, quant_min = quant_min,
            quant_max = quant_max, eps = eps, 
            zero_point_dtype=zero_point_dtype,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type,
        )
    
    return apply_intx_weight_only_quant