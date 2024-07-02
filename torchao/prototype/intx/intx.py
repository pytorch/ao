import functools
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import torch

from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_primitives import choose_qparams_affine, MappingType
from torchao.prototype.intx.bitpacking import pack, unpack, numbits
from torchao.dtypes.utils import (
    _implements,
    _ATEN_OP_OR_TORCH_FN_TABLE,
    _register_layout_cls,
    _get_layout_tensor_constructor,
)
from torchao.quantization.quant_primitives import (
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
    ZeroPointDomain,
    MappingType,
    int_scaled_matmul,
)
import pdb
aten = torch.ops.aten

class IntxLayout(torch.Tensor):
    """
    Base class for the layout tensor for `IntxTensor`
    """
    # this should be set for each layout class during registration
    extended_layout: Optional[str] = None

    def get_plain() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bit_size: int,
    ):
        pass

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
    

class IntxTensor(torch.Tensor):
    """
    Quantized tensor subclass for sub byte quantization (1-7 bits) with affine assymetric quantization
    
    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    fields:
      layout_tensor (AQTLayout): tensor that serves as a general layout storage for the quantized data,
         e.g. storing plain tensors (int_data, scale, zero_point) or packed formats depending on device
         and operator/kernel
      block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
         e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      shape (torch.Size): the shape for the Tensor
      quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      zero_point_domain (ZeroPointDomain): the domain that zero_point is in, should be eitehr integer or float
        if zero_point is in integer domain, zero point is added to the quantized integer value during
        quantization
        if zero_point is in floating point domain, zero point is subtracted from the floating point (unquantized)
        value during quantization
        default is ZeroPointDomain.INT
      dtype: dtype for external representation of the tensor, e.g. torch.float32
    """

    @staticmethod
    def __new__(
        cls,
        layout_tensor: IntxLayout,
        block_size: Tuple[int, ...],
        bit_size: int,
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        kwargs = {}
        kwargs["device"] = layout_tensor.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else layout_tensor.layout
        )
        kwargs["dtype"] = dtype
        if strides is not None:
            kwargs["strides"] = strides
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        layout_tensor: IntxLayout,
        block_size: Tuple[int, ...],
        bit_size: int,
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        self.layout_tensor = layout_tensor
        self.block_size = block_size
        self.bit_size = bit_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.zero_point_domain = zero_point_domain

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self, output_dtype=torch.float16):
        int_data, scale, zero_point = self.layout_tensor.get_plain()
        return dequantize_affine(int_data, self.block_size, scale, zero_point, int_data.dtype, self.quant_min, self.quant_max, self.zero_point_domain, output_dtype=output_dtype)

    def __tensor_flatten__(self):
        return ["layout_tensor"], [self.block_size, self.bit_size, self.shape,  self.quant_min, self.quant_max, self.zero_point_domain, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        layout_tensor = tensor_data_dict["layout_tensor"]
        block_size, bit_size, shape, quant_min, quant_max, zero_point_domain, dtype = tensor_attributes
        return cls(
            layout_tensor, block_size, bit_size,
            shape if outer_size is None else outer_size,
            quant_min, quant_max, zero_point_domain,
            dtype=dtype, strides=outer_stride,
        )

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        bit_size: int,
        pack_dim: Optional[int] = -1,
        quant_min: Optional[int] = None,
        quant_max: Optional[int]  = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        extended_layout: str = "packed",
        
    ):
        original_shape = input_float.shape
        quant_min = 0 if quant_min is None else quant_min
        quant_max = 2**bit_size-1 if quant_max is None else quant_max
        target_dtype = torch.uint8
        scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)
        
        layout_cls_ctr = get_layout_tensor_constructor(extended_layout)
        # TODO: this is temporary, need to come up with the proper UX
        if extended_layout == "plain":
            layout_tensor = layout_cls_ctr(int_data, scale, zero_point)
        elif extended_layout == "packed": 
            layout_tensor = layout_cls_ctr(int_data, scale, zero_point, bit_size, pack_dim)
        else:
            raise NotImplementedError(f"Only 'packed' or 'plain' layout is currently implemented")
        return cls(
            layout_tensor,
            block_size,
            bit_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype
        )

    @property
    def extended_layout(self) -> str:
        return self.layout_tensor.extended_layout

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


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
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.layout_tensor.to(kwargs["device"]),
            self.block_size,
            self.bit_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.layout_tensor),
            self.block_size,
            self.bit_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            dtype=self.dtype,
            strides=self.stride(),
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(
            f"IntxTensor dispatch: attempting to run {func}, this is not supported"
        )
        
        
def implements(aten_ops_or_torch_fn):
    return _implements(IntxTensor, aten_ops_or_torch_fn)

def register_layout_cls(extended_layout: str):
    return _register_layout_cls(IntxTensor, extended_layout)

def get_layout_tensor_constructor(extended_layout: str):
    return _get_layout_tensor_constructor(IntxTensor, extended_layout)


@register_layout_cls("packed")
class PackedTensorLayout(IntxLayout):
    """
    Splits int data into packed shards based on bit size
    fields:
      int4_shard (torch.Tensor): 4 bit packed shard
      int2_shard (torch.Tensor): 2 bit packed shard
      int1_shard (torch.Tensor): 1 bit packed shard
      
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
      bit_size (int): element size in bits
    """
    def __new__(
        cls,
        int4_shard: torch.Tensor,
        int2_shard: torch.Tensor,
        int1_shard: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        packed_shape: List[int],
        bit_size: int,
        pack_dim: int,
    ):
        kwargs = {}
        kwargs["device"] = scale.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else scale.layout
        )
        kwargs["dtype"] = torch.int8
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, packed_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int4_shard: torch.Tensor,
        int2_shard: torch.Tensor,
        int1_shard: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        packed_shape: List[int],
        bit_size: int,
        pack_dim: int,
    ):
        self.int4_shard = int4_shard
        self.int2_shard = int2_shard
        self.int1_shard = int1_shard
        self.scale = scale
        self.zero_point = zero_point
        self.packed_shape = packed_shape
        self.bit_size = bit_size
        self.pack_dim = pack_dim
    def __repr__(self):
        return f"{self.__class__.__name__}Int{self.bit_size}{self.packed_shape}"
    def __tensor_flatten__(self):
        tensor_data = ["scale", "zero_point"]
        if self.int4_shard is not None:
            tensor_data.append("int4_shard")
        if self.int2_shard is not None:
            tensor_data.append("int2_shard")
        if self.int1_shard is not None:
            tensor_data.append("int1_shard")
        return tensor_data, [self.packed_shape, self.bit_size, self.pack_dim]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int4_shard = tensor_data_dict["int4_shard"] if "int4_shard" in tensor_data_dict else None
        int2_shard = tensor_data_dict["int2_shard"] if "int2_shard" in tensor_data_dict else None
        int1_shard = tensor_data_dict["int1_shard"] if "int1_shard" in tensor_data_dict else None
        scale = tensor_data_dict["scale"]
        zero_point = tensor_data_dict["zero_point"]
        packed_shape, bit_size, pack_dim = tensor_attributes
        return cls(int4_shard, int2_shard, int1_shard, scale, zero_point, packed_shape, bit_size, pack_dim)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        int4_shard = self.int4_data.to(kwargs["device"]) if self.int4_shard is not None else None
        int2_shard = self.int2_data.to(kwargs["device"]) if self.int2_shard is not None else None
        int1_shard = self.int1_data.to(kwargs["device"]) if self.int1_shard is not None else None
        return self.__class__(
            int4_shard,
            int2_shard,
            int1_shard,
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
            self.packed_shape,
            self.bit_size,
            self.pack_dim
        )

    def _apply_fn_to_data(self, fn):
        int4_shard = fn(self.int4_shard) if self.int4_shard is not None else None
        int2_shard = fn(self.int2_shard) if self.int2_shard is not None else None
        int1_shard = fn(self.int1_shard) if self.int1_shard is not None else None  
        return self.__class__(
            int4_shard,
            int2_shard,
            int1_shard,
            fn(self.scale),
            fn(self.zero_point),
            self.packed_shape,
            self.bit_size,
            self.pack_dim,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        
        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        if func is aten.t.default:
            tensor = args[0]
            new_shape = tensor.packed_shape[::-1]
            new_pack_dim = len(tensor.packed_shape) - tensor.pack_dim -1
            int4_shard = tensor.int4_shard.t() if tensor.int4_shard is not None else None
            int2_shard = tensor.int2_shard.t() if tensor.int2_shard is not None else None
            int1_shard = tensor.int1_shard.t() if tensor.int1_shard is not None else None
            new = tensor.__class__(
                int4_shard,
                int2_shard,
                int1_shard,
                tensor.scale,
                tensor.zero_point,
                tensor.packed_shape,
                tensor.bit_size,
                new_pack_dim,
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        raise NotImplementedError(
            f"PlainTensorLayout dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self):
        shards = [self.int4_shard, self.int2_shard, self.int1_shard]
        shards = [shard for shard in shards if shard is not None]
        int_data = unpack(shards, self.bit_size, dim = self.pack_dim)
        return int_data, self.scale, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bit_size: int,
        pack_dim: int,
    ):
        shards = pack(int_data, bit_size, dim=pack_dim)
        int4_shard = shards[0] if bit_size >= 4 else None
        int1_shard = shards[-1] if bit_size % 2 else None
        int2_shard = shards[bit_size > 4] if 2 in numbits[bit_size] else None
        shape = list(int_data.shape)
        shape[pack_dim] *= bit_size // 8
        return cls(int4_shard, int2_shard, int1_shard, scale, zero_point, shape, bit_size, pack_dim)

    
@register_layout_cls("plain")
class PlainTensorLayout(IntxLayout):
    """
    normal affine layout
    fields:
      int_data (torch.Tensor): the quantized tensor
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
      bit_size (int): element size in bits
    """
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ):
        shape = int_data.shape
        kwargs = {}
        kwargs["device"] = scale.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else scale.layout
        )
        kwargs["dtype"] = torch.int8
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.int_data.shape}) range[{self.int_data.min()}, {self.int_data.max()}])"
        )
        
    def __tensor_flatten__(self):
        return ["int_data", "scale", "zero_point"], []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        zero_point = tensor_data_dict["zero_point"]
        return cls(int_data, scale, zero_point)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
        )

    def _apply_fn_to_data(self, fn):

        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point),
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        
        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        if func is aten.t.default:
            tensor = args[0]
            new_shape = tensor.shape[::-1]
                
            new = tensor.__class__(
                tensor.int_data.view(new_shape),
                tensor.scale,
                tensor.zero_point,
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        raise NotImplementedError(
            f"PlainTensorLayout dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self):
        return self.int_data, self.scale, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ):
        return cls(int_data, scale, zero_point)


@implements(torch.nn.functional.linear)
def functional_linear(*args, **kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(input_tensor, IntxTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight_tensor, IntxTensor):
        weight_tensor = weight_tensor.dequantize(output_dtype=input_tensor.dtype)
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

@implements([aten.mm.default, aten.addmm.default])
def aten_mm(func, *args, **kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(f"{func} is not implemented for non floating point input")

    # using try/except here so that we can have a general fallback when input_tensor/weight_tensor
    # is not picked up by any of the dispatch paths in `_quantized_linear_op`, this allows us to
    # make the branches easier to understand in `_quantized_linear_op`
    if func == aten.addmm.default:
        input_tensor, weight_tensor, bias = (
            args[1],
            args[2],
            args[0],
        )
        if isinstance(input_tensor, IntxTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, IntxTensor):
            weight_tensor = weight_tensor.dequantize(output_dtype=input_tensor.dtype)
        return func(bias, input_tensor, weight_tensor)
    else:
        input_tensor, weight_tensor, bias = (
            args[0],
            args[1],
            None
        )
        if isinstance(input_tensor, IntxTensor):
            input_tensor = input_tensor.dequantize()
        if isinstance(weight_tensor, IntxTensor):
            weight_tensor = weight_tensor.dequantize(output_dtype=input_tensor.dtype)
        return func(input_tensor, weight_tensor)

@implements([aten.detach.default])
def detach(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )

@implements([aten.sub.Tensor])
def sub_tensor(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(lambda x: x - args[1])
    )
@implements([aten.clone.default])
def clone(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements([aten._to_copy.default])
def _to_copy(func, *args, **kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements([aten.t.default])
def t(func, *args, **kwargs):
    block_size = args[0].block_size
    assert len(block_size) == 2
    transposed_block_size = (block_size[1], block_size[0])
    tensor = args[0]
    shape = tensor.shape[::-1]
    new = tensor.__class__(
        tensor.layout_tensor.t(), transposed_block_size,tensor.bit_size, shape, tensor.quant_min, tensor.quant_max, tensor.zero_point_domain, dtype=tensor.dtype, strides=tensor.stride()
    )
    return return_and_correct_aliasing(func, args, kwargs, new)

to_intx_quantized = IntxTensor.from_float