import functools
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
        nbits: int,
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
    Affine quantized tensor subclass. Affine quantization means we quantize the floating point tensor with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

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
      input_quant_func (Optional[Callable]): function for quantizing the input float Tensor to a quantized tensor subclass object, that takes float Tensor as input and outputs an AffineQuantizedTensor object
      dtype: dtype for external representation of the tensor, e.g. torch.float32
    """

    @staticmethod
    def __new__(
        cls,
        layout_tensor: IntxLayout,
        block_size: Tuple[int, ...],
        nbits: int,
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
        nbits: int,
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        dtype=None,
        strides=None,
    ):
        self.layout_tensor = layout_tensor
        self.block_size = block_size
        self.nbits = nbits
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.zero_point_domain = zero_point_domain

    def __repr__(self):
        print("Tensor repr")
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad})"
        )

    def dequantize(self, output_dtype=None):
        print("Tensor dequantize")
        if output_dtype is None:
            output_dtype = self.dtype
        int_data, scale, zero_point = self.layout_tensor.get_plain()
        return dequantize_affine(int_data, self.block_size, scale, zero_point, int_data.dtype, self.quant_min, self.quant_max, self.zero_point_domain, output_dtype=output_dtype)

    def __tensor_flatten__(self):
        print("Tensor flatten")
        return ["layout_tensor"], [self.block_size, self.nbits, self.shape,  self.quant_min, self.quant_max, self.zero_point_domain, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        print("Tensor unflatten")
        layout_tensor = tensor_data_dict["layout_tensor"]
        block_size, nbits, shape, quant_min, quant_max, zero_point_domain, dtype = tensor_attributes
        return cls(
            layout_tensor, block_size, nbits,
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
        nbits: int, # maybe in the future this can be a torch.dtype
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
        quant_min = -2**(nbits-1) if quant_min is None else quant_min
        quant_max = 2**(nbits-1) - 1 if quant_max is None else quant_max
        target_dtype = torch.int8
        scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype, preserve_zero, zero_point_domain)
        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max, zero_point_domain)

        layout_cls_ctr = get_layout_tensor_constructor(extended_layout)
        # TODO: this is temporary, need to come up with the proper UX
        if extended_layout == "packed":
            layout_tensor = layout_cls_ctr(int_data, scale, zero_point, nbits, pack_dim)
        else:
            raise NotImplementedError(f"Only 'packed' layout is currently implemented")
        return cls(
            layout_tensor,
            block_size,
            nbits,
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
        print("Tensor to")
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.layout_tensor.to(kwargs["device"]),
            self.block_size,
            self.nbits,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.zero_point_domain,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        print("Tensor apply fn", fn)
        return self.__class__(
            fn(self.layout_tensor),
            self.block_size,
            self.nbits,
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
            f"AffineQuantizedTensor dispatch: attempting to run {func}, this is not supported"
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
    Layout storage class for plain layout for Intx tensors, it stores scales and zeros as plain tensors 
    while packing weights from (n, m*8) -> (n, m * nbits)

    fields:
      int_data (torch.Tensor): the quantized tensor
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
      nbits (int): element size in bits
      pack_dim (int): the dimension that the tensor was packed along
    """
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        nbits: int,
        pack_dim: int,
    ):
        print("PackedTensorLayout __new__")
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
        nbits: int,
        pack_dim: int,
    ):
        print("PackedTensorLayout __init__")
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.nbits = nbits
        self.pack_dim = pack_dim

    def __tensor_flatten__(self):
        print("PackedTensorLayout __tensor_flatten__")
        return ["int_data", "scale", "zero_point"], [self.nbits, self.pack_dim]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        print("PackedTensorLayout __tensor_unflatten__")
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        zero_point = tensor_data_dict["zero_point"]
        nbits, pack_dim = tensor_attributes
        return cls(int_data, scale, zero_point, nbits, pack_dim)

    def to(self, *args, **kwargs):
        print("PackedTensorLayout to")
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
        )

    def _apply_fn_to_data(self, fn):
        print("_apply_fn_to_data: ", fn)

        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point),
            self.nbits,
            self.pack_dim,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        if func is aten.detach.default:
            print("__torch_dispatch__aten.detach.default")
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        if func is aten.t.default:
            print("__torch_dispatch__aten.t.default")
            tensor = args[0]
            new_pack_dim = len(tensor.shape) - tensor.pack_dim -1
            new_shape = tensor.shape[::-1]
                
            new = tensor.__class__(
                tensor.int_data.view(new_shape),
                tensor.scale,
                tensor.zero_point,
                tensor.nbits,
                new_pack_dim,
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        raise NotImplementedError(
            f"PackedTensorLayout dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self):
        print("get_plain")
        unpacked = unpack(self.int_data, self.nbits, dim=self.pack_dim)
        print('unpacked intdata')
        return unpacked, self.scale, self.zero_point

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        nbits: int,
        pack_dim: int = -1,
    ):
        print("from_plain")
        packed_weight = pack(int_data, nbits, dim=pack_dim)
        return cls(packed_weight, scale, zero_point, nbits, pack_dim)
    


@implements(torch.nn.functional.linear)
def functional_linear(*args, **kwargs):
    print("torch.nn.functional.linear")
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(input_tensor, IntxTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight_tensor, IntxTensor):
        weight_tensor = weight_tensor.dequantize()
    print(f'returning linear({input_tensor}\n\n,{weight_tensor}\n\n,{bias})')
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)

@implements([aten.mm.default, aten.addmm.default])
def aten_mm(func, *args, **kwargs):
    print("aten.mm.default")
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
            weight_tensor = weight_tensor.dequantize()
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
            weight_tensor = weight_tensor.dequantize()
        return func(input_tensor, weight_tensor)

@implements([aten.detach.default])
def detach(func, *args, **kwargs):
    print("aten.detach.default", args[0].shape)
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )


@implements([aten.clone.default])
def clone(func, *args, **kwargs):
    print("aten.clone.default")
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements([aten._to_copy.default])
def _to_copy(func, *args, **kwargs):
    print("aten._to_copy.default")
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements([aten.t.default])
def t(func, *args, **kwargs):
    print("aten.t.default")
    block_size = args[0].block_size
    assert len(block_size) == 2
    transposed_block_size = (block_size[1], block_size[0])
    tensor = args[0]
    shape = tensor.shape[::-1]
    new = tensor.__class__(
        tensor.layout_tensor.t(), transposed_block_size,tensor.nbits, shape, tensor.quant_min, tensor.quant_max, tensor.zero_point_domain, dtype=tensor.dtype, strides=tensor.stride()
    )
    return return_and_correct_aliasing(func, args, kwargs, new)

# @implements([aten.select.default])
to_intx_quantized = IntxTensor.from_float