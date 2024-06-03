import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.quantization.quant_primitives import (
    dequantize_per_channel,
    dynamically_quantize_per_channel,
    groupwise_affine_quantize_tensor,
    quant_int8_dynamic_per_token_linear,
    unpack_tinygemm_scales_and_zeros,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
)
from typing import Tuple, Optional, Callable, Dict, Any
aten = torch.ops.aten
from torchao.dtypes.nf4tensor import expect_args_len_at_k
from torchao.dtypes.nf4tensor import CompareOp


AQT_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Use this decorator to implement a function for an aten op in __torch_dispatch__"""

    def decorator(func):
        for op in aten_ops:
            AQT_OPS_TABLE[op] = func
        return func

    return decorator


# TODO: A smarter version of this decorator would do
# arg checking for me and I wouldn't need to deal with
# brittle args and kwargs.
@implements([torch.ops.aten.t.default])
def t_default(func, *args, **kwargs):
    # NOTE: It seems that args[0] is the new args
    # and args[1] the old kwargs
    # In turn kwargs is now always empty
    a = args[0][0]
    return AffineQuantizedTensor(
        a.int_data,
        a.scale,
        a.zero_point,
        a.block_size,
        a.shape,
        a.quant_min,
        a.quant_max,
        input_quant_func=a.input_quant_func,
        dtype=a.dtype,
        strides=(a.stride(1), a.stride(0)),
    )

@implements(
    [
        aten.view.default,
    ]
)
def aqt_view(aten_op, args, kwargs=None):
    a = args[0]
    size = args[1]
    int_data = a.int_data.view(size)
    return AffineQuantizedTensor(
        int_data,
        a.scale,
        a.zero_point,
        a.block_size,
        int_data.shape,
        a.quant_min,
        a.quant_max,
        input_quant_func=a.input_quant_func,
        dtype=a.dtype,
        strides=int_data.stride(),
    )

@implements([torch.ops.aten.mm.default])
def mm_default(func, *args, **kwargs):
    return (args[0][0], args[0][1])
    from torchao.quantization.quant_primitives import quant_int8_per_token_matmul
    quant_int8_per_token_matmul(args[0][0].int_data, args[0][0].scales,
                                args[0][1].int_data, args[0][1].scales)

class AffineQuantizedTensor(torch.Tensor):
    """
    Base affine quantized tensor subclass. When the from_float method is used,
    to create an instance of any AffineQuantizedTensor

    The shape and dtype of the tensor subclass represent how the tensor subclass looks externally,
    regardless of the internal representation's type or orientation.

    Affine quantization means we quantize the floating point tensor with an affine transformation:
       quantized_tensor = float_tensor / scale + zero_point

    fields:
      int_data (torch.Tensor): the quantized integer data Tensor
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
      block_size (Tuple[int, ...]): granularity of quantization, this means the size of the tensor elements that's sharing the same qparam
         e.g. when size is the same as the input tensor dimension, we are using per tensor quantization
      shape (torch.Size): the shape for the Tensor
      quant_min (Optional[int]): minimum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      quant_max (Optional[int]): maximum quantized value for the Tensor, if not specified, it will be derived from dtype of `int_data`
      input_quant_func (Optional[Callable]): function for quantizing the input float Tensor to a quantized tensor subclass object, that takes input Tensor as input and outputs an AffineQuantizedTensor object
      dtype: dtype for external representation of the tensor, e.g. torch.float32
    """

    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        input_quant_func: Optional[Callable] = None,
        dtype=None,
        *args,
        **kwargs
    ):
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        if dtype is None:
            dtype = scale.dtype
        kwargs["dtype"] = dtype
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: Tuple[int, ...],
        shape: torch.Size,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        input_quant_func: Optional[Callable] = None,
        dtype=None,
        *args,
        **kwargs
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.input_quant_func = input_quant_func

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.dequantize()}, shape={self.shape}, "
            f"device={self.device}, dtype={self.dtype}, input_quant_func={self.input_quant_func}, requires_grad={self.requires_grad})"
        )

    def dequantize(self, output_dtype=torch.float32):
        return dequantize_affine(self.int_data, self.block_size, self.scale, self.zero_point, self.int_data.dtype, self.quant_min, self.quant_max, output_dtype=output_dtype)

    def __tensor_flatten__(self):
        return ["int_data", "scales", "zero_point"], [self.block_size, self.shape, self.quant_min, self.quant_max, self.input_quant_func, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = tensor_data_dict["int_data"], tensor_data_dict["scale"], tensor_data_dict["zero_point"]
        block_size, shape, quant_min, quant_max, input_quant_func, dtype = tensor_attributes
        return cls(
            int_data,
            scale,
            zero_point,
            block_size,
            shape if outer_size is None else outer_size,
            quant_min,
            quant_max,
            input_quant_func=input_quant_func,
            dtype=dtype,
            strides=outer_stride,
        )

    @classmethod
    def from_float(
        cls,
        input_float,
        mapping_type,
        block_size,
        target_dtype,
        quant_min = None,
        quant_max = None,
        eps = None,
        scale_dtype = None,
        zero_point_dtype = None,
        input_quant_func = None,
    ):
        scale, zero_point = choose_qparams_affine(input_float, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, scale_dtype, zero_point_dtype)
        int_data = quantize_affine(input_float, block_size, scale, zero_point, target_dtype, quant_min, quant_max)
        return cls(
            int_data,
            scale,
            zero_point,
            block_size,
            input_float.shape,
            quant_min,
            quant_max,
            input_quant_func=input_quant_func,
            dtype=input_float.dtype
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        print("tf func: ", func)
        kwargs = {} if kwargs is None else kwargs
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
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"]),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.input_quant_func,
            **kwargs,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point),
            self.block_size,
            self.shape,
            self.quant_min,
            self.quant_max,
            self.input_quant_func,
            dtype=self.dtype,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print("td func: ", func)
        if func is aten.select.int:
            import pdb; pdb.set_trace()
        if func in AQT_OPS_TABLE:
            return AQT_OPS_TABLE[func](func, args, kwargs)
        # Note: we only added cpu path here for 8da4w, this is for executorch, in the future
        # 1. we'll add cpu/cuda version (int4mm etc.)
        # 2. we'll need to hide the 8da4w executorch version under things like layouts (we also have multiple impl for cpu kernel as Michael mentioned), so it will be something like
        #   cpu device + et laytout --> gives current 8da4w executorch representation
        #   cpu device + avx layout --> gives optimized kernel for 8da4w in avx cpu etc.
        #   cuda device + some layout --> gives cuda kernel

        # two scenarios where we currently fall back to vanilla mm:
        # 1 - when tensor is on CUDA: we'll add this later, we'll also enable dispatching to optimized
        #     kernels in CPU as well, see the note above
        # 2 - we're given non-floats - quantizing long to int8 is crazy
        if (
            func in [aten.mm.default, aten.addmm.default]
            and args[0].is_floating_point()
            and args[0].device == torch.device("cpu")
        ):
            if func == aten.addmm.default:
                assert args[1].shape[-1] == args[2].shape[0], (
                    f"need mat1 shape: {args[1].shape} final"
                    f"dim to match mat2 shape: {args[2].shape} first dim "
                )
                input_tensor, weight_qtensor, bias = (
                    args[1],
                    args[2],
                    args[0],
                )
            else:
                assert args[0].shape[-1] == args[1].shape[0], (
                    f"need mat1 shape: {args[0].shape} final dim"
                    f"to match mat2 shape: {args[1].shape} first dim"
                )
                input_tensor, weight_qtensor, bias = (
                    args[0],
                    args[1],
                    None if len(args) == 2 else args[2],
                )
            if weight_qtensor.input_quant_func is not None:
                # dynamic quantization
                input_tensor = weight_qtensor.input_quant_func(input_tensor)
                input_tensor = input_tensor.dequantize()
            weight_tensor = weight_qtensor.dequantize()
            return func(input_tensor, weight_tensor, bias)

        if (func is aten.detach.default or
            func is aten.clone.default or
            func is aten._to_copy.default):
            return args[0]

        if func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        if func is aten._to_copy.default:
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
            )

        raise NotImplementedError(f"{func} is not supported by torch_dispatch")
