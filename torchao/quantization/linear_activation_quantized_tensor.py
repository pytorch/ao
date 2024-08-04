import torch
from torchao.dtypes.utils import (
    _implements,
    _dispatch__torch_function__,
    _dispatch__torch_dispatch__,
)
from typing import Callable
from torch.utils._python_dispatch import return_and_correct_aliasing

__all__ = [
    "LinearActivationQuantizedTensor",
    "to_linear_activation_quantized",
]

aten = torch.ops.aten

class LinearActivationQuantizedTensor(torch.Tensor):
    """
    Applies activation quantization for linear operator
    """
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable,
    ):
        kwargs = {}
        dtype = original_weight_tensor.dtype
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        kwargs["device"] = original_weight_tensor.device
        shape = original_weight_tensor.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.input_quant_func = input_quant_func

    def __tensor_flatten__(self):
        return ["original_weight_tensor"], [self.input_quant_func]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        input_quant_func, = tensor_attributes
        return cls(
            original_weight_tensor,
            input_quant_func,
        )

    @classmethod
    def from_float(cls, input_float, input_quant_func):
        return cls(input_float, input_quant_func)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            self.input_quant_func,
        )

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
            self.original_weight_tensor.to(**kwargs),
            self.input_quant_func,
        )

    implements = classmethod(_implements)
    __torch_function__ = classmethod(_dispatch__torch_function__)
    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)

implements = LinearActivationQuantizedTensor.implements

@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, LinearActivationQuantizedTensor):
        input_quant_func = weight_tensor.input_quant_func
        original_weight_tensor = weight_tensor.original_weight_tensor
        aqt = input_quant_func(input_tensor)
        return torch.nn.functional.linear(aqt, original_weight_tensor, bias)

    raise NotImplementedError("LinearActivationQuantizedTensor: No specialized dispatch found for linear op")

@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(f"LinearActivationQuantizedTensor: expecting a floating point input")

    if func == aten.addmm.default:
        assert args[1].shape[-1] == args[2].shape[0], (
            f"need mat1 shape: {args[1].shape} final"
            f"dim to match mat2 shape: {args[2].shape} first dim "
        )
        input_tensor, weight_tensor, bias = (
            args[1],
            args[2],
            args[0],
        )
        input_quant_func = weight_tensor.input_quant_func
        original_weight_tensor = weight_tensor.original_weight_tensor
        aqt = input_quant_func(input_tensor)
        return func(bias, aqt, original_weight_tensor)
    else:
        # aten.mm.default
        assert args[0].shape[-1] == args[1].shape[0], (
            f"need mat1 shape: {args[0].shape} final dim"
            f"to match mat2 shape: {args[1].shape} first dim"
        )
        input_tensor, weight_tensor = (
            args[0],
            args[1],
        )
        input_quant_func = weight_tensor.input_quant_func
        original_weight_tensor = weight_tensor.original_weight_tensor
        aqt = input_quant_func(input_tensor)
        return func(aqt, original_weight_tensor)


@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
    )

@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )

@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )

@implements(aten.t.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
    )

to_linear_activation_quantized = LinearActivationQuantizedTensor.from_float
