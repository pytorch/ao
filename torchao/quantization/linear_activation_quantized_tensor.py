import torch
from typing import Callable
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import (
    TorchAOBaseTensor,
    TORCH_VERSION_AT_LEAST_2_5,
)

__all__ = [
    "LinearActivationQuantizedTensor",
    "to_linear_activation_quantized",
]

aten = torch.ops.aten

class LinearActivationQuantizedTensor(TorchAOBaseTensor):
    """
    Applies activation quantization for linear operator, this is used to support
    dynamic quantization or static quantization, user can pass in a `input_quant_func`
    that is used to quantize the activation

    Args:
      `original_weight_tensor`: the weight tensor, if weight need to be quantized as well, we'd need
        to apply quantization to weight first, e.g. for int8 dynamic activation int8 weight quantization
        we will first apply int8 quantization to weight and then apply LinearActivationQuantizedTensor
        on top of it
      `input_quant_func` (Callable[[torch.Tensor], torch.Tensor]): a function that takes a high precision floating point tensor and returns
        a quantized tensor, this is used to quantize input
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
        input_quant_func: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.original_weight_tensor = original_weight_tensor
        self.input_quant_func = input_quant_func

    def __repr__(self):
        return f"LinearActivationQuantizedTensor({self.original_weight_tensor}, {self.input_quant_func})"

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

    @staticmethod
    def _quantized_linear_op(input_tensor, weight_tensor, bias):
        input_quant_func = weight_tensor.input_quant_func
        original_weight_tensor = weight_tensor.original_weight_tensor
        aqt = input_quant_func(input_tensor)
        return torch.nn.functional.linear(aqt, original_weight_tensor, bias)

    @classmethod
    def from_float(cls, input_float, input_quant_func):
        return cls(input_float, input_quant_func)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            self.input_quant_func,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.original_weight_tensor.to(**kwargs),
            self.input_quant_func,
        )

implements = LinearActivationQuantizedTensor.implements

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, LinearActivationQuantizedTensor):
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)

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

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([LinearActivationQuantizedTensor])
