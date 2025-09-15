# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor

__all__ = [
    "LinearActivationQuantizedTensor",
    "to_linear_activation_quantized",
]

aten = torch.ops.aten


class LinearActivationQuantizedTensor(TorchAOBaseTensor):
    """
    Applies activation quantization for linear operator, this is used to support
    dynamic quantization, user can pass in a `input_quant_func`
    that is used to quantize the activation

    Args:
      `original_weight_tensor`: the weight tensor, if weight need to be quantized as well, we'd need
        to apply quantization to weight first, e.g. for int8 dynamic activation int8 weight quantization
        we will first apply int8 quantization to weight and then apply LinearActivationQuantizedTensor
        on top of it
      `input_quant_func` (Callable[[torch.Tensor], torch.Tensor]): a function that takes a high precision floating point tensor and returns
        a quantized tensor, this is used to quantize input
      `quant_kwargs` (Dict[str, Any]): Additional keyword arguments for the quantization function.
        Restriction: Must not contain tensor values.
    """

    quant_kwargs: Dict[str, Any]

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_quant_func: Callable,
        quant_kwargs: Dict[str, Any],
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
        quant_kwargs: Dict[str, Any],
    ):
        self.original_weight_tensor = original_weight_tensor
        self.input_quant_func = input_quant_func
        self.quant_kwargs = quant_kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.original_weight_tensor}, {self.input_quant_func}, quant_kwargs={self.quant_kwargs}))"

    def __tensor_flatten__(self):
        return ["original_weight_tensor"], [self.input_quant_func, self.quant_kwargs]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        input_quant_func, quant_kwargs = tensor_attributes
        return cls(original_weight_tensor, input_quant_func, quant_kwargs)

    @staticmethod
    def _quantized_linear_op(
        input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
    ):
        if input_tensor.numel() == 0:
            return input_tensor
        input_quant_func = weight_tensor.input_quant_func
        original_weight_tensor = weight_tensor.original_weight_tensor
        quant_kwargs = weight_tensor.quant_kwargs
        quantized_tensor = input_quant_func(input_tensor, **quant_kwargs)
        return torch.nn.functional.linear(
            quantized_tensor, original_weight_tensor, bias
        )

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        input_quant_func: Callable,
        quant_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if quant_kwargs is None:
            quant_kwargs = {}
        return cls(input_float, input_quant_func, quant_kwargs)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            self.input_quant_func,
            self.quant_kwargs,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.original_weight_tensor.to(**kwargs),
            self.input_quant_func,
            self.quant_kwargs,
        )


def _same_metadata(
    self: LinearActivationQuantizedTensor, src: LinearActivationQuantizedTensor
):
    return (
        isinstance(self, LinearActivationQuantizedTensor)
        and isinstance(src, LinearActivationQuantizedTensor)
        and self.shape == src.shape
        and self.input_quant_func == src.input_quant_func
        and self.quant_kwargs == src.quant_kwargs
    )


implements = LinearActivationQuantizedTensor.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor = kwargs.get("input", args[0] if len(args) > 0 else None)
    weight_tensor = kwargs.get("weight", args[1] if len(args) > 1 else None)
    bias = kwargs.get("bias", args[2] if len(args) > 2 else None)

    assert input_tensor is not None, "input tensor must not be None"
    assert weight_tensor is not None, "weight tensor must not be None"

    if isinstance(weight_tensor, LinearActivationQuantizedTensor):
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)

    raise NotImplementedError(
        "LinearActivationQuantizedTensor: No specialized dispatch found for linear op"
    )


@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(
            "LinearActivationQuantizedTensor: expecting a floating point input"
        )

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
        qtensor = input_quant_func(input_tensor, **weight_tensor.quant_kwargs)
        return func(bias, qtensor, original_weight_tensor)
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
        qtensor = input_quant_func(input_tensor, **weight_tensor.quant_kwargs)
        return func(qtensor, original_weight_tensor)


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(func)
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


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if _same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    elif type(self) is torch.Tensor and type(src) is LinearActivationQuantizedTensor:
        new_src = src.to(dtype=self.dtype, device=self.device)
        self.copy_(new_src)
        return

    raise ValueError(
        f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        LinearActivationQuantizedTensor(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].input_quant_func,
            args[0].quant_kwargs,
        ),
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        LinearActivationQuantizedTensor(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].input_quant_func,
            args[0].quant_kwargs,
        ),
    )


@implements(aten.index.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        LinearActivationQuantizedTensor(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].input_quant_func,
            args[0].quant_kwargs,
        ),
    )


# this is needed for DTensor.from_local() and for flattening tensor
@implements(aten.view.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        LinearActivationQuantizedTensor(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].input_quant_func,
            args[0].quant_kwargs,
        ),
    )


to_linear_activation_quantized = LinearActivationQuantizedTensor.from_float  # Converts a float tensor to LinearActivationQuantizedTensor for dynamic activation quantization

# Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([LinearActivationQuantizedTensor])
