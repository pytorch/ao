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
    "WeightTensorWithLinearActivationQuantizationMetadata",
    "to_weight_tensor_with_linear_activation_quantization_metadata",
]

aten = torch.ops.aten


class WeightTensorWithLinearActivationQuantizationMetadata(TorchAOBaseTensor):
    """
    Tensor subclass that wraps a weight tensor and provides metadata for linear activation static quantization.

    Args:
        original_weight_tensor (torch.Tensor): The weight tensor to be wrapped.
        input_quant_func_static (Callable): The quantization function for inputs.
            Must have the signature: (Tensor, scale: Tensor, zero_point: Optional[Tensor], **quant_kwargs) -> Tensor
        scale (torch.Tensor): The scale tensor for activation quantization.
        zero_point (Optional[torch.Tensor]): The zero point tensor for activation quantization. Default is None.
        quant_kwargs (Dict[str, Any]): Additional keyword arguments for the quantization function.
            Restriction: Must not contain tensor values.
    """

    original_weight_tensor: torch.Tensor
    input_quant_func_static: Callable
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor]
    quant_kwargs: Dict[str, Any]

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        input_quant_func_static: Callable,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
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
        input_quant_func_static: Callable[
            [torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]],
            torch.Tensor,
        ],
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        quant_kwargs: Dict[str, Any],
    ):
        self.original_weight_tensor = original_weight_tensor
        self.input_quant_func_static = input_quant_func_static
        self.scale = scale
        self.zero_point = zero_point
        self.quant_kwargs = quant_kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.original_weight_tensor}, {self.input_quant_func_static}, scale={self.scale}, zero_point={self.zero_point}, quant_kwargs={self.quant_kwargs})"

    def __tensor_flatten__(self):
        tensor_data = ["original_weight_tensor", "scale"]
        if self.zero_point is not None:
            tensor_data.append("zero_point")
        return tensor_data, [self.input_quant_func_static, self.quant_kwargs]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        input_quant_func_static, quant_kwargs = tensor_attributes
        zero_point = tensor_data_dict.get("zero_point", None)
        return cls(
            original_weight_tensor,
            input_quant_func_static,
            tensor_data_dict["scale"],
            zero_point,
            quant_kwargs,
        )

    @staticmethod
    def _quantized_linear_op(
        input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
    ):
        input_quant_func_static = weight_tensor.input_quant_func_static
        original_weight_tensor = weight_tensor.original_weight_tensor
        scale = weight_tensor.scale
        zero_point = weight_tensor.zero_point
        quant_kwargs = weight_tensor.quant_kwargs
        quantized_input_act = input_quant_func_static(
            input_tensor, scale=scale, zero_point=zero_point, **quant_kwargs
        )
        return torch.nn.functional.linear(
            quantized_input_act, original_weight_tensor, bias
        )

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        input_quant_func: Callable,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        quant_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if quant_kwargs is None:
            quant_kwargs = {}
        return cls(input_float, input_quant_func, scale, zero_point, quant_kwargs)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            self.input_quant_func_static,
            fn(self.scale),
            fn(self.zero_point) if self.zero_point is not None else None,
            self.quant_kwargs,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.original_weight_tensor.to(device),
            self.input_quant_func_static,
            self.scale.to(device),
            self.zero_point.to(device) if self.zero_point is not None else None,
            self.quant_kwargs,
        )


implements = WeightTensorWithLinearActivationQuantizationMetadata.implements
implements_torch_function = (
    WeightTensorWithLinearActivationQuantizationMetadata.implements_torch_function
)


@implements_torch_function(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, WeightTensorWithLinearActivationQuantizationMetadata):
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)

    raise NotImplementedError(
        "LinearActivationQuantizedTensor: No specialized dispatch found for linear op"
    )


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


to_weight_tensor_with_linear_activation_quantization_metadata = (
    WeightTensorWithLinearActivationQuantizationMetadata.from_float
)

# Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals(
    [WeightTensorWithLinearActivationQuantizationMetadata]
)
