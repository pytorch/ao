from typing import Any, Callable, Dict, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

__all__ = [
    "WeightTensorWithLinearActivationScaleMetadata",
    "to_weight_tensor_with_linear_activation_scale_metadata",
]

aten = torch.ops.aten


class WeightTensorWithLinearActivationScaleMetadata(TorchAOBaseTensor):
    """
    Tensor subclass that wraps a weight tensor and provides metadata for linear activation scaling.
    Right now we hardcode how we apply the scale:
       scaled_linear_act = input_act / scale
       out = F.linear(scaled_linear_act, weight, ...)

    We can generalize this to accept a function as well if needed.

    Args:
        original_weight_tensor (torch.Tensor): The weight tensor to be wrapped.
        scale (torch.Tensor): The scale tensor to be applied to activation.
    """

    original_weight_tensor: torch.Tensor
    scale: torch.Tensor

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        scale: torch.Tensor,
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
        scale: torch.Tensor,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.scale = scale

    def __repr__(self):
        return f"WeightTensorWithLinearActivationScaleMetadata({self.original_weight_tensor}, scale={self.scale}"

    def __tensor_flatten__(self):
        tensor_data = ["original_weight_tensor", "scale"]
        return tensor_data, []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        return cls(
            tensor_data_dict["original_weight_tensor"],
            tensor_data_dict["scale"],
        )

    @staticmethod
    def _quantized_linear_op(
        input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
    ):
        original_weight_tensor = weight_tensor.original_weight_tensor
        scale = weight_tensor.scale
        # Note: we can make this function configurable as well
        scaled_input_act = input_tensor / scale
        return torch.nn.functional.linear(
            scaled_input_act, original_weight_tensor, bias
        )

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
        scale: torch.Tensor,
    ):
        return cls(input_float, scale)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            fn(self.scale),
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.original_weight_tensor.to(device),
            self.scale.to(device),
        )


implements = WeightTensorWithLinearActivationScaleMetadata.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, WeightTensorWithLinearActivationScaleMetadata):
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


to_weight_tensor_with_linear_activation_scale_metadata = (
    WeightTensorWithLinearActivationScaleMetadata.from_float
)

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals(
        [WeightTensorWithLinearActivationScaleMetadata]
    )
