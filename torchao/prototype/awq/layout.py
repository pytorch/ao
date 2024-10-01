import torch
from typing import Callable, Optional, Dict, Any
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import (
    TorchAOBaseTensor,
    TORCH_VERSION_AT_LEAST_2_5,
)
from torchao.dtypes import AffineQuantizedTensor


aten = torch.ops.aten


class WeightTensorWithEqualizationScales(TorchAOBaseTensor):
    """
    Tensor subclass that wraps a quantized weight tensor and provides the equalization scales which are applied to activations.

    Args:
        quantized_weight_tensor (torch.Tensor): The weight tensor to be wrapped.
        scale (torch.Tensor): The scale tensor for activation quantization.
        zero_point (Optional[torch.Tensor]): The zero point tensor for activation quantization. Default is None.
        equalization_scale (torch.Tensor): The equalization scale tensor.
    """

    quantized_weight_tensor: AffineQuantizedTensor
    equalization_scale: torch.Tensor

    def __new__(
        cls,
        quantized_weight_tensor: torch.Tensor,
        equalization_scale: torch.Tensor
    ):
        kwargs = {}
        dtype = quantized_weight_tensor.dtype
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        kwargs["device"] = quantized_weight_tensor.device
        shape = quantized_weight_tensor.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        quantized_weight_tensor: torch.Tensor,
        equalization_scale: torch.Tensor
    ):
        self.quantized_weight_tensor = quantized_weight_tensor
        self.equalization_scale = equalization_scale

    def __repr__(self):
        return f"LinearActivationQuantizedTensor({self.quantized_weight_tensor}, eq_scale={self.equalization_scale})"

    def __tensor_flatten__(self):
        tensor_data = ["quantized_weight_tensor", "equalization_scale"]
        return tensor_data, []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        quantized_weight_tensor = tensor_data_dict["quantized_weight_tensor"]
        equalization_scale = tensor_data_dict["equalization_scale"]
        return cls(
            quantized_weight_tensor,
            equalization_scale,
        )

    @staticmethod
    def _quantized_linear_op(
        input_tensor: torch.Tensor, weight_tensor: torch.Tensor, bias: torch.Tensor
    ):
        return torch.nn.functional.linear(
            input_tensor / weight_tensor.equalization_scale, weight_tensor.quantized_weight_tensor.dequantize(), bias
        )

    @classmethod
    def from_quantized(
        cls,
        quantized_weight_tensor: AffineQuantizedTensor,
        equalization_scale: torch.Tensor
    ):
        return cls(quantized_weight_tensor, equalization_scale)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.quantized_weight_tensor),
            fn(self.equalization_scale),
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.quantized_weight_tensor.to(device),
            self.equalization_scale.to(device),
        )


implements = WeightTensorWithEqualizationScales.implements


@implements(torch.nn.functional.linear)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, WeightTensorWithEqualizationScales):
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

to_weight_tensor_with_equalization_scales = WeightTensorWithEqualizationScales.from_quantized
if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with LinearActivationQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals(
        [WeightTensorWithEqualizationScales]
    )