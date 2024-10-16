import torch
from typing import Callable, Optional
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.utils import (
    TorchAOBaseTensor,
    TORCH_VERSION_AT_LEAST_2_5,
)
from torchao.dtypes import to_affine_quantized_intx, to_affine_quantized_intx_static
from torchao.quantization.utils import _get_per_token_block_size
from torchao.quantization.quant_primitives import MappingType

__all__ = [
    "LinearActivationScaleQuantizedTensor",
    "to_linear_scale_activation_quantized",
]

aten = torch.ops.aten

class LinearActivationScaleQuantizedTensor(TorchAOBaseTensor):
    """
    Applies activation scaling then quantization for linear operator, this is used to support
    SmoothQuant with dynamic quantization or static quantization, user can pass in a `input_quant_func`
    that is used to quantize the activation

    Args:
      `original_weight_tensor`: the weight tensor, if weight need to be quantized as well, we'd need
        to apply quantization to weight first, e.g. for int8 dynamic activation int8 weight quantization
        we will first apply int8 quantization to weight and then apply LinearActivationScaleQuantizedTensor
        on top of it
      `scale`: The scale tensor to be applied to activation.
      `input_quant_func` (Callable[[torch.Tensor], torch.Tensor]): a function that takes a high precision
        floating point tensor and returns a quantized tensor, this is used to quantize input
    """
    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        equalization_scale: torch.Tensor,
        act_scales: Optional[torch.Tensor],
        act_zero_points: Optional[torch.Tensor],
        target_dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
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
        equalization_scale: torch.Tensor,
        act_scales: Optional[torch.Tensor],
        act_zero_points: Optional[torch.Tensor],
        target_dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.equalization_scale = equalization_scale
        self.act_scales = act_scales
        self.act_zero_points = act_zero_points
        self.target_dtype = target_dtype
        self.quant_min = quant_min
        self.quant_max = quant_max

    def __repr__(self):
        return (f"LinearActivationScaleQuantizedTensor({self.original_weight_tensor}, "
                f"equalization_scale={self.equalization_scale}, "
                f"act_scales={self.act_scales}), "
                f"act_zero_points={self.act_zero_points}, "
                f"target_dtype={self.target_dtype}, "
                f"quant_min={self.quant_min}, "
                f"quant_max={self.quant_max})"
                )

    def __tensor_flatten__(self):
        tensor_data = [
            "original_weight_tensor",
            "equalization_scale",
        ]
        tensor_attributes = [self.target_dtype, self.quant_min, self.quant_max]
        if self.act_scales is not None:
            tensor_data.append("act_scales")
        if self.act_zero_points is not None:
            tensor_data.append("act_zero_points")
        return tensor_data, tensor_attributes

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        equalization_scale = tensor_data_dict["equalization_scale"]
        act_scales = tensor_data_dict["act_scales"] if "act_scales" in tensor_data_dict else None
        act_zero_points = tensor_data_dict["act_zero_points"] if "act_zero_points" in tensor_data_dict else None
        target_dtype, quant_min, quant_max = tensor_attributes
        return cls(
            original_weight_tensor,
            equalization_scale,
            act_scales,
            act_zero_points,
            target_dtype,
            quant_min,
            quant_max,
        )

    @staticmethod
    def _quantized_linear_op(input_tensor, weight_tensor, bias):
        original_weight_tensor = weight_tensor.original_weight_tensor
        equalization_scale = weight_tensor.equalization_scale
        scaled_input_act = input_tensor / equalization_scale
        scaled_input_act = scaled_input_act.to(input_tensor.dtype)
        if weight_tensor.act_scales is not None:
            # static quant
            act_zero_points = (
                weight_tensor.act_zero_points
                if weight_tensor.act_zero_points is not None
                else torch.zeros_like(weight_tensor.act_scales, dtype=torch.int64)
            )
            aqt = to_affine_quantized_intx_static(
                scaled_input_act,
                weight_tensor.act_scales,
                act_zero_points,
                list(scaled_input_act.shape),
                weight_tensor.target_dtype,
                quant_min=weight_tensor.quant_min,
                quant_max=weight_tensor.quant_max,
            )
        else:
            # dynamic quant
            block_size = _get_per_token_block_size(scaled_input_act)
            aqt = to_affine_quantized_intx(
                scaled_input_act,
                MappingType.SYMMETRIC,
                block_size,
                weight_tensor.target_dtype,
                quant_min=weight_tensor.quant_min,
                quant_max=weight_tensor.quant_max,
            )

        return torch.nn.functional.linear(aqt, original_weight_tensor, bias)

    @classmethod
    def from_float(cls, input_float, equalization_scale, act_scales, act_zero_points, target_dtype, quant_min, quant_max):
        return cls(
            input_float,
            equalization_scale,
            act_scales,
            act_zero_points,
            target_dtype,
            quant_min,
            quant_max,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            fn(self.equalization_scale),
            (
                fn(self.act_scales)
                if self.act_scales is not None
                else None
            ),
            (
                fn(self.act_zero_points)
                if self.act_zero_points is not None
                else None
            ),
            self.target_dtype,
            self.quant_min,
            self.quant_max,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.original_weight_tensor.to(**kwargs),
            self.equalization_scale.to(**kwargs),
            (
                self.act_scales.to(**kwargs)
                if self.act_scales is not None
                else None
            ),
            (
                self.act_zero_points.to(**kwargs)
                if self.act_zero_points is not None
                else None
            ),
            self.target_dtype,
            self.quant_min,
            self.quant_max,
        )

implements = LinearActivationScaleQuantizedTensor.implements

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, LinearActivationScaleQuantizedTensor):
        return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)

    raise NotImplementedError("LinearActivationScaleQuantizedTensor: No specialized dispatch found for linear op")

@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(f"LinearActivationScaleQuantizedTensor: expecting a floating point input")

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
    else:
        # aten.mm.default
        assert args[0].shape[-1] == args[1].shape[0], (
            f"need mat1 shape: {args[0].shape} final dim"
            f"to match mat2 shape: {args[1].shape} first dim"
        )
        input_tensor, weight_tensor, bias = (
            args[0],
            args[1],
            None,
        )
    original_weight_tensor = weight_tensor.original_weight_tensor
    equalization_scale = weight_tensor.equalization_scale
    scaled_input_act = input_tensor / equalization_scale
    scaled_input_act = scaled_input_act.to(input_tensor.dtype)
    if weight_tensor.act_scales is not None:
        # static quant
        act_zero_points = (
            weight_tensor.act_zero_points
            if weight_tensor.act_zero_points is not None
            else torch.zeros_like(weight_tensor.act_scales, dtype=torch.int64)
        )
        aqt = to_affine_quantized_intx_static(
            scaled_input_act,
            weight_tensor.act_scales,
            act_zero_points,
            list(scaled_input_act.shape),
            weight_tensor.target_dtype,
            quant_min=weight_tensor.quant_min,
            quant_max=weight_tensor.quant_max,
        )
    else:
        # dynamic quant
        block_size = _get_per_token_block_size(scaled_input_act)
        aqt = to_affine_quantized_intx(
            scaled_input_act,
            MappingType.SYMMETRIC,
            block_size,
            weight_tensor.target_dtype,
            quant_min=weight_tensor.quant_min,
            quant_max=weight_tensor.quant_max,
        )

    if func == aten.addmm.default:
        return func(bias, aqt, original_weight_tensor)
    else:
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

to_linear_scale_activation_quantized = LinearActivationScaleQuantizedTensor.from_float

if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with LinearActivationScaleQuantizedTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([LinearActivationScaleQuantizedTensor])
