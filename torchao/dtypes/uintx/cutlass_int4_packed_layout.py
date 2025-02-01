from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.uintx.plain_layout import (
    _aqt_is_int8_reduced_range,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout

aten = torch.ops.aten


def _aqt_is_int4(aqt):
    """Check if an AffineQuantizedTensor is int4 quantized Tensor"""
    # TODO: use torch.int4
    return (
        aqt.tensor_impl.dtype == torch.int8
        and aqt.quant_min == -8
        and aqt.quant_max == 7
    )


@dataclass(frozen=True)
class CutlassInt4PackedLayout(Layout):
    """Layout class for int4 packed layout for affine quantized tensor, for cutlass kernel."""

    pass


@register_layout(CutlassInt4PackedLayout)
class Int4PackedTensorImpl(AQTTensorImpl):
    """
    TensorImpl storage class for int4 packed layout for affine quantized tensor.
    """

    @staticmethod
    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = int_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else int_data.layout
        )
        kwargs["dtype"] = int_data.dtype
        kwargs["requires_grad"] = False
        shape = int_data.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        self.int_data = int_data
        self.scale = scale
        self._layout = _layout

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        raise NotImplementedError(
            f"Int4PackedTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [
            self._layout,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data = tensor_data_dict["int_data"]
        scale = tensor_data_dict["scale"]
        _layout = tensor_attributes
        return cls(int_data, scale, _layout)

    def get_plain(self):
        int_data = torch.stack(
            ((self.int_data << 4) >> 4, self.int_data >> 4), dim=2
        ).view((self.int_data.shape[0], 2 * self.int_data.shape[1]))
        return int_data, self.scale, None

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert zero_point is None or torch.all(zero_point == 0)

        int_data_s4 = ((int_data[:, 1::2] & 0xF) << 4) | (int_data[:, 0::2] & 0xF)
        return cls(
            int_data_s4,
            scale,
            _layout,
        )

    def get_layout(self) -> Layout:
        return self._layout

    def _apply_fn_to_data(self, fn):
        self.int_data = fn(self.int_data)
        self.scale = fn(self.scale)
        return self


def _linear_int8_act_int4_weight_cutlass_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and input_tensor.dtype in (torch.float16, torch.bfloat16)
        and len(input_tensor.shape) >= 2
        and input_tensor.tensor_impl.scale.dtype == input_tensor.dtype
        and len(input_tensor.tensor_impl.scale.shape) == len(input_tensor.shape) - 1
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_int4(weight_tensor)
        and weight_tensor.dtype == input_tensor.dtype
        and len(weight_tensor.shape) == 2
        and weight_tensor.tensor_impl.scale.dtype == weight_tensor.dtype
        and len(weight_tensor.tensor_impl.scale.shape) == 1
        and (bias is None or bias.dtype == input_tensor.dtype)
        and (bias is None or len(bias.shape) == 1)
    )


def _linear_int8_act_int4_weight_cutlass_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import rowwise_scaled_linear_cutlass_s8s4

    weight = weight_tensor.tensor_impl.int_data
    weight_scale = weight_tensor.tensor_impl.scale
    input = input_tensor.tensor_impl.int_data
    input_scale = input_tensor.tensor_impl.scale

    out = rowwise_scaled_linear_cutlass_s8s4(
        input, input_scale, weight, weight_scale, bias
    )

    return out


def _linear_int4_act_int4_weight_cutlass_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int4(input_tensor)
        and input_tensor.dtype in (torch.float16, torch.bfloat16)
        and len(input_tensor.shape) >= 2
        and input_tensor.tensor_impl.scale.dtype == input_tensor.dtype
        and len(input_tensor.tensor_impl.scale.shape) == len(input_tensor.shape) - 1
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_int4(weight_tensor)
        and weight_tensor.dtype == input_tensor.dtype
        and len(weight_tensor.shape) == 2
        and weight_tensor.tensor_impl.scale.dtype == weight_tensor.dtype
        and len(weight_tensor.tensor_impl.scale.shape) == 1
    )


def _linear_int4_act_int4_weight_cutlass_impl(input_tensor, weight_tensor, bias):
    from torchao.ops import rowwise_scaled_linear_cutlass_s4s4

    weight = weight_tensor.tensor_impl.int_data
    weight_scale = weight_tensor.tensor_impl.scale
    input = input_tensor.tensor_impl.int_data
    input_scale = input_tensor.tensor_impl.scale

    out = rowwise_scaled_linear_cutlass_s4s4(
        input, input_scale, weight, weight_scale, bias
    )

    return out
