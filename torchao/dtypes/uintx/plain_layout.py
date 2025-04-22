# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, PlainLayout
from torchao.kernel import (
    int_scaled_matmul,
)
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
)
from torchao.utils import fill_defaults

aten = torch.ops.aten


def _same_metadata(self: "PlainAQTTensorImpl", src: "PlainAQTTensorImpl") -> bool:
    return (
        isinstance(self, PlainAQTTensorImpl)
        and isinstance(src, PlainAQTTensorImpl)
        and self.shape == src.shape
        and self.int_data.shape == src.int_data.shape
        and self.scale.shape == src.scale.shape
        and (self.zero_point is None and src.zero_point is None)
        or (
            self.zero_point is not None
            and src.zero_point is not None
            and self.zero_point.shape == src.zero_point.shape
        )
        and type(self._layout) == type(src._layout)
    )


@register_layout(PlainLayout)
class PlainAQTTensorImpl(AQTTensorImpl):
    """
    TensorImpl for plain layout for affine quantized tensor, it stores int_data, scale, zero_point
    tensors directly as plain tensors.

    fields:
      int_data (torch.Tensor): the quantized integer data Tensor
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
    """

    def __new__(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
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
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        if self.zero_point is None:
            return ["int_data", "scale"], [self._layout]
        return ["int_data", "scale", "zero_point"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        int_data, scale, zero_point = (
            tensor_data_dict["int_data"],
            tensor_data_dict["scale"],
            tensor_data_dict.get("zero_point", None),
        )
        (_layout,) = tensor_attributes
        return cls(int_data, scale, zero_point, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.int_data.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            self.zero_point.to(kwargs["device"])
            if self.zero_point is not None
            else None,
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.int_data),
            fn(self.scale),
            fn(self.zero_point) if self.zero_point is not None else None,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        elif func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        elif func is aten.copy_.default:
            self = args[0]
            src = args[1]
            if _same_metadata(self, src):
                self_tensors = self.__tensor_flatten__()[0]
                for tensor_name in self_tensors:
                    getattr(self, tensor_name).copy_(getattr(src, tensor_name))
                return
            raise ValueError(
                f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
            )

        elif func is aten.t.default:
            tensor = args[0]
            new = tensor.__class__(
                tensor.int_data.t(), tensor.scale, tensor.zero_point, tensor._layout
            )
            return return_and_correct_aliasing(func, args, kwargs, new)

        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == 0:
                return return_and_correct_aliasing(
                    func,
                    args,
                    kwargs,
                    args[0]._apply_fn_to_data(
                        lambda x: aten.slice.Tensor(x, dim, start, end, step)
                    ),
                )
            elif dim == 1:
                assert len(self.scale.shape) == 1, (
                    f"slice dim==1 only works when len(scale.shape) == 1 currently, got: {self.scale.shape}"
                )
                return PlainAQTTensorImpl(
                    aten.slice.Tensor(self.int_data, dim, start, end, step),
                    self.scale.view(-1),
                    self.zero_point.view(-1) if self.zero_point is not None else None,
                    self._layout,
                )
            else:
                raise NotImplementedError(
                    f"PlainAQTTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"PlainAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.int_data, self.scale, self.zero_point

    def get_layout(self) -> Layout:
        return self._layout

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, PlainLayout)
        return cls(int_data, scale, zero_point, _layout)


def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is int8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.int8
        and (aqt.quant_min is None or aqt.quant_min == -128)
        and (aqt.quant_max is None or aqt.quant_max == 127)
    )


def _aqt_is_int8_reduced_range(aqt):
    return (
        aqt.tensor_impl.dtype == torch.int8
        and aqt.quant_min == -127
        and (aqt.quant_max is None or aqt.quant_max == 127)
    )


def _linear_fp_act_int8_weight_check(input_tensor, weight_tensor, bias):
    return (
        # input is native float tensor
        not is_traceable_wrapper_subclass(input_tensor)
        and input_tensor.is_floating_point()
        and
        # weight is int8 per channel quantized affine quantized tensor
        isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_int8(weight_tensor)
        and len(weight_tensor.shape) == 2
        and len(weight_tensor.block_size) == 2
        and weight_tensor.block_size[0] == 1
        and weight_tensor.block_size[1] == weight_tensor.shape[1]
        and weight_tensor.zero_point_domain == ZeroPointDomain.INT
        and isinstance(weight_tensor._layout, PlainLayout)
    )


def _linear_fp_act_int8_weight_impl(input_tensor, weight_tensor, bias):
    # TODO: enable cpu and mps efficient path
    # is_cpu and is_mps only, some issue with is_contiguous() currently
    # return torch.ops.aten._weight_int8pack_mm(input_tensor.contiguous(), w_vals_int8_t, weight_tensor.tensor_impl.scale)

    # per channel int8 weight only quantizated mm
    w_vals_int8_t = weight_tensor.tensor_impl.int_data.t()
    scale = weight_tensor.tensor_impl.scale
    m = torch.mm(
        input_tensor.reshape(-1, input_tensor.shape[-1]),
        w_vals_int8_t.to(input_tensor.dtype),
    )
    y = m * scale.to(m.dtype)
    y = y.reshape(*input_tensor.shape[:-1], y.shape[-1])
    if bias is not None:
        y += bias.to(m.dtype)
    return y


def _linear_int8_act_int8_weight_check(input_tensor, weight_tensor, bias):
    return (
        isinstance(input_tensor, AffineQuantizedTensor)
        and _aqt_is_int8_reduced_range(input_tensor)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_int8(weight_tensor)
        and input_tensor.dtype == weight_tensor.dtype
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor._layout, PlainLayout)
    )


def _linear_int8_act_int8_weight_impl(input_tensor, weight_tensor, bias):
    #
    # 1. do the matrix form of dot(X_i, W_j)
    #
    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    x_vals_int8 = input_tensor.tensor_impl.int_data
    x_scales = input_tensor.tensor_impl.scale
    w_vals_int8_t = weight_tensor.tensor_impl.int_data.contiguous().t()
    w_scales = weight_tensor.tensor_impl.scale
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    x_scales_dtype = x_scales.dtype
    # Cast fp16 scale to float to avoid overflow in int_scaled_matmul
    intermediate_dtype = torch.float if x_scales_dtype == torch.half else x_scales_dtype
    y_dot_scaled = int_scaled_matmul(
        tmp, w_vals_int8_t, x_scales.reshape(-1, 1).to(intermediate_dtype)
    )
    y_dot_scaled = y_dot_scaled.to(x_scales_dtype)

    y = (y_dot_scaled * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_scaled.shape[-1]
    )

    # can downcast only at the very end
    output_dtype = input_tensor.dtype
    y = y.to(output_dtype)
    if bias is not None:
        y += bias
    return y
