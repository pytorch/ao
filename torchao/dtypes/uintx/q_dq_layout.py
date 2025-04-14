# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


from dataclasses import dataclass
from typing import Optional, Tuple

from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.utils import fill_defaults

aten = torch.ops.aten


@dataclass(frozen=True)
class QDQLayout(Layout):
    pass


def _same_metadata(self: "QDQTensorImpl", src: "QDQTensorImpl") -> bool:
    return (
        isinstance(self, QDQTensorImpl)
        and isinstance(src, QDQTensorImpl)
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


@register_layout(QDQLayout)
class QDQTensorImpl(AQTTensorImpl):
    """
    TensorImpl for QDQLayout layout for affine quantized tensor, it stores int_data, scale, zero_point
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
            if dim in [0, 1]:
                int_data, scale, zero_point = self.get_plain()
                data_len = int_data.shape[dim]
                scale_len = scale.shape[dim]
                ratio = data_len / scale_len
                start_scale = int(start / ratio)
                end_scale = int(end / ratio)

                int_data = aten.slice.Tensor(int_data, dim, start, end, step)
                scale = aten.slice.Tensor(scale, dim, start_scale, end_scale, step)
                if zero_point is not None:
                    zero_point = aten.slice.Tensor(
                        zero_point, dim, start_scale, end_scale, step
                    )
                sliced = self.from_plain(int_data, scale, zero_point, self._layout)
                return return_and_correct_aliasing(func, args, kwargs, sliced)
            else:
                raise NotImplementedError(
                    f"QDQTensorImpl dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"QDQTensorImpl dispatch: attempting to run {func}, this is not supported"
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
        assert isinstance(_layout, QDQLayout)
        return cls(int_data, scale, zero_point, _layout)


def _linear_check(input_tensor, weight_tensor, bias):
    layout = weight_tensor.tensor_impl.get_layout()
    return isinstance(layout, QDQLayout)


def _linear_impl(input_tensor, weight_tensor, bias):
    if isinstance(input_tensor, AffineQuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight_tensor, AffineQuantizedTensor):
        weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias)


def _embedding_check(args, kwargs):
    _, weight_tensor = args
    layout = weight_tensor.tensor_impl.get_layout()
    return isinstance(layout, QDQLayout)


def _embedding_impl(args, kwargs):
    input_tensor, weight_tensor = args
    if isinstance(weight_tensor, AffineQuantizedTensor):
        weight_tensor = weight_tensor.dequantize()
    return torch.nn.functional.embedding(input_tensor, weight_tensor, **kwargs)
