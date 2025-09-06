# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass
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
from torchao.dtypes.utils import AQTTensorImpl, Layout, is_device
from torchao.quantization.quant_primitives import (
    ZeroPointDomain,
    _get_reduction_params,
    _quantize_affine_int4_common,
)
from torchao.utils import (
    fill_defaults,
)

aten = torch.ops.aten


def _same_metadata(self: "Int4StorageQuantizationAQTTensorImpl", src: "Int4StorageQuantizationAQTTensorImpl") -> bool:
    return (
        isinstance(self, Int4StorageQuantizationAQTTensorImpl)
        and isinstance(src, Int4StorageQuantizationAQTTensorImpl)
        and self.shape == src.shape
        and self.packed_weight.shape == src.packed_weight.shape
        and self.scale.shape == src.scale.shape
        and (self.zero_point is None and src.zero_point is None)
        or (
            self.zero_point is not None
            and src.zero_point is not None
            and self.zero_point.shape == src.zero_point.shape
        )
        and type(self._layout) == type(src._layout)
    )


@dataclass(frozen=True)
class Int4StorageQuantizationLayout(Layout):
    """Layout class for int4 storage quantization layout for affine quantized tensor."""

    pass


@register_layout(Int4StorageQuantizationLayout)
class Int4StorageQuantizationAQTTensorImpl(AQTTensorImpl):
    """TensorImpl for int4 storage quantization layout for affine quantized tensor, this is for int4 only,

    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 2-d tensor of
    dimension: [n][k / 2] (uint8 dtype)
    (unpacked Tensor shape is n * k)

    fields:
      packed_weight (torch.Tensor): the 2-d packed tensor in a Int4 storage quantization layout
      scale (torch.Tensor): the scale Tensor used to map between floating point tensor to quantized tensor
      zero_point (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["layout"] = (
            kwargs.get("layout")
            if kwargs.get("layout", False)
            else packed_weight.layout
        )
        kwargs["dtype"] = packed_weight.dtype
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __tensor_flatten__(self):
        if self.zero_point is None:
            return ["packed_weight", "scale"], [self._layout]
        return ["packed_weight", "scale", "zero_point"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale, zero_point = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scale"],
            tensor_data_dict.get("zero_point", None),
        )
        (_layout,) = tensor_attributes
        return cls(packed_weight, scale, zero_point, _layout)

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        # H TODO
        device = kwargs["device"]
        if not is_device(torch.device(self.device).type, device):
            raise ValueError(
                f"{self.__class__.__name__} does not support conversion from {self.device} to {device}"
            )

        return self.__class__(
            self.packed_weight.to(kwargs["device"]),
            self.scale.to(kwargs["device"]),
            (
                self.zero_point.to(kwargs["device"])
                if self.zero_point is not None
                else None
            ),
            self._layout,
        )

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scale),
            fn(self.zero_point) if self.zero_point is not None else None,
            self._layout,
        )

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, Int4StorageQuantizationLayout)

        assert int_data.dtype == torch.int32, " int_data expects `int32` dtype"

        def quant_2d(int_data_2d):
            int_data_2d = (int_data_2d[::, ::2] << 4 | int_data_2d[::, 1::2]).to(
                torch.uint8
            )
            return int_data_2d

        if int_data.dim() == 2:
            packed_weight = quant_2d(int_data)
        else:
            raise NotImplementedError(
                f"Int4StorageQuantizationAQTTensorImpl Not supported dim {int_data.dim()} quantization"
            )

        return cls(packed_weight, scale, zero_point, _layout)

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

        elif func in [aten.select.int, aten.index.Tensor]:
            assert not (
                func is aten.select.int and args[1] != 0
            ), "aten.select.int currently only has support for dim=0"
            return return_and_correct_aliasing(
                func,
                args,
                kwargs,
                args[0]._apply_fn_to_data(lambda x: func(x, *args[1:], **kwargs)),
            )

        elif func is aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            cur_shape = self.shape
            assert len(cur_shape) == 2

            if dim == 0:
                return return_and_correct_aliasing(
                    func,
                    args,
                    kwargs,
                    args[0]._apply_fn_to_data(
                        lambda x: aten.slice.Tensor(x, dim, start, end, step)
                    ),
                )
            else:
                raise NotImplementedError(
                    f"{cls.__name__} dispatch: attempting to run {func}, with dim={dim}, that is not supported"
                )

        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        high = (self.packed_weight >> 4).to(torch.int32)
        low = (self.packed_weight & 0xF).to(torch.int32)

        original_shape = (self.packed_weight.shape[0], self.packed_weight.shape[1] * 2)
        original_data = torch.empty(
            original_shape, dtype=torch.int32, device=self.device
        )
        original_data[::, ::2] = high
        original_data[::, 1::2] = low

        return original_data, self.scale, self.zero_point

    def get_layout(self) -> Layout:
        return self._layout
