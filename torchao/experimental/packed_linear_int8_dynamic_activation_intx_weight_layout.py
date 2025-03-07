# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import auto, Enum
from typing import Optional, Tuple, Union

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.dtypes.affine_quantized_tensor import register_layout
from torchao.dtypes.affine_quantized_tensor_ops import (
    register_aqt_quantized_linear_dispatch,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.utils import TORCH_VERSION_AT_LEAST_2_6

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Target(Enum):
    """Enum that indicates the backend target"""

    # AUTO target will automatically select a packing format
    # based on the available hardware.
    # TODO: in future, add the ability to specify specific
    # hardware targets
    AUTO = auto()

    # ATEN target will use the ATen operator
    ATEN = auto()


def target_from_str(target: str) -> Target:
    if target.lower() == "auto":
        return Target.AUTO
    elif target.lower() == "aten":
        return Target.ATEN
    else:
        raise ValueError(f"Invalid target: {target}")


class PackedLinearInt8DynamicActivationIntxWeightLayout(Layout):
    bit_width: Optional[int]
    group_size: Optional[int]
    has_weight_zeros: Optional[bool]
    has_bias: Optional[bool]
    target: Optional[Target]

    def __init__(
        self,
        target: Union[str, Target] = "auto",
    ):
        if isinstance(target, str):
            target = target_from_str(target)
        self.target = target

        self.bit_width: Optional[int] = None
        self.group_size: Optional[int] = None
        self.has_weight_zeros: Optional[bool] = None
        # has_bias is whether the packed weights
        # have bias packed with them, not whether the
        # linear operator has bias
        self.has_bias: Optional[bool] = None

    def extra_repr(self):
        return f"group_size={self.group_size}, bit_width={self.bit_width}, has_weight_zeros={self.has_weight_zeros}, has_bias={self.has_bias}, target={self.target}"

    def has_params_set(self) -> bool:
        return (
            (self.bit_width is not None)
            and (self.group_size is not None)
            and (self.has_weight_zeros is not None)
            and (self.has_bias is not None)
            and (self.target is not None)
        )

    def set_params(
        self, bit_width: int, group_size: int, has_weight_zeros: bool, has_bias: bool
    ):
        assert bit_width >= 1 and bit_width <= 8, "bit_width must be 1 to 8"
        assert group_size >= 1, f"group_size must be positive, got {group_size}"

        self.bit_width = bit_width
        self.group_size = group_size
        self.has_weight_zeros = has_weight_zeros
        self.has_bias = has_bias
        assert self.has_params_set()


@register_layout(PackedLinearInt8DynamicActivationIntxWeightLayout)
class PackedLinearInt8DynamicActivationIntxWeightAQTTensorImpl(AQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        _layout: Layout,
        # TODO(T200095131): remove group_size_tensor, n_tensor, k_tensor
        # when AOTI supports int
        group_size_tensor: torch.Tensor,
        n_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
    ):
        kwargs = {}
        kwargs["device"] = packed_weight.device
        kwargs["dtype"] = packed_weight.dtype
        assert not packed_weight.requires_grad
        kwargs["requires_grad"] = False
        shape = packed_weight.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_weight: torch.Tensor,
        _layout: Layout,
        # TODO(T200095131): remove group_size_tensor, n_tensor, k_tensor
        # when AOTI supports int
        group_size_tensor: torch.Tensor,
        n_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
    ):
        assert isinstance(_layout, PackedLinearInt8DynamicActivationIntxWeightLayout)
        self.packed_weight = packed_weight
        self._layout = _layout
        self.group_size_tensor = group_size_tensor
        self.n_tensor = n_tensor
        self.k_tensor = k_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(packed_weight={str(self.packed_weight)}, layout={self.get_layout()})"

    def get_layout(self) -> Layout:
        return self._layout

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError(
            "get_plain is not implemented for PackedLinearInt8DynamicActivationIntxWeightAQTTensorImpl"
        )

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        layout: Layout,
        bias: Optional[torch.Tensor] = None,
    ):
        assert isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout)
        assert (
            layout.has_params_set()
        ), "PackedLinearInt8DynamicActivationIntxWeightLayout params must be set before calling from_plain"
        assert layout.target in {
            Target.AUTO,
            Target.ATEN,
        }, f"Unexpected target: {layout.target}"

        # TODO(T200095131): remove group_size_tensor, n_tensor, k_tensor
        # when AOTI supports int
        n, k = int_data.shape
        group_size_tensor = torch.empty(0, layout.group_size, dtype=torch.int8)
        n_tensor = torch.empty(0, n, dtype=torch.int8)
        k_tensor = torch.empty(0, k, dtype=torch.int8)

        if layout.target == Target.ATEN:
            assert (
                TORCH_VERSION_AT_LEAST_2_6
            ), "aten target is requires torch version > 2.6.0"
            int_data = int_data.add(8)
            int_data = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(torch.uint8)

            # If layout does not have bias packed with the weights, set bias to None
            # It will be applied later in the linear function
            if not layout.has_bias:
                bias = None
            packed_weight = torch.ops.aten._dyn_quant_pack_4bit_weight(
                int_data, scale, bias, layout.group_size, k, n
            )
            return cls(packed_weight, layout, group_size_tensor, n_tensor, k_tensor)

        assert not layout.has_bias, "has_bias is not supported yet"
        if layout.has_weight_zeros:
            args = [
                int_data.to(torch.int8),
                scale.reshape(-1),
                zero_point.reshape(-1).to(torch.int8),
                group_size_tensor,
            ]
        else:
            args = [
                int_data.to(torch.int8),
                scale.reshape(-1),
                group_size_tensor,
            ]

        wzp_suffix = "" if layout.has_weight_zeros else "0zp"
        packed_weight = getattr(
            torch.ops.torchao,
            f"_pack_8bit_act_{layout.bit_width}bit{wzp_suffix}_weight",
        )(*args)

        return cls(packed_weight, layout, group_size_tensor, n_tensor, k_tensor)

    def _apply_fn_to_data(self, fn):
        self.packed_weight = fn(self.packed_weight)

        # TODO(T200095131): remove group_size_tensor, n_tensor, k_tensor
        # when AOTI supports int
        self.group_size_tensor = fn(self.group_size_tensor)
        self.n_tensor = fn(self.n_tensor)
        self.k_tensor = fn(self.k_tensor)
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.ops.aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        if func is torch.ops.aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )

        raise NotImplementedError(
            f"PackedLinearInt8DynamicActivationIntxWeightAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        # TODO(T200095131): remove group_size_tensor, n_tensor, k_tensor
        # when AOTI supports int
        return ["packed_weight", "group_size_tensor", "n_tensor", "k_tensor"], [
            self.get_layout()
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight = tensor_data_dict["packed_weight"]

        # TODO(T200095131): remove group_size_tensor, n_tensor, k_tensor
        # when AOTI supports int
        group_size_tensor = tensor_data_dict["group_size_tensor"]
        n_tensor = tensor_data_dict["n_tensor"]
        k_tensor = tensor_data_dict["k_tensor"]

        (layout,) = tensor_attributes
        return cls(packed_weight, layout, group_size_tensor, n_tensor, k_tensor)


def _linear_check(input_tensor, weight_tensor, bias):
    layout = weight_tensor.tensor_impl.get_layout()
    return isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout)


def _linear_impl(input_tensor, weight_tensor, bias):
    def _impl_2d_auto(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k
        group_size = weight_tensor.tensor_impl.get_layout().group_size

        assert group_size == weight_tensor.tensor_impl.group_size_tensor.shape[1]
        assert n == weight_tensor.tensor_impl.n_tensor.shape[1]
        assert k == weight_tensor.tensor_impl.k_tensor.shape[1]

        assert (
            not weight_tensor.tensor_impl.get_layout().has_bias
        ), "has_bias is not supported yet"

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        args = (
            input_tensor,
            weight_tensor.tensor_impl.packed_weight,
            weight_tensor.tensor_impl.group_size_tensor,
            weight_tensor.tensor_impl.n_tensor,
            weight_tensor.tensor_impl.k_tensor,
        )

        has_weight_zeros = weight_tensor.zero_point_domain != ZeroPointDomain.NONE

        assert len(weight_tensor.block_size) == 2
        assert weight_tensor.block_size[0] == 1
        assert group_size == weight_tensor.block_size[1]
        bit_width = weight_tensor.tensor_impl.get_layout().bit_width

        wzp_suffix = "" if has_weight_zeros else "0zp"
        return getattr(
            torch.ops.torchao, f"_linear_8bit_act_{bit_width}bit{wzp_suffix}_weight"
        )(*args)

    def _impl_2d_aten(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k
        group_size = weight_tensor.tensor_impl.get_layout().group_size
        packed_weight = weight_tensor.tensor_impl.packed_weight
        return torch.ops.aten._dyn_quant_matmul_4bit(
            input_tensor, packed_weight, group_size, k_, n
        )

    target = weight_tensor.tensor_impl.get_layout().target

    if weight_tensor.tensor_impl.get_layout().has_bias:
        assert (
            bias is None
        ), "bias should be None because it is already packed with the weights (has_bias=True)"

    if target == Target.ATEN:
        assert TORCH_VERSION_AT_LEAST_2_6 == 1, "Target.ATEN requires torch >= 2.6.0"
        _impl_2d = _impl_2d_aten
    elif target == Target.AUTO:
        _impl_2d = _impl_2d_auto

    if input_tensor.dim() == 2:
        res = _impl_2d(input_tensor, weight_tensor)
    else:
        assert input_tensor.dim() >= 3
        lead_shape = input_tensor.shape[0:-2]
        m, k = input_tensor.shape[-2], input_tensor.shape[-1]
        n, k_ = weight_tensor.shape
        assert k_ == k

        res = _impl_2d(input_tensor.reshape(-1, k), weight_tensor)
        res = res.reshape(*lead_shape, m, n)

    if bias is not None:
        res = res + bias
    return res


register_aqt_quantized_linear_dispatch(
    _linear_check,
    _linear_impl,
)
