# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import auto, Enum

import logging
from typing import List, Optional, Tuple

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.affine_quantized_tensor import (
    register_layout,
)
from torchao.dtypes.utils import AQTTensorImpl
from torchao.dtypes.affine_quantized_tensor_ops import register_aqt_quantized_linear_dispatch
from torchao.dtypes.utils import Layout
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)

from torchao.quantization.quant_api import to_affine_quantized_intx


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Target(Enum):
    """Enum that indicates the backend target
    """
    NATIVE = auto()
    FALLBACK = auto()

def target_from_str(target: str) -> Target:
    if target.lower() == "native":
        return Target.NATIVE
    elif target.lower() == "fallback":
        return Target.FALLBACK
    else:
        raise ValueError(f"Invalid target: {target}")


# This format is intended for use with int8 dynamic quantization
class Linear8BitActXBitWeightLayout(Layout):
    nbit: int
    group_size: int

    # The target platform for the layout, either 'native' or 'fallback'.
    target: Target

    def __init__(
        self,
        nbit: int,
        group_size: int,
        target: str,
    ):
        assert nbit <= 8
        self.nbit = nbit
        self.group_size = group_size
        self.target = target_from_str(target)

    def extra_repr(self):
        return f"nbit={self.nbit}, group_size={self.group_size}, target={self.target}"


def _pack_weights_native(
    int_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    layout: Layout,
):
    assert isinstance(layout, Linear8BitActXBitWeightLayout)
    assert layout.target == Target.NATIVE
    nbit = layout.nbit
    group_size = layout.group_size
    has_weight_zeros = zero_point is not None

    if has_weight_zeros:
        args = [
            int_data.to(torch.int8),
            scale.reshape(-1),
            zero_point.reshape(-1).to(torch.int8),
            torch.empty(0, group_size, dtype=torch.int8),
        ]
    else:
        args = [
            int_data.to(torch.int8),
            scale.reshape(-1),
            torch.empty(0, group_size, dtype=torch.int8),
        ]

    wzp_suffix = "" if has_weight_zeros else "0zp"
    return getattr(torch.ops.torchao, f"_pack_8bit_act_{nbit}bit{wzp_suffix}_weight")(
        *args
    )


@register_layout(Linear8BitActXBitWeightLayout)
class Linear8BitActXBitWeightAQTTensorImpl(AQTTensorImpl):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
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
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        assert isinstance(_layout, Linear8BitActXBitWeightLayout)

        # In the native case, scale and zero_point information is inside
        # the packed_weight
        if _layout.target == Target.NATIVE:
            assert scale is None
            assert zero_point is None

        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self._layout = _layout

    def __repr__(self):
        layout = self.get_layout()
        return f"{self.__class__.__name__}(packed_weight={str(self.packed_weight)}, scale={str(self.scale)}, zero_point={str(self.zero_point)}, layout={layout})"

    def get_layout(self) -> Layout:
        return self._layout

    def get_plain(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.get_layout().target == Target.FALLBACK:
            return self.packed_weight, self.scale, self.zero_point
        raise NotImplementedError("get_plain is not supported for Linear8BitActXBitWeightAQTTensorImpl when target is not fallback")

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout: Layout,
    ):
        assert isinstance(layout, Linear8BitActXBitWeightLayout)

        try:
            if layout.target == Target.NATIVE:
                packed_weight = _pack_weights_native(
                    int_data, scale, zero_point, layout
                )
                scale = None
                zero_point = None
                return cls(packed_weight, scale, zero_point, layout)
        except Exception as e:
            logger.warning(
                f"A failure occurred when packing weights with Linear8BitActXBitWeightLayout.target={layout.target}: {e}\n"
                + "Falling back to **slow** implementation Linear8BitActXBitWeightLayout.target=fallback."
            )
            layout.target = Target.FALLBACK

        # Fallback
        assert layout.target == Target.FALLBACK
        packed_weight = int_data.to(torch.int32)
        return cls(packed_weight, scale, zero_point, layout)

    def _apply_fn_to_data(self, fn):
        self.packed_weight = fn(self.packed_weight)
        if self.scale is not None:
            self.scale = fn(self.scale)

        if self.zero_point is not None:
            self.zero_point = fn(self.zero_point)
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
            f"Linear8BitActXBitWeightAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        if self.get_layout().target == Target.NATIVE:
            return ["packed_weight"], [self.get_layout()]

        # fallback
        assert self.get_layout().target == Target.FALLBACK
        if self.zero_point is None:
            return ["packed_weight", "scale"], [self.get_layout()]
        return ["packed_weight", "scale", "zero_point"], [self.get_layout()]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale, zero_point = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict.get("scale", None),
            tensor_data_dict.get("zero_point", None),
        )
        (layout,) = tensor_attributes
        return cls(packed_weight, scale, zero_point, layout)


def _linear_int8_dynamic_activation_intx_weight_check(
    input_tensor, weight_tensor, bias
):
    layout = weight_tensor.tensor_impl.get_layout()
    return isinstance(layout, Linear8BitActXBitWeightLayout) and bias is None


def _linear_int8_dynamic_activation_intx_weight_fallback_impl(
    input_tensor, weight_tensor, bias
):
    assert weight_tensor.tensor_impl.get_layout().target == Target.FALLBACK
    assert bias is None

    def _impl_2d(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        weight_qvals = weight_tensor.tensor_impl.packed_weight.to(torch.int32)
        weight_scales = weight_tensor.tensor_impl.scale
        weight_zeros = weight_tensor.tensor_impl.zero_point
        group_size = weight_tensor.tensor_impl.get_layout().group_size
        has_weight_zeros = weight_zeros is not None
        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k

        weights_dequantized = weight_tensor.dequantize()

        # Quantize activations
        activations_dequantized = to_affine_quantized_intx(
            input_tensor,
            mapping_type=MappingType.ASYMMETRIC,
            block_size=(1, k),
            target_dtype=torch.int32,
            quant_min=-128,
            quant_max=127,
            eps=0.0,
            zero_point_dtype=torch.int32,
            preserve_zero=True,
            zero_point_domain=ZeroPointDomain.INT,
            use_hqq=False,
        ).dequantize()

        return torch.matmul(
            activations_dequantized, weights_dequantized.transpose(1, 0)
        )

    if input_tensor.dim() == 2:
        return _impl_2d(input_tensor, weight_tensor)

    assert input_tensor.dim() >= 3
    lead_shape = input_tensor.shape[0:-2]
    m, k = input_tensor.shape[-2], input_tensor.shape[-1]
    n, k_ = weight_tensor.shape
    assert k_ == k

    res = _impl_2d(input_tensor.reshape(-1, k), weight_tensor)
    res = res.reshape(*lead_shape, m, n)

    return res


def _linear_int8_dynamic_activation_intx_weight_native_impl(
    input_tensor, weight_tensor, bias
):
    assert weight_tensor.tensor_impl.get_layout().target == Target.NATIVE
    assert bias is None

    def _impl_2d(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k
        group_size = weight_tensor.tensor_impl.get_layout().group_size
        packed_weight = weight_tensor.tensor_impl.packed_weight

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        args = (
            input_tensor,
            packed_weight,
            torch.empty(0, group_size, dtype=torch.int8),
            torch.empty(0, n, dtype=torch.int8),
            torch.empty(0, k, dtype=torch.int8),
        )

        has_weight_zeros = (weight_tensor.zero_point_domain != ZeroPointDomain.NONE)

        assert len(weight_tensor.block_size) == 2
        assert weight_tensor.block_size[0] == 1
        group_size = weight_tensor.block_size[1]
        assert group_size == weight_tensor.tensor_impl.get_layout().group_size
        nbit = weight_tensor.tensor_impl.get_layout().nbit

        n, k = weight_tensor.shape
        m, k_ = input_tensor.shape
        assert k_ == k

        packed_weight = weight_tensor.tensor_impl.packed_weight
        wzp_suffix = "" if has_weight_zeros else "0zp"
        return getattr(
            torch.ops.torchao, f"_linear_8bit_act_{nbit}bit{wzp_suffix}_weight"
        )(*args)

    if input_tensor.dim() == 2:
        return _impl_2d(input_tensor, weight_tensor)

    assert input_tensor.dim() >= 3
    lead_shape = input_tensor.shape[0:-2]
    m, k = input_tensor.shape[-2], input_tensor.shape[-1]
    n, k_ = weight_tensor.shape
    assert k_ == k

    res = _impl_2d(input_tensor.reshape(-1, k), weight_tensor)
    res = res.reshape(*lead_shape, m, n)
    return res


def _linear_int8_dynamic_activation_intx_weight_impl(input_tensor, weight_tensor, bias):
    target = weight_tensor.tensor_impl.get_layout().target
    if target == Target.NATIVE:
        return _linear_int8_dynamic_activation_intx_weight_native_impl(
            input_tensor, weight_tensor, bias
        )

    if target == Target.FALLBACK:
        return _linear_int8_dynamic_activation_intx_weight_fallback_impl(
            input_tensor, weight_tensor, bias
        )

    assert False, f"Unknown target {target}"


register_aqt_quantized_linear_dispatch(
    _linear_int8_dynamic_activation_intx_weight_check,
    _linear_int8_dynamic_activation_intx_weight_impl,
)
