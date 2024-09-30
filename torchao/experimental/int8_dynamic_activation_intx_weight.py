# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple

import torch
from torch.ao.quantization.fx._decomposed import (
    dequantize_per_channel_group,
    quantize_per_channel_group,
)
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.dtypes.affine_quantized_tensor import (
    AQTLayout,
    register_aqt_quantized_linear_dispatch,
    register_layout_cls,
)
from torchao.dtypes.utils import LayoutType
from torchao.utils import TorchAOBaseTensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import sys

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# TODO: replace with torchao API
def _quantize(vals: torch.Tensor, group_size: int, nbit: int, has_weight_zeros: bool):
    assert nbit >= 1 and nbit <= 8
    qmin = -(1 << (nbit - 1))
    qmax = (1 << (nbit - 1)) - 1

    n, k = vals.shape
    vals = vals.reshape(-1, group_size)
    vmins, _ = torch.min(vals, axis=1)
    vmaxs, _ = torch.max(vals, axis=1)
    group_scales = (vmaxs - vmins) / (qmax - qmin)

    if not has_weight_zeros:
        group_zeros = torch.zeros_like(group_scales)
    else:
        group_zeros = qmin - torch.round(vmins / group_scales)

    vals = vals.reshape(n, k)
    group_scales = group_scales.reshape(n, -1)
    group_zeros = group_zeros.reshape(n, -1)

    group_qvals = quantize_per_channel_group(
        input=vals,
        scales=group_scales,
        zero_points=group_zeros,
        quant_min=qmin,
        quant_max=qmax,
        dtype=torch.int8,
        group_size=group_size,
    )

    if not has_weight_zeros:
        group_zeros = None

    return group_qvals, group_scales, group_zeros


_VALID_TARGETS: List[str] = ["native", "fallback"]


def _target_equal(target, other):
    assert other in _VALID_TARGETS
    assert target in _VALID_TARGETS
    return target == other


# This format is intended for use with int8 dynamic quantization
class IntxWeightLayoutType(LayoutType):
    nbit: int
    group_size: int

    # The target platform for the layout, either 'native' or 'fallback'.
    target: str

    def __init__(
        self,
        nbit: int,
        group_size: int,
        target: str,
    ):
        assert nbit <= 7
        self.nbit = nbit
        self.group_size = group_size
        assert target in _VALID_TARGETS
        self.target = target

    def extra_repr(self):
        return f"nbit={self.nbit}, group_size={self.group_size}, target={self.target}"


def _pack_weights_native(
    int_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    layout_type: LayoutType,
):
    assert isinstance(layout_type, IntxWeightLayoutType)
    assert _target_equal(layout_type.target, "native")
    nbit = layout_type.nbit
    group_size = layout_type.group_size
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

    wzp_suffix = "z" if has_weight_zeros else ""
    return getattr(torch.ops.torchao, f"_pack_weights_a8sz_w{nbit}s{wzp_suffix}")(*args)


@register_layout_cls(IntxWeightLayoutType)
class IntxWeightAQTLayout(AQTLayout):
    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scale: Optional[torch.Tensor],
        zero_point: Optional[torch.Tensor],
        layout_type: LayoutType,
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
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, IntxWeightLayoutType)

        # In the native case, scale and zero_point information is inside
        # the packed_weight
        if _target_equal(layout_type.target, "native"):
            assert scale is None
            assert zero_point is None

        self.packed_weight = packed_weight
        self.scale = scale
        self.zero_point = zero_point
        self.layout_type = layout_type

    def __repr__(self):
        layout_type = self.get_layout_type()
        return f"{self.__class__.__name__}(packed_weight={str(self.packed_weight)}, scale={str(self.scale)}, zero_point={str(self.zero_point)}, layout_type={layout_type})"

    def get_layout_type(self) -> LayoutType:
        return self.layout_type

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        layout_type: LayoutType,
    ):
        assert isinstance(layout_type, IntxWeightLayoutType)

        try:
            if _target_equal(layout_type.target, "native"):
                packed_weight = _pack_weights_native(
                    int_data, scale, zero_point, layout_type
                )
                scale = None
                zero_point = None
                return cls(packed_weight, scale, zero_point, layout_type)
        except Exception as e:
            logger.warning(
                f"A failure occurred when packing weights with IntxWeightLayoutType.target={layout_type.target}: {e}\n"
                + "Falling back to **slow** implementation IntxWeightLayoutType.target=fallback."
            )
            layout_type.target = "fallback"

        # Fallback
        assert _target_equal(layout_type.target, "fallback")
        packed_weight = int_data.to(torch.int8)
        return cls(packed_weight, scale, zero_point, layout_type)

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
            f"IntxWeightAQTLayout dispatch: attempting to run {func}, this is not supported"
        )

    def __tensor_flatten__(self):
        if _target_equal(self.get_layout_type().target, "native"):
            return ["packed_weight"], [self.get_layout_type()]

        # fallback
        assert _target_equal(self.get_layout_type().target, "fallback")
        if self.zero_point is None:
            return ["packed_weight", "scale"], [self.get_layout_type()]
        return ["packed_weight", "scale", "zero"], [self.get_layout_type()]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scale, zero_point = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict.get("scale", None),
            tensor_data_dict.get("zero_point", None),
        )
        (layout_type,) = tensor_attributes
        return cls(packed_weight, scale, zero_point, layout_type)


def _linear_int8_dynamic_activation_intx_weight_check(
    input_tensor, weight_tensor, bias
):
    layout_type = weight_tensor.layout_tensor.layout_type
    return isinstance(layout_type, IntxWeightLayoutType) and bias is None


def _linear_int8_dynamic_activation_intx_weight_fallback_impl(
    input_tensor, weight_tensor, bias
):
    assert weight_tensor.layout_tensor.layout_type.target == "fallback"
    assert bias is None

    def _impl_2d(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        weight_qvals = weight_tensor.layout_tensor.packed_weight.to(torch.int32)
        weight_scales = weight_tensor.layout_tensor.scale
        weight_zeros = weight_tensor.layout_tensor.zero_point
        group_size = weight_tensor.layout_tensor.layout_type.group_size
        has_weight_zeros = weight_zeros is not None
        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k

        weights_dequantized = dequantize_per_channel_group(
            w_int8=weight_qvals,
            scales=weight_scales,
            zero_points=(
                weight_zeros if has_weight_zeros else torch.zeros_like(weight_scales)
            ),
            quant_min=None,  # TODO: why is this an arg for this function
            quant_max=None,  # TODO: why is this an arg for this function
            dtype=None,  # TODO: why is this an arg for this function
            group_size=group_size,
            output_dtype=torch.float32,
        )

        activation_qvals, activation_scales, activation_zeros = _quantize(
            input_tensor, group_size=k, nbit=8, has_weight_zeros=True
        )
        activations_dequantized = dequantize_per_channel_group(
            w_int8=activation_qvals,
            scales=activation_scales,
            zero_points=activation_zeros,
            quant_min=None,  # TODO: why is this an arg for this function
            quant_max=None,  # TODO: why is this an arg for this function
            dtype=None,  # TODO: why is this an arg for this function
            group_size=k,
            output_dtype=torch.float32,
        )

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

    input_tensor = input_tensor.reshape(-1, m, k)

    res = [_impl_2d(input_tensor[i, :, :], weight_tensor) for i in range(input_tensor.shape[0])]
    res = torch.stack(res)
    res = res.reshape(*lead_shape, m, n)
    return res


def _linear_int8_dynamic_activation_intx_weight_native_impl(
    input_tensor, weight_tensor, bias
):
    assert weight_tensor.layout_tensor.layout_type.target == "native"
    assert bias is None

    def _impl_2d(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k
        group_size = weight_tensor.layout_tensor.layout_type.group_size
        packed_weight = weight_tensor.layout_tensor.packed_weight

        # TODO(T200095131): convert self.n, self.k, self.group_size to
        # int when supported by AOTI
        args = (
            packed_weight,
            torch.empty(0, n, dtype=torch.int8),
            torch.empty(0, k, dtype=torch.int8),
            torch.empty(0, group_size, dtype=torch.int8),
            input_tensor,
        )

        has_weight_zeros = weight_tensor.zero_point_domain is not None

        assert len(weight_tensor.block_size) == 2
        assert weight_tensor.block_size[0] == 1
        group_size = weight_tensor.block_size[1]
        assert group_size == weight_tensor.layout_tensor.layout_type.group_size
        nbit = weight_tensor.layout_tensor.layout_type.nbit

        n, k = weight_tensor.shape
        m, k_ = input_tensor.shape
        assert k_ == k

        packed_weight = weight_tensor.layout_tensor.packed_weight
        wzp_suffix = "z" if has_weight_zeros else ""
        return getattr(torch.ops.torchao, f"_linear_a8sz_w{nbit}s{wzp_suffix}")(*args)

    if input_tensor.dim() == 2:
        return _impl_2d(input_tensor, weight_tensor)

    assert input_tensor.dim() >= 3
    lead_shape = input_tensor.shape[0:-2]
    m, k = input_tensor.shape[-2], input_tensor.shape[-1]
    n, k_ = weight_tensor.shape
    assert k_ == k

    input_tensor = input_tensor.reshape(-1, m, k)

    res = [_impl_2d(input_tensor[i, :, :], weight_tensor) for i in range(input_tensor.shape[0])]
    res = torch.stack(res)
    res = res.reshape(*lead_shape, m, n)
    return res


def _linear_int8_dynamic_activation_intx_weight_impl(input_tensor, weight_tensor, bias):
    target = weight_tensor.layout_tensor.layout_type.target
    if _target_equal(target, "native"):
        return _linear_int8_dynamic_activation_intx_weight_native_impl(
            input_tensor, weight_tensor, bias
        )

    if _target_equal(target, "fallback"):
        return _linear_int8_dynamic_activation_intx_weight_fallback_impl(
            input_tensor, weight_tensor, bias
        )

    assert False, f"Unknown target {target}"


register_aqt_quantized_linear_dispatch(
    _linear_int8_dynamic_activation_intx_weight_check,
    _linear_int8_dynamic_activation_intx_weight_impl,
)
