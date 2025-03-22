# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum, auto
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
    AUTO = auto()
    UNIVERSAL = auto()
    KLEIDIAI = auto()

    # ATEN target will use the ATen operator
    ATEN = auto()


_TARGET_AND_STR = [
    (Target.AUTO, "auto"),
    (Target.ATEN, "aten"),
    (Target.UNIVERSAL, "universal"),
    (Target.KLEIDIAI, "kleidiai"),
]


def target_to_str(target: Target) -> str:
    target_to_str = {t: s for t, s in _TARGET_AND_STR}
    return target_to_str[target]


def target_from_str(target: str) -> Target:
    str_to_target = {s: t for t, s in _TARGET_AND_STR}
    if target.lower() in str_to_target:
        return str_to_target[target.lower()]
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
    ):
        assert isinstance(_layout, PackedLinearInt8DynamicActivationIntxWeightLayout)
        self.packed_weight = packed_weight
        self._layout = _layout

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
        assert layout.has_params_set(), "PackedLinearInt8DynamicActivationIntxWeightLayout params must be set before calling from_plain"
        assert layout.target in [
            t for t, _ in _TARGET_AND_STR
        ], f"Unexpected target: {layout.target}"

        n, k = int_data.shape
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
            return cls(packed_weight, layout)

        args = [
            int_data.to(torch.int8),
            scale.reshape(-1),
            zero_point.reshape(-1).to(torch.int8) if layout.has_weight_zeros else None,
            layout.group_size,
            bias if layout.has_bias else None,
            target_to_str(layout.target) if layout.target != Target.AUTO else None,
        ]

        packed_weight = getattr(
            torch.ops.torchao,
            f"_pack_8bit_act_{layout.bit_width}bit_weight",
        )(*args)

        return cls(packed_weight, layout)

    def _apply_fn_to_data(self, fn):
        self.packed_weight = fn(self.packed_weight)
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
        return ["packed_weight"], [self.get_layout()]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight = tensor_data_dict["packed_weight"]
        (layout,) = tensor_attributes
        return cls(packed_weight, layout)


def _linear_check(input_tensor, weight_tensor, bias):
    layout = weight_tensor.tensor_impl.get_layout()
    return isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout)


def _linear_impl(input_tensor, weight_tensor, bias):
    def _impl_2d_non_aten(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k
        group_size = weight_tensor.tensor_impl.get_layout().group_size

        args = (
            input_tensor,
            weight_tensor.tensor_impl.packed_weight,
            group_size,
            n,
            k,
        )

        assert len(weight_tensor.block_size) == 2
        assert weight_tensor.block_size[0] == 1
        assert group_size == weight_tensor.block_size[1]
        bit_width = weight_tensor.tensor_impl.get_layout().bit_width

        return getattr(torch.ops.torchao, f"_linear_8bit_act_{bit_width}bit_weight")(
            *args
        )

    def _impl_2d_aten(input_tensor, weight_tensor):
        assert input_tensor.dim() == 2
        assert weight_tensor.dim() == 2

        m, k = input_tensor.shape
        n, k_ = weight_tensor.shape
        assert k_ == k
        group_size = weight_tensor.tensor_impl.get_layout().group_size
        packed_weight = weight_tensor.tensor_impl.packed_weight
        return torch.ops.aten._dyn_quant_matmul_4bit(
            input_tensor, packed_weight, group_size, k, n
        )

    target = weight_tensor.tensor_impl.get_layout().target

    if weight_tensor.tensor_impl.get_layout().has_bias:
        assert (
            bias is None
        ), "bias should be None because it is already packed with the weights (has_bias=True)"

    if target == Target.ATEN:
        assert TORCH_VERSION_AT_LEAST_2_6 == 1, "Target.ATEN requires torch >= 2.6.0"
        _impl_2d = _impl_2d_aten
    else:
        _impl_2d = _impl_2d_non_aten

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

import math

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    get_tensor_impl_constructor,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout, PlainLayout
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
    choose_qparams_and_quantize_affine_hqq,
    quantize_affine,
)


class _AffineQuantizedTensor(AffineQuantizedTensor):
    """
    PackedLinearInt8DynamicActivationIntxWeightAtenTensor quantized tensor subclass which inherits AffineQuantizedTensor class.
    """

    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = True,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
        _layout: Layout = PlainLayout(),
        use_hqq: bool = False,
        tensor_impl_ctr_kwargs: Optional[dict] = None,
    ):
        """Convert a high precision tensor to an integer affine quantized tensor."""
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)

        if use_hqq:
            assert (
                zero_point_domain == ZeroPointDomain.FLOAT
                and mapping_type == MappingType.ASYMMETRIC
                and quant_min == 0
            ), "Invalid input parameters for HQQ quantization."
            nbits = int(math.log2(quant_max + 1))
            axis = 1 if (block_size[0] == 1) else 0
            group_size = max(block_size)
            compute_dtype = (
                zero_point_dtype
                if (zero_point_dtype is not None)
                else input_float.dtype
            )
            device = input_float.device
            from torchao.dtypes.uintx import TensorCoreTiledLayout

            data, scale, zero_point, _ = choose_qparams_and_quantize_affine_hqq(
                input_float,
                nbits=nbits,
                group_size=group_size,
                axis=axis,
                compute_dtype=compute_dtype,
                device=device,
                verbose=False,
                raw_output=not isinstance(
                    _layout, (TensorCoreTiledLayout, PlainLayout)
                ),
                # raw_output=False is basically the 'convert to TensorCoreTiledLayout zero_point version' option (add scale*midpoint)
                # note in choose_qparams_affine, preserve_zero = False does this same thing while also controlling whether
                # zero is preserved.
                # TODO uncouple preserve_zero and conversion of zero_point to TensorCoreTiledLayout version
                # TODO move the conversion of zero_point out of quant_primitives and into TensorCoreTiledLayout.from_plain
                # TODO change PlainLayout to use raw_output.
            )
            data = data.to(target_dtype)
        else:
            scale, zero_point = choose_qparams_affine(
                input_float,
                mapping_type,
                block_size,
                target_dtype,
                quant_min,
                quant_max,
                eps,
                scale_dtype,
                zero_point_dtype,
                preserve_zero,
                zero_point_domain,
            )
            # choose_qparams_affine is a custom op that does support returning optional Tensors. We thus set the zero_point to None if its domain is None
            if zero_point_domain == ZeroPointDomain.NONE:
                zero_point = None
            data = quantize_affine(
                input_float,
                block_size,
                scale,
                zero_point,
                target_dtype,
                quant_min,
                quant_max,
                zero_point_domain,
            )
            # Note: output will be uint8 tensor for sub byte tensors for now

        data = _layout.post_process(data)
        tensor_impl_ctr = get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(
            data, scale, zero_point, _layout, **(tensor_impl_ctr_kwargs or {})
        )
        return cls(
            tensor_impl,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype,
        )


to_affine_quantized_intx_experimental = _AffineQuantizedTensor.from_hp_to_intx
