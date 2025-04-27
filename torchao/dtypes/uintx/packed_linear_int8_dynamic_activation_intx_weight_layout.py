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
from torchao.dtypes.utils import AQTTensorImpl, Layout
from torchao.experimental.op_lib_utils import _check_torchao_ops_loaded
from torchao.quantization.quant_primitives import (
    _DTYPE_TO_BIT_WIDTH,
    _DTYPE_TO_QVALUE_BOUNDS,
    ZeroPointDomain,
)
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
        *,
        validate_inputs: bool = True,
    ):
        assert isinstance(layout, PackedLinearInt8DynamicActivationIntxWeightLayout)
        assert layout.target in [t for t, _ in _TARGET_AND_STR], (
            f"Unexpected target: {layout.target}"
        )
        assert layout.has_params_set(), (
            "PackedLinearInt8DynamicActivationIntxWeightLayout params must be set before calling from_plain"
        )

        if layout.target != Target.ATEN:
            _check_torchao_ops_loaded()
        else:
            assert TORCH_VERSION_AT_LEAST_2_6, (
                "aten target is requires torch version > 2.6.0"
            )
            assert torch.backends.kleidiai.is_available(), (
                "ATEN target requires torch.backends.kleidiai.is_available()"
            )
            layout.bit_width == 4, "ATEN target only supports torch.int4"
            assert not layout.has_weight_zeros, "ATEN target does not support zeros"

        data_dtype = getattr(torch, f"int{layout.bit_width}")
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[data_dtype]

        int_types = [torch.int8, torch.int16, torch.int32, torch.int64]

        # Check int_data
        assert int_data.device == torch.device("cpu")
        assert int_data.dtype in int_types
        n, k = int_data.shape
        assert k % layout.group_size == 0, "k must be divisible by group_size"
        if validate_inputs:
            assert int_data.min().item() >= qmin
            assert int_data.max().item() <= qmax
        int_data = int_data.to(torch.int8)

        # Check scale
        assert scale.device == torch.device("cpu")
        if scale.dtype != torch.float32:
            logging.info(f"scale has dtype {scale.dtype}, converting to torch.float32")
            scale = scale.to(torch.float32)
        n_, _ = scale.shape
        assert n_ == n
        assert scale.numel() * layout.group_size == int_data.numel(), (
            "must have 1 scale per group"
        )
        if validate_inputs:
            assert scale.min().item() > 0
            # Some targets round scales to bfloat16, give warning if scales are at higher precision
            scale_is_rounded_to_bf16 = torch.allclose(
                scale, scale.to(torch.bfloat16).to(torch.float32)
            )
            if not scale_is_rounded_to_bf16:
                if layout.target == Target.ATEN and (layout.group_size < k):
                    logging.warning(
                        "When using Target.ATEN with group_size < k, scales will be rounded to bfloat16"
                    )
                if layout.target in [Target.AUTO, Target.KLEIDIAI]:
                    logging.warning(
                        "When using [Target.AUTO, Target.KLEIDIAI], scales will be rounded to bfloat16"
                    )

        # Check zero_point
        if zero_point is None:
            assert not layout.has_weight_zeros, (
                "zero_point must be provided if has_weight_zeros=True"
            )
        else:
            assert zero_point.device == torch.device("cpu")
            assert zero_point.shape == scale.shape
            assert zero_point.dtype in int_types
            assert zero_point.numel() * layout.group_size == int_data.numel(), (
                "must have 1 zero_point per group"
            )
            if validate_inputs:
                zero_point_min = zero_point.min().item()
                zero_point_max = zero_point.max().item()
                assert zero_point.min().item() >= qmin
                assert zero_point.max().item() <= qmax
                has_weight_zeros = True
                if zero_point_min == 0 and zero_point_max == 0:
                    has_weight_zeros = False
                assert has_weight_zeros == layout.has_weight_zeros, (
                    "zero_point being all zeros must be consistent with layout.has_weight_zeros"
                )
            zero_point = zero_point.to(torch.int8)

        # Check bias
        has_bias = bias is not None
        assert has_bias == layout.has_bias, (
            "bias being None must be consistent with layout.has_bias"
        )
        if has_bias:
            assert bias.device == torch.device("cpu")
            if bias.dtype != torch.float32:
                logging.info(
                    f"bias has dtype {bias.dtype}, converting to torch.float32"
                )
                bias = bias.to(torch.float32)
            assert bias.shape == (n,)

        # Construct packed_weight
        if layout.target == Target.ATEN:
            int_data = int_data.add(8)
            int_data = (int_data[::, 1::2] << 4 | int_data[::, ::2]).to(torch.uint8)

            # If group_size < k, convert scales to bfloat16
            # to call optimized kernel
            if layout.group_size < k:
                scale = scale.to(torch.bfloat16)
            packed_weight = torch.ops.aten._dyn_quant_pack_4bit_weight(
                int_data, scale, bias, layout.group_size, k, n
            )
            return cls(packed_weight, layout)

        args = [
            int_data,
            scale.reshape(-1),
            zero_point.reshape(-1) if layout.has_weight_zeros else None,
            layout.group_size,
            bias,
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
        assert bias is None, (
            "bias should be None because it is already packed with the weights (has_bias=True)"
        )

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


from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
from torchao.dtypes.utils import AQTTensorImpl, Layout


def make_packed_linear_int8_dynamic_activation_intx_weight_tensor(
    int_data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    data_dtype: torch.dtype,
    target: Union[str, Target] = "auto",
    *,
    validate_inputs: bool = True,
) -> AffineQuantizedTensor:
    """
    Constructs an AffineQuantizedTensor with PackedLinearInt8DynamicActivationIntxWeightLayout
    from plain data.
    """
    # TORCH_VERSION_AT_LEAST_2_6 is needed for torch.intx with x < 8
    assert TORCH_VERSION_AT_LEAST_2_6, (
        "Using PackedLinearInt8DynamicActivationIntxWeightLayout requires torch version > 2.6.0"
    )

    layout = PackedLinearInt8DynamicActivationIntxWeightLayout(target=target)

    bit_width = _DTYPE_TO_BIT_WIDTH[data_dtype]
    qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[data_dtype]

    n, k = int_data.shape
    n_, groups_per_k = scale.shape
    assert k % groups_per_k == 0
    group_size = k // groups_per_k

    has_weight_zeros = True
    if zero_point is None:
        has_weight_zeros = False
    else:
        zero_point_min = zero_point.min().item()
        zero_point_max = zero_point.max().item()
        if zero_point_min == 0 and zero_point_max == 0:
            has_weight_zeros = False

    has_bias = bias is not None

    layout.set_params(bit_width, group_size, has_weight_zeros, has_bias)
    assert layout.has_params_set()
    tensor_impl = PackedLinearInt8DynamicActivationIntxWeightAQTTensorImpl.from_plain(
        int_data,
        scale,
        zero_point,
        layout,
        bias,
        validate_inputs=validate_inputs,
    )

    return AffineQuantizedTensor(
        tensor_impl,
        block_size=(1, group_size),
        shape=int_data.shape,
        quant_min=qmin,
        quant_max=qmax,
        zero_point_domain=ZeroPointDomain.INT,
    )
