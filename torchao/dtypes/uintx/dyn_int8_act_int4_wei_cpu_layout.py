# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils._python_dispatch import (
    return_and_correct_aliasing,
)

from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
    register_layout,
)
from torchao.dtypes.utils import Layout, PlainLayout, is_device
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_7,
    TORCH_VERSION_AT_LEAST_2_8,
)

from .int4_cpu_layout import (
    Int4CPUAQTTensorImpl,
    _is_float,
)

aten = torch.ops.aten


@dataclass(frozen=True)
class Int8DynamicActInt4WeightCPULayout(Layout):
    """Layout class for da8w4 CPU layout for affine quantized tensor"""

    pass


@register_layout(Int8DynamicActInt4WeightCPULayout)
class DA8W4CPUAQTTensorImpl(Int4CPUAQTTensorImpl):
    """TensorImpl for da8w4 CPU layout for affine quantized tensor
    It stores the original tensor of dimension [n][k] (int32 dtype) as packed weight of 2-d tensor of
    dimension: [n][k / 2] (uint8 dtype)
    It is similar to Int4CPUAQTTensorImpl but with a different memory layout of weight data
    fields:
      packed_weight (torch.Tensor): the 2-d packed tensor in a Int4 CPU layout
      scales (torch.Tensor): the scales Tensor used to map between floating point tensor to quantized tensor
      qzeros (torch.Tensor): the zero_point Tensor used to map between floating point tensor to quantized tensor
    """

    def __new__(
        cls,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        compensation: torch.Tensor,
        transposed: bool,
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
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        compensation: torch.Tensor,
        transposed: bool,
        _layout: Layout,
    ):
        self.packed_weight = packed_weight
        self.scales = scales
        self.qzeros = qzeros
        self.compensation = compensation
        self.transposed = transposed
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_weight", "scales", "qzeros", "compensation"], [
            self.transposed,
            self._layout,
        ]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_weight, scales, qzeros, compensation = (
            tensor_data_dict["packed_weight"],
            tensor_data_dict["scales"],
            tensor_data_dict["qzeros"],
            tensor_data_dict["compensation"],
        )
        (
            transposed,
            _layout,
        ) = tensor_attributes
        return cls(packed_weight, scales, qzeros, compensation, transposed, _layout)

    @classmethod
    def from_plain(
        cls,
        int_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        _layout: Layout,
    ):
        assert isinstance(_layout, Int8DynamicActInt4WeightCPULayout)
        assert int_data.dtype == torch.uint8, "DA8W4 CPU: expects uint8 weight"
        assert int_data.shape[1] % 2 == 0, "DA8W4 CPU: expects even number of columns"
        if scale.dim() == 1:
            scale.unsqueeze_(-1)
        scale = scale.to(torch.float)
        if zero_point.dim() == 1:
            zero_point.unsqueeze_(-1)

        weight_int4, scales, qzeros, compensation = (
            torch.ops.torchao.da8w4_linear_prepack_cpu(int_data, scale, zero_point)
        )
        return cls(weight_int4, scales, qzeros, compensation, False, _layout)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_weight),
            fn(self.scales),
            fn(self.qzeros),
            fn(self.compensation),
            self.transposed,
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs
        if func is aten.t.default:
            """we don't need to repack the weight and just rely on external
            shape being changed and record the status of transpose/no-transpose
            """
            transposed = DA8W4CPUAQTTensorImpl(
                args[0].packed_weight,
                args[0].scales,
                args[0].qzeros,
                args[0].compensation,
                not args[0].transposed,
                args[0]._layout,
            )
            return return_and_correct_aliasing(func, args, kwargs, transposed)
        else:
            return super().__torch_dispatch__(func, types, args, kwargs)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def block_size(self):
        assert len(self.packed_weight.shape) == 2
        weight_shape = self.packed_weight.shape
        N = weight_shape[0]
        K = weight_shape[1] * 2
        groups = self.scales.numel() // N
        group_size = K // groups
        return (1, group_size)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unpack weight by linear(eye(K), packed_weight).t()
        packed_w_shape = self.packed_weight.shape
        if len(packed_w_shape) == 4:
            K = packed_w_shape[1] * packed_w_shape[2]
        else:
            K = packed_w_shape[1]
        x = torch.eye(K).to(torch.uint8)
        x_scale = torch.ones(K).float()
        x_qzero = torch.zeros(K).to(torch.int32)
        w_scale = torch.ones_like(self.scales).float()
        w_qzero = torch.zeros_like(self.qzeros).to(torch.int8)
        plain_weight = torch.ops.torchao.da8w4_linear_cpu.default(
            x,
            x_scale,
            x_qzero,
            self.packed_weight,
            w_scale,
            w_qzero,
            self.compensation,
            None,  # bias
            torch.float,  # out_dtype
        )
        plain_weight = plain_weight.t().contiguous()
        plain_weight = plain_weight.to(torch.int8)

        if self.scales.dim() == 2:
            assert self.qzeros.dim() == 2
            plain_scales = self.scales
            plain_qzeros = self.qzeros
        else:
            assert self.scales.dim() == 3 and self.qzeros.dim() == 3
            packed_shape = self.scales.shape  # [Nc, G, block_n]
            plain_scales = (
                self.scales.permute([0, 2, 1]).contiguous().view([-1, packed_shape[1]])
            )
            plain_qzeros = (
                self.qzeros.permute([0, 2, 1]).contiguous().view([-1, packed_shape[1]])
            )

        return plain_weight, plain_scales, plain_qzeros


def _aqt_is_uint8(aqt):
    """Check if an AffineQuantizedTensor is uint8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.uint8
        and aqt.quant_min == 0
        and aqt.quant_max == 255
    )


def _aqt_is_int8(aqt):
    """Check if an AffineQuantizedTensor is uint8 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.int8
        and aqt.quant_min == -127
        and aqt.quant_max == 127
    )


def _aqt_is_uint4(aqt):
    """Check if an AffineQuantizedTensor is uint4 quantized Tensor"""
    return (
        aqt.tensor_impl.dtype == torch.uint8
        and aqt.quant_min == 0
        and aqt.quant_max == 15
    )


def _linear_int8_act_int4_weight_cpu_check(input_tensor, weight_tensor, bias):
    return (
        TORCH_VERSION_AT_LEAST_2_7
        and is_device(input_tensor.device.type, "cpu")
        and is_device(weight_tensor.device.type, "cpu")
        and (bias is None or is_device(bias.device.type, "cpu"))
        and isinstance(input_tensor, AffineQuantizedTensor)
        and (_aqt_is_uint8(input_tensor) or _aqt_is_int8(input_tensor))
        and _is_float(input_tensor.dtype)
        and isinstance(input_tensor._layout, PlainLayout)
        and isinstance(weight_tensor, AffineQuantizedTensor)
        and _aqt_is_uint4(weight_tensor)
        and _is_float(weight_tensor.dtype)
        and isinstance(weight_tensor._layout, Int8DynamicActInt4WeightCPULayout)
    )


def _linear_int8_act_int4_weight_cpu_impl(input_tensor, weight_tensor, bias):
    assert TORCH_VERSION_AT_LEAST_2_7, (
        f"Requires PyTorch version at least 2.7, but got: {torch.__version__}"
    )
    if _aqt_is_int8(input_tensor):
        assert TORCH_VERSION_AT_LEAST_2_8, (
            f"Requires PyTorch version at least 2.8, but got: {torch.__version__}"
        )
    assert is_device(input_tensor.device.type, "cpu"), (
        f"For CPU device only but got: {input_tensor.device}"
    )
    assert weight_tensor.block_size[0] == 1, (
        f"Requires groupwise quantization, got block_size: {weight_tensor.block_size}"
    )
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"need input_tensor shape: {input_tensor.shape} final"
        f"dim to match weight_tensor shape: {weight_tensor.shape} second dim "
    )

    act_mat = input_tensor
    act = act_mat.tensor_impl.int_data
    act_scales = act_mat.tensor_impl.scale
    act_qzeros = act_mat.tensor_impl.zero_point

    packed_weight = weight_tensor.tensor_impl.packed_weight
    wei_scales = weight_tensor.tensor_impl.scales
    wei_qzeros = weight_tensor.tensor_impl.qzeros
    compensation = weight_tensor.tensor_impl.compensation

    orig_act_size = act_mat.size()
    orig_dtype = act_mat.dtype

    # reshape to 2D
    act = act.reshape(-1, act.shape[-1])

    y = torch.ops.torchao.da8w4_linear_cpu.default(
        act.contiguous(),
        act_scales,
        act_qzeros,
        packed_weight,
        wei_scales,
        wei_qzeros,
        compensation,
        bias.float() if bias is not None else bias,  # requires bias to be float
        orig_dtype,  # out_dtype
    )

    # remove out_feature padding
    orig_out_features = weight_tensor.shape[-2]
    y = y[:, :orig_out_features]
    y = y.reshape(*orig_act_size[:-1], orig_out_features)

    return y.to(orig_dtype)


# Register the concat linear fusion pass
from ...prototype.inductor.fx_passes import register_da8w4_concat_linear_cpu_pass

register_da8w4_concat_linear_cpu_pass()
