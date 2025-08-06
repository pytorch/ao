# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from torchao.dtypes import AffineQuantizedTensor, Layout, QDQLayout
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig
from torchao.quantization.quant_primitives import (
    _SUB_BYTE_UINT_BOUNDS,
    MappingType,
    ZeroPointDomain,
    _get_reduction_params,
    dequantize_affine,
)
from torchao.quantization.transform_module import register_quantize_module_handler


def choose_qparams_stretched_affine(
    input_float: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int, ...],
    target_dtype: torch.dtype,
    b: int,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale_dtype is None:
        scale_dtype = input_float.dtype
    if eps is None:
        eps = torch.finfo(input_float.dtype).eps
    if zero_point_dtype is None:
        zero_point_dtype = input_float.dtype

    assert len(block_size) == input_float.dim(), f"Got {input.dim()=}, {block_size=}"
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input_float.size()
    )
    input_float = input_float.view(shape_for_reduction)

    q_abs = input_float.abs()
    max_val = torch.minimum(
        b * q_abs.mean(dim=reduction_dims, keepdim=True),
        torch.amax(q_abs, dim=reduction_dims, keepdim=True),
    ).clamp_(min=eps)

    scale = max_val / quant_max
    scale = scale.to(dtype=scale_dtype, device=input_float.device)
    zero_point = torch.full_like(scale, -0.5, dtype=zero_point_dtype)
    return scale, zero_point


def quantize_stretched_affine(
    input_float: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_dtype: torch.dtype,
    quant_min: Optional[int] = None,
    quant_max: Optional[int] = None,
) -> torch.Tensor:
    if target_dtype in _SUB_BYTE_UINT_BOUNDS:
        target_dtype = torch.uint8
    assert input_float.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"Unsupported input_float dtype: {input_float.dtype}"
    )
    assert len(block_size) == input_float.dim(), (
        f"Got {input_float.dim()=}, {block_size=}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input_float.size()
    )
    original_shape = input_float.shape
    input_float = input_float.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    if zero_point is not None and zero_point.numel() > 0:
        zero_point = zero_point.view(shape_after_reduction)
    else:
        zero_point = None

    max_val = scale.mul(quant_max)
    input_float = input_float.clamp(min=-max_val, max=max_val)
    with torch.no_grad():
        # difference from quantize_affine: add zero_point before rounding
        quant = torch.round(input_float / scale + zero_point)
    quant = quant.to(dtype=target_dtype).view(original_shape)
    return quant


class StretchedAffineQuantizedTensor(AffineQuantizedTensor):
    @classmethod
    def from_hp_to_intx(
        cls,
        input_float: torch.Tensor,
        mapping_type: MappingType,
        block_size: Tuple[int, ...],
        target_dtype: torch.dtype,
        b: int,
        quant_min: Optional[float] = None,
        quant_max: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
        _layout: Layout = QDQLayout(),  # noqa: B008
    ):
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)

        scale, zero_point = choose_qparams_stretched_affine(
            input_float,
            mapping_type,
            block_size,
            target_dtype,
            b,
            quant_min=quant_min,
            quant_max=quant_max,
        )
        data = quantize_stretched_affine(
            input_float,
            block_size,
            scale,
            zero_point,
            target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
        )
        data, scale, zero_point = _layout.post_process(
            data, scale, zero_point, block_size
        )
        tensor_impl_ctr = cls.get_tensor_impl_constructor(type(_layout))
        tensor_impl = tensor_impl_ctr(data, scale, zero_point, _layout)
        return cls(
            tensor_impl,
            block_size,
            original_shape,
            quant_min,
            quant_max,
            zero_point_domain,
            dtype=input_float.dtype,
        )

    def dequantize(self, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = self.dtype

        if not isinstance(self._layout, QDQLayout):
            raise NotImplementedError(
                f"StretchedAffineQuantizedTensor only supports QDQLayout but got {self._layout}"
            )

        data, scale, zero_point = self.tensor_impl.get_plain()
        dq = dequantize_affine(
            data,
            self.block_size,
            scale,
            zero_point,
            data.dtype,
            self.quant_min,
            self.quant_max,
            output_dtype=output_dtype,
        )
        return dq


to_stretched_affine_quantized_intx = StretchedAffineQuantizedTensor.from_hp_to_intx


@dataclass
class StretchedIntxWeightOnlyConfig(IntxWeightOnlyConfig):
    b: Optional[int] = None
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None


@register_quantize_module_handler(StretchedIntxWeightOnlyConfig)
def _stretched_intx_weight_only_transform(
    module: nn.Module, config: StretchedIntxWeightOnlyConfig
) -> nn.Module:
    weight = module.weight
    granularity = config.granularity
    mapping_type = MappingType.ASYMMETRIC

    assert weight.dim() == 2, (
        f"StretchedIntxWeightOnlyConfig only works for 2-d Tensor, got: {weight.dim()}"
    )
    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerAxis):
        assert granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {granularity.axis}"
        )
        group_size = weight.shape[-1]
    else:
        raise ValueError(f"granularity must be PerGroup or PerAxis, got {granularity}")

    weight = to_stretched_affine_quantized_intx(
        input_float=weight,
        mapping_type=mapping_type,
        block_size=(1, group_size),
        target_dtype=torch.int8,
        b=config.b,
        quant_min=config.quant_min,
        quant_max=config.quant_max,
        scale_dtype=config.scale_dtype,
        _layout=config.layout,
    )
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module
