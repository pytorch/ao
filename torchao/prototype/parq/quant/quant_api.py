# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from torchao.dtypes import AffineQuantizedTensor, Layout, PlainLayout, QDQLayout
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.quant_api import IntxWeightOnlyConfig
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine_with_min_max,
    dequantize_affine,
    quantize_affine,
)
from torchao.quantization.transform_module import register_quantize_module_handler

from .uniform import get_q_max


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
        eps: Optional[float] = None,
        scale_dtype: Optional[torch.dtype] = None,
        zero_point_dtype: Optional[torch.dtype] = None,
        preserve_zero: bool = False,
        zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT,
        _layout: Layout = PlainLayout(),  # noqa: B008
        scale_method: str = "mean",
    ):
        original_shape = input_float.shape
        input_float = _layout.pre_process(input_float)

        dim = None
        qmax_shape = []
        for d, size in enumerate(block_size):
            if size > 1:
                dim = d
                qmax_shape.append(original_shape[d] // size)
            else:
                qmax_shape.append(original_shape[d])
        assert dim is not None, (
            "block_size must have at least one dimension greater than 1"
        )
        reduction_shape = [-1 if i != dim else b for i, b in enumerate(block_size)]
        input_float = input_float.view(reduction_shape)
        q_max = get_q_max(input_float, b, dim=dim, scale_method=scale_method)
        q_max = q_max.clamp(min=torch.finfo(input_float.dtype).tiny)
        q_max = q_max.view(qmax_shape)
        input_float = input_float.view(original_shape)

        scale, zero_point = choose_qparams_affine_with_min_max(
            -q_max,
            q_max,
            mapping_type,
            block_size,
            target_dtype,
            eps=eps,
            quant_min=quant_min,
            quant_max=quant_max,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
        )
        data = quantize_affine(
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
    quant_min: Optional[float] = None
    quant_max: Optional[float] = None
    mapping_type: MappingType = MappingType.ASYMMETRIC
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.FLOAT


@register_quantize_module_handler(StretchedIntxWeightOnlyConfig)
def _stretched_intx_weight_only_transform(
    module: nn.Module, config: StretchedIntxWeightOnlyConfig
) -> nn.Module:
    weight = module.weight
    granularity = config.granularity
    mapping_type = config.mapping_type

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
        zero_point_dtype=torch.int8,
        preserve_zero=(mapping_type == MappingType.SYMMETRIC),
        zero_point_domain=config.zero_point_domain,
        _layout=config.layout,
    )
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module
