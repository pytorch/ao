# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.parq.quant.quant_api import StretchedAffineQuantizedTensor
from torchao.prototype.quantization.dynamic_activation_lut.int8_dynamic_activation_lut_tensor import (
    Int8DynamicActivationLutTensor,
)
from torchao.quantization.granularity import Granularity, PerAxis, PerGroup
from torchao.quantization.quant_primitives import _DTYPE_TO_QVALUE_BOUNDS
from torchao.quantization.transform_module import register_quantize_module_handler


@dataclass
class StretchedAffineQuantizedTensor_to_Int8DynamicActivationLutTensorConfig(
    AOBaseConfig
):
    bit_width: int
    granularity: Granularity

    def get_filter_fn(self) -> Callable[[nn.Module, str], bool]:
        return lambda m, fqn: isinstance(m, torch.nn.Linear) and isinstance(
            m.weight, StretchedAffineQuantizedTensor
        )


@register_quantize_module_handler(
    StretchedAffineQuantizedTensor_to_Int8DynamicActivationLutTensorConfig
)
def _(
    module: nn.Module,
    config: StretchedAffineQuantizedTensor_to_Int8DynamicActivationLutTensorConfig,
) -> nn.Module:
    weight = module.weight
    bias = module.bias
    assert isinstance(weight, StretchedAffineQuantizedTensor)

    b = config.bit_width
    granularity = config.granularity
    if isinstance(granularity, PerGroup):
        group_size = granularity.group_size
    elif isinstance(granularity, PerAxis):
        assert granularity.axis == 0, (
            f"axis must be 0 with PerAxis, but got {granularity.axis}"
        )
        group_size = weight.shape[-1]
    else:
        raise ValueError(f"granularity must be PerGroup or PerAxis, got {granularity}")

    int_data, scale, zero_point = weight.tensor_impl.get_plain()
    q_min, q_max = _DTYPE_TO_QVALUE_BOUNDS[getattr(torch, f"int{b}")]

    # Construct LUT as 2 * ([q_min, q_max] - 0.5)
    assert torch.all(zero_point == -0.5)
    lut = torch.arange(q_min, q_max + 1)
    lut = 2 * lut + 1

    # Construct idx values
    qval_idx = int_data - q_min

    # Construct scale
    scale = scale.reshape(-1).to(torch.float32)
    scale = 0.5 * scale  # since we multiply LUT values by 2

    weight_tensor = Int8DynamicActivationLutTensor.from_plain(
        qval_idx,
        lut,
        scale,
        group_size,
        bias.to(torch.float32) if bias is not None else None,
    )
    module.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
    module.bias = None
    return module
