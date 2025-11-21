# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch

import torchao
from torchao.core.config import AOBaseConfig

if TYPE_CHECKING:
    from torchao.quantization.granularity import PerGroup, PerRow, PerTensor


# Define FP8Granularity type alias to break circular import dependencies
FP8Granularity = Union["PerTensor", "PerRow", "PerGroup"]

import types
from functools import partial

from torchao.quantization.quant_api import _module_extra_repr
from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.quantization.utils import get_block_size

from .float8_opaque_tensor import Float8OpaqueTensor


@dataclass
class Float8DynamicActivationFloat8WeightOpaqueTensorConfig(AOBaseConfig):
    """
    Configuration for applying float8 dynamic symmetric quantization to both activations and weights of linear layers.

    Args:
        activation_dtype (torch.dtype): The target data type for activation quantization. Only torch.float8_e4m3fn supported.
        weight_dtype (torch.dtype): The target data type for weight quantization. Only torch.float8_e4m3fn supported.
        granularity (Optional[Union[FP8Granularity, List[FP8Granularity]]]):
            The granularity for quantization. Can be either a single granularity (applied to both
            activations and weights) or a tuple of two granularities (one for activations, one for weights).
            If None, defaults to PerTensor for both. Currently both quantizations need to be the same type. And
            only PerTensor/PerRow/PerGroup are supported.

    """

    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    granularity: Optional[Union[FP8Granularity, List[FP8Granularity]]] = None
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.quantization.Float8DynamicActivationFloat8WeightConfig"
        )
        activation_granularity, weight_granularity = (
            Float8OpaqueTensor._normalize_and_check_granularity(self.granularity)
        )
        self.granularity = [activation_granularity, weight_granularity]


def _float8_dynamic_activation_float8_weight_opaque_tensor_quantize(weight, config):
    activation_dtype = config.activation_dtype
    granularity = config.granularity

    activation_granularity, weight_granularity = granularity

    act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
        activation_dtype,
        activation_granularity,
    )

    block_size = get_block_size(weight.shape, weight_granularity)
    quantized_weight = Float8OpaqueTensor.from_hp(
        weight,
        block_size=block_size,
        act_quant_kwargs=act_quant_kwargs,
    )

    return quantized_weight


@register_quantize_module_handler(Float8DynamicActivationFloat8WeightOpaqueTensorConfig)
def _float8_dynamic_activation_float8_weight_opaque_tensor_transform(
    module: torch.nn.Module,
    config: Float8DynamicActivationFloat8WeightOpaqueTensorConfig,
    *,
    parameter_name: str = "weight",
):
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, parameter_name), (
        f"applying float8 dynamic activation quant requires module to have parameter {parameter_name} attribute"
        + f" but {module} does not have one"
    )
    quantized_tensor = _float8_dynamic_activation_float8_weight_opaque_tensor_quantize(
        getattr(module, parameter_name), config
    )
    setattr(
        module,
        parameter_name,
        torch.nn.Parameter(quantized_tensor, requires_grad=False),
    )
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr,
            parameter_name=parameter_name,
        ),
        module,
    )
    return module
