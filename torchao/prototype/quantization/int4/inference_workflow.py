# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import types
from dataclasses import dataclass

import torch

import torchao
from torchao.core.config import AOBaseConfig

logger = logging.getLogger(__name__)

from torchao.quantization.quant_api import _linear_extra_repr
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows import (
    Int4ChooseQParamsAlgorithm,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

from .int4_opaque_tensor import Int4OpaqueTensor


@dataclass
class PrototypeInt4WeightOnlyConfig(AOBaseConfig):
    """
    Configuration for int4 weight only quantization, only groupwise quantization is supported right now.

    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller size is more fine grained, choices are [256, 128, 64, 32]
        `int4_choose_qparams_algorithm`: variants of choose qparams algorithm to use for int4, currently support TINYGEMM ("tinygemm") and HQQ ("hqq")
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values
    """

    group_size: int = 128
    int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = (
        Int4ChooseQParamsAlgorithm.TINYGEMM
    )
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.prototype.quantization.int4.PrototypeInt4WeightOnlyConfig"
        )


def _int4_weight_only_opaque_tensor_quantize(weight, config):
    group_size = config.group_size
    int4_choose_qparams_algorithm = config.int4_choose_qparams_algorithm

    if weight.shape[-1] % group_size != 0:
        logger.info(
            f"Skipping quantizing weight with int4 weight only quantization because the shape of weight {weight.shape} is not compatible with group_size {group_size}"
        )
        return weight

    block_size = tuple([1 for _ in range(weight.ndim - 1)] + [group_size])

    block_size = list(block_size)

    new_weight = Int4OpaqueTensor.from_hp(
        weight,
        block_size,
        int4_choose_qparams_algorithm=int4_choose_qparams_algorithm,
    )
    return new_weight


@register_quantize_module_handler(PrototypeInt4WeightOnlyConfig)
def _int4_weight_only_transform(
    module: torch.nn.Module, config: PrototypeInt4WeightOnlyConfig
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying int4 weight only quant requires module to have weight attribute"
        + " but {module} does not have one"
    )
    new_weight = _int4_weight_only_opaque_tensor_quantize(module.weight, config)
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class Int8DynamicActivationInt4WeightConfig(AOBaseConfig):
    """
    Configuration for int8 dynamic activation + int4 weight quantization on CPU,
    using Int4OpaqueTensor (tensor subclassing) with the da8w4_linear_cpu backend.

    Weights are quantized per-group (asymmetric int4) and prepacked at quantization time.
    Activations are quantized dynamically per-token at runtime.

    Args:
        `group_size`: quantization group size for weights; K must be divisible by group_size;
            choices are [128, 64, 32]; otherwise weight will not be quantized
        `act_mapping_type`: activation quantization type:
            - MappingType.ASYMMETRIC (default): uint8 activation quantization
            - MappingType.SYMMETRIC: int8 activation quantization (requires PyTorch >= 2.8)
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values
    Example:

    .. literalinclude:: ../../examples/prototype/int8_dynamic_activation_int4_weight.py
       :language: python
    """

    group_size: int = 32
    act_mapping_type: MappingType = MappingType.ASYMMETRIC
    set_inductor_config: bool = True

    def __post_init__(self):
        torch._C._log_api_usage_once(
            "torchao.prototype.quantization.int4.Int8DynamicActivationInt4WeightConfig"
        )


@register_quantize_module_handler(Int8DynamicActivationInt4WeightConfig)
def _int8_dynamic_act_int4_weight_transform(
    module: torch.nn.Module, config: Int8DynamicActivationInt4WeightConfig
) -> torch.nn.Module:
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    assert hasattr(module, "weight"), (
        "applying DA8W4 quant requires module to have weight attribute"
        + f" but {module} does not have one"
    )
    assert "CPU" in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"), (
        "DA8W4 on CPU requires the da8w4_linear_cpu kernel to be built and available"
    )
    weight = module.weight
    if weight.shape[-1] % config.group_size != 0 or config.group_size not in [
        128,
        64,
        32,
    ]:
        logger.info(
            f"Skipping DA8W4 quantization: weight shape {weight.shape} is not compatible "
            f"with group_size {config.group_size} (must be divisible and one of [128, 64, 32])"
        )
        return module
    if weight.shape[0] % 32 != 0 or weight.shape[-1] % 2 != 0:
        logger.info(
            f"Skipping DA8W4 quantization: weight shape {weight.shape} requires "
            "N divisible by 32 and K divisible by 2"
        )
        return module

    new_weight = Int4OpaqueTensor.from_hp_da8w4(
        weight,
        group_size=config.group_size,
        act_mapping_type=config.act_mapping_type,
    )
    module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module
