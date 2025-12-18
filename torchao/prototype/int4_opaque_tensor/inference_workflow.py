# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass

import torch

import torchao
from torchao.core.config import AOBaseConfig

logger = logging.getLogger(__name__)
import types

from torchao.quantization.quant_api import _linear_extra_repr
from torchao.quantization.quantize_.workflows import (
    Int4ChooseQParamsAlgorithm,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

from .int4_opaque_tensor import Int4OpaqueTensor


@dataclass
class Int4WeightOnlyOpaqueTensorConfig(AOBaseConfig):
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
            "torchao.prototype.int4_opaque_tensor.Int4WeightOnlyOpaqueTensorConfig"
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


@register_quantize_module_handler(Int4WeightOnlyOpaqueTensorConfig)
def _int4_weight_only_transform(
    module: torch.nn.Module, config: Int4WeightOnlyOpaqueTensorConfig
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
