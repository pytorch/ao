# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


@dataclass
class IntNWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying int N-bit weight only quantization to a linear layer.
    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller size is more fine grained, choices are [512, 256, 128, 64, 32]
        `n`: number of bits to quantize to, choices are [8, 6, 5, 4, 3, 2]
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    Usage:
        from torchao.quantization import quantize_
        quantize_(model, intN_weight_only(n=your_bit_choice, group_size=group_size), optional_filter_func_for_desired_layers_to_quantize)
    """

    group_size: int = 32
    n: int = 8
    symmetric: bool = False
    set_inductor_config: bool = True


# for bc
intN_weight_only = IntNWeightOnlyConfig


@register_quantize_module_handler(IntNWeightOnlyConfig)
def _intN_weight_only_transform(
    module: torch.nn.Module,
    config: IntNWeightOnlyConfig,
) -> torch.nn.Module:
    raise AssertionError(
        "This feature is currently broken, see https://github.com/pytorch/ao/pull/4151"
        " and https://github.com/pytorch/ao/pull/4245 for more details"
    )
