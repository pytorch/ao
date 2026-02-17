# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import register_quantize_module_handler


class GroupedMMConfig(AOBaseConfig):
    """Base configuration for grouped matrix multiplication. Not intended to be used directly."""

    pass


class FP8GroupedMMRecipe(Enum):
    """FP8 recipes for grouped matrix multiplication."""

    FP8_ROWWISE = "fp8_rowwise"


@dataclass
class FP8GroupedMMConfig(GroupedMMConfig):
    """
    Configuration for FP8 grouped matrix multiplication.
    """

    # Output dtype for the FP8 grouped GEMMs.
    out_dtype: Optional[torch.dtype] = torch.bfloat16

    @classmethod
    def from_recipe(
        cls,
        recipe: FP8GroupedMMRecipe,
    ) -> "FP8GroupedMMConfig":
        """Factory method to create a FP8GroupedMMConfig from a FP8GroupedMMRecipe."""
        if recipe == FP8GroupedMMRecipe.FP8_ROWWISE:
            return cls()
        else:
            raise ValueError(f"Unsupported FP8 recipe: {recipe}")


@register_quantize_module_handler(FP8GroupedMMConfig)
def _fp8_grouped_mm_transform(
    module: nn.Module,
    config: FP8GroupedMMConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Parameter` data tensor with a ScaledGroupedMMTensor.

    Args:
        module: Module to modify.
        config: FP8GroupedMMConfig which defines how to perform the FP8 MoE training transform.
        parameter_name: If specified, only transform this specific parameter. Otherwise transform all parameters.

    Returns:
     nn.Module: The modified module with swapped parameters.
    """
    from torchao.prototype.mx_formats.grouped_mm.conversion_utils import _swap_params

    out = _swap_params(module, config=config, target_parameter_name=parameter_name)
    return out
