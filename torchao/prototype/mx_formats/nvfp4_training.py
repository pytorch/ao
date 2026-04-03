# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 training configuration and linear module.

Provides NVFP4TrainingConfig for use with quantize_() and an
NVFP4Linear module that performs NVFP4 quantized GEMMs
in both forward and backward passes.

Usage:
    from torchao.prototype.mx_formats.nvfp4_training import NVFP4TrainingConfig
    from torchao.quantization import quantize_

    quantize_(model, NVFP4TrainingConfig())
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.nvfp4_linear import nvfp4_linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler


@dataclass
class NVFP4TrainingConfig(AOBaseConfig):
    """Configuration for NVFP4 quantized training.

    When passed to quantize_(), replaces nn.Linear modules with
    NVFP4Linear, which quantizes all three GEMMs (forward
    and backward) to NVFP4.

    Args:
        kernel_preference: Backend for quantization kernels.
            TRITON: Pure-Triton RHT + stochastic rounding path.
            Default: TRITON.
    """

    kernel_preference: KernelPreference = KernelPreference.TRITON


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 quantized forward and backward GEMMs.

    Drop-in replacement for nn.Linear that quantizes activations, weights,
    and gradients to NVFP4 for all three training GEMMs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        kernel_preference: KernelPreference = KernelPreference.TRITON,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.kernel_preference = kernel_preference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nvfp4_linear(
            x, self.weight, self.bias, kernel_preference=self.kernel_preference
        )

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        kernel_preference: KernelPreference = KernelPreference.TRITON,
    ) -> "NVFP4Linear":
        new = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            kernel_preference=kernel_preference,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        # Copy weights (don't re-init)
        if mod.weight.device != torch.device("meta"):
            new.weight = mod.weight
            if mod.bias is not None:
                new.bias = mod.bias
        return new


@register_quantize_module_handler(NVFP4TrainingConfig)
def _nvfp4_training_transform(
    module: nn.Module,
    config: NVFP4TrainingConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """Handler for quantize_(): replaces nn.Linear with NVFP4Linear."""
    if isinstance(module, nn.Linear):
        return NVFP4Linear.from_linear(
            module,
            kernel_preference=config.kernel_preference,
        )
    return module
