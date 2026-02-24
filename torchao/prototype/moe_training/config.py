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
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import register_as_pytree_constant


class FP8GroupedMMRecipe(Enum):
    """FP8 recipes for grouped matrix multiplication."""

    FP8_ROWWISE = "fp8_rowwise"


class MXFP8TrainingRecipe(Enum):
    """MXFP8 recipes for grouped matrix multiplication."""

    # TODO: add floor variants
    MXFP8_RCEIL = "mxfp8_rceil"
    MXFP8_RCEIL_WGRAD_WITH_HP = "mxfp8_rceil_wgrad_with_hp"
    MXFP8_EMULATED_RCEIL = "mxfp8_emulated_rceil"


class TrainingBaseConfig(AOBaseConfig):
    """
    Base configuration for low precision training. Not intended to be used directly.

    Purpose is to support generic model conversion function for linear and grouped gemm
    low precision training.
    """

    pass


@dataclass
class FP8GroupedMMConfig(TrainingBaseConfig):
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


# register as pytree constant so we can use dynamo nonstrict trace in torchao.prototype.moe_training.ep
@register_as_pytree_constant
@dataclass
class MXFP8TrainingConfig(TrainingBaseConfig):
    """
    The MXFP8TrainingConfig defines the MXFP8 training config for nn.Linear layers
    and grouped GEMM ops.

    MXFP8TrainingConfig has a module handler registered to it which will
    find all nn.Parameters whose parent module matches the module filter function,
    and swap their data tensor with a MXFP8TrainingTensor.

    The MXFP8TrainingTensor dispatches matmul and grouped gemm ops to custom
    autograd functions which dynamically quantize inputs to MXFP8.

    For all other ops, MXFP8TrainingTensor behaves like a regular torch.Tensor.
    """

    # AUTO = Use best supported kernel for quantization ops and GEMMs (CUDA and Triton for quantizatoin, CUTLASS for MXFP8 grouped GEM
    # EMULATED = Hardware agnostic mode that can be used for debugging or development on non-SM100 machines.
    #            Uses PyTorch native quantization ops, then dequantizes and uses emulated MXFP8 grouped GEMMs implemented in PyTorch.
    #            Not recommended for performance.
    kernel_preference: KernelPreference = KernelPreference.AUTO

    # Output dtype for the MXFP8 grouped GEMMs.
    out_dtype: Optional[torch.dtype] = torch.bfloat16

    # Whether to compute the gradient of the weights in high precision (True) or use MXFP8 (False).
    wgrad_with_hp: bool = False

    # Rounding mode to use when calculating the e8m0 scale factors.
    scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL

    @classmethod
    def from_recipe(
        cls,
        recipe: MXFP8TrainingRecipe,
    ) -> "MXFP8TrainingConfig":
        """Factory method to create a MXFP8TrainingConfig from a MXFP8TrainingRecipe."""
        if recipe == MXFP8TrainingRecipe.MXFP8_RCEIL:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        elif recipe == MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=True,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        elif recipe == MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL:
            return cls(
                kernel_preference=KernelPreference.EMULATED,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        else:
            raise ValueError(f"Unsupported MXFP8 recipe: {recipe}")

    def __eq__(self, other):
        if isinstance(other, MXFP8TrainingConfig):
            return (
                self.kernel_preference == other.kernel_preference
                and self.out_dtype == other.out_dtype
                and self.wgrad_with_hp == other.wgrad_with_hp
                and self.scale_calculation_mode == other.scale_calculation_mode
            )
        return NotImplemented

    def __hash__(self):
        return hash(
            (
                self.kernel_preference,
                self.out_dtype,
                self.wgrad_with_hp,
                self.scale_calculation_mode,
            )
        )


@register_quantize_module_handler(MXFP8TrainingConfig)
def _moe_training_transform(
    module: nn.Module,
    config: TrainingBaseConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Parameter` data tensor with a MXFP8TrainingTensor.

    Args:
        module: Module to modify.
        config: TrainingBaseConfig which defines how to perform the training transform (i.e., convert linears and grouped GEMMs)
        parameter_name: If specified, only transform this specific parameter. Otherwise transform all parameters.

    Returns:
     nn.Module: The modified module with swapped parameters.
    """
    from torchao.prototype.moe_training.conversion_utils import _swap_params

    out = _swap_params(module, config=config, target_parameter_name=parameter_name)
    return out
