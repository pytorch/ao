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
from torchao.utils import is_MI300, register_as_pytree_constant


class Float8TrainingRecipe(Enum):
    """FP8 recipes for grouped matrix multiplication."""

    FP8_ROWWISE = "fp8_rowwise"


class MXFP8TrainingRecipe(Enum):
    """MXFP8 recipes for grouped matrix multiplication."""

    # TODO: add floor variants
    MXFP8_RCEIL = "mxfp8_rceil"
    MXFP8_RCEIL_WGRAD_WITH_HP = "mxfp8_rceil_wgrad_with_hp"
    MXFP8_EMULATED_RCEIL = "mxfp8_emulated_rceil"


class TrainingOpBaseConfig(AOBaseConfig):
    """
    Base configuration for low precision training. Not intended to be used directly.

    Purpose is to support generic model conversion function for linear and grouped gemm
    low precision training.
    """

    pass


@dataclass
class Float8TrainingOpConfig(TrainingOpBaseConfig):
    """
    Configuration for FP8 grouped matrix multiplication.
    """

    # Float8 dtype for the FP8 grouped GEMMs.
    float8_dtype: torch.dtype = (
        torch.float8_e4m3fnuz if is_MI300() else torch.float8_e4m3fn
    )
    # Output dtype for the FP8 grouped GEMMs.
    out_dtype: Optional[torch.dtype] = torch.bfloat16

    # TODO: support pad_token_groups_for_grouped_mm field like MXFP8TrainingOpConfig

    @classmethod
    def from_recipe(
        cls,
        recipe: Float8TrainingRecipe,
    ) -> "Float8TrainingOpConfig":
        """Factory method to create a Float8TrainingOpConfig from a Float8TrainingRecipe."""
        if recipe == Float8TrainingRecipe.FP8_ROWWISE:
            return cls()
        else:
            raise ValueError(f"Unsupported FP8 recipe: {recipe}")


# register as pytree constant so we can use dynamo nonstrict trace in torchao.prototype.moe_training.ep
@register_as_pytree_constant
@dataclass
class MXFP8TrainingOpConfig(TrainingOpBaseConfig):
    """
    The MXFP8TrainingOpConfig defines the MXFP8 training config for nn.Linear layers
    and grouped GEMM ops.

    MXFP8TrainingOpConfig has a module handler registered to it which will
    find all nn.Parameters whose parent module matches the module filter function,
    and swap their data tensor with a MXFP8TrainingWeightWrapperTensor.

    The MXFP8TrainingWeightWrapperTensor dispatches matmul and grouped gemm ops to custom
    autograd functions which dynamically quantize inputs to MXFP8.

    For all other ops, MXFP8TrainingWeightWrapperTensor behaves like a regular torch.Tensor.
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

    # Whether to pad the token group sizes to multiples of 32 (MXFP8 scaling block size).
    pad_token_groups_for_grouped_mm: bool = False

    @classmethod
    def from_recipe(
        cls,
        recipe: MXFP8TrainingRecipe,
    ) -> "MXFP8TrainingOpConfig":
        """Factory method to create a MXFP8TrainingOpConfig from a MXFP8TrainingRecipe."""
        if recipe == MXFP8TrainingRecipe.MXFP8_RCEIL:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
                pad_token_groups_for_grouped_mm=True,
            )
        elif recipe == MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=True,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
                pad_token_groups_for_grouped_mm=True,
            )
        elif recipe == MXFP8TrainingRecipe.MXFP8_EMULATED_RCEIL:
            return cls(
                kernel_preference=KernelPreference.EMULATED,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
                pad_token_groups_for_grouped_mm=True,
            )
        else:
            raise ValueError(f"Unsupported MXFP8 recipe: {recipe}")

    def __eq__(self, other):
        if isinstance(other, MXFP8TrainingOpConfig):
            return (
                self.kernel_preference == other.kernel_preference
                and self.out_dtype == other.out_dtype
                and self.wgrad_with_hp == other.wgrad_with_hp
                and self.scale_calculation_mode == other.scale_calculation_mode
                and self.pad_token_groups_for_grouped_mm
                == other.pad_token_groups_for_grouped_mm
            )
        return NotImplemented

    def __hash__(self):
        return hash(
            (
                self.kernel_preference,
                self.out_dtype,
                self.wgrad_with_hp,
                self.scale_calculation_mode,
                self.pad_token_groups_for_grouped_mm,
            )
        )


@register_quantize_module_handler(Float8TrainingOpConfig)
@register_quantize_module_handler(MXFP8TrainingOpConfig)
def _moe_training_transform(
    module: nn.Module,
    config: TrainingOpBaseConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Parameter` data tensor with the appropriate training tensor
    subclass (Float8TrainingWeightWrapperTensor or MXFP8TrainingWeightWrapperTensor) based on the config type.

    Args:
        module: Module to modify.
        config: TrainingOpBaseConfig which defines how to perform the training transform (i.e., convert linears and grouped GEMMs)
        parameter_name: If specified, only transform this specific parameter. Otherwise transform all parameters.

    Returns:
     nn.Module: The modified module with swapped parameters.
    """
    from torchao.prototype.moe_training.conversion_utils import _swap_params

    out = _swap_params(module, config=config, target_parameter_name=parameter_name)
    return out
