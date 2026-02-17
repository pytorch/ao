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

from torchao.prototype.moe_training.config import GroupedMMConfig
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.utils import register_as_pytree_constant


class MXFP8GroupedMMRecipe(Enum):
    """MXFP8 recipes for grouped matrix multiplication."""

    # TODO: add floor variants
    MXFP8_RCEIL = "mxfp8_rceil"
    MXFP8_RCEIL_WGRAD_WITH_HP = "mxfp8_rceil_wgrad_with_hp"
    MXFP8_EMULATED_RCEIL = "mxfp8_emulated_rceil"


# register as pytree constant so we can use dynamo nonstrict trace in torchao.prototype.moe_training.ep
@register_as_pytree_constant
@dataclass
class MXFP8GroupedMMConfig(GroupedMMConfig):
    """
    The MXFP8GroupedMMConfig is specifically designed to be used on MoE models using
    `torch._grouped_mm` to implement expert computation in token-choice routing,
    where expert weights are implemented as 3D nn.Parameters wit `num_experts` as
    the leading dim.

    MXFP8GroupedMMConfig has a module handler registered to it which will
    find all nn.Parameters whose parent module matches the module filter function,
    and swap their data tensor with a ScaledGroupedMMTensor.

    The ScaledGroupedMMTensor is a tensor subclass which overrides the
    `torch._grouped_mm` op by dispatching to a differentiable scaled grouped mm,
    which performs dynamic quantization on scaled grouped GEMM operands in both
    the forward and backward pass, based on the quantization config (FP8/MXFP8/etc).

    For all other ops, ScaledGroupedMMTensor behaves like a regular torch.Tensor.
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
        recipe: MXFP8GroupedMMRecipe,
    ) -> "MXFP8GroupedMMConfig":
        """Factory method to create a MXFP8GroupedMMConfig from a MXFP8GroupedMMRecipe."""
        if recipe == MXFP8GroupedMMRecipe.MXFP8_RCEIL:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        elif recipe == MXFP8GroupedMMRecipe.MXFP8_RCEIL_WGRAD_WITH_HP:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=True,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        elif recipe == MXFP8GroupedMMRecipe.MXFP8_EMULATED_RCEIL:
            return cls(
                kernel_preference=KernelPreference.EMULATED,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        else:
            raise ValueError(f"Unsupported MXFP8 recipe: {recipe}")

    def __eq__(self, other):
        if isinstance(other, MXFP8GroupedMMConfig):
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


@register_quantize_module_handler(MXFP8GroupedMMConfig)
def _mxfp8_grouped_mm_transform(
    module: nn.Module,
    config: MXFP8GroupedMMConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Parameter` data tensor with a ScaledGroupedMMTensor.

    Args:
        module: Module to modify.
        config: MXFP8GroupedMMConfig which defines how to perform the MoE training transform.
        parameter_name: If specified, only transform this specific parameter. Otherwise transform all parameters.

    Returns:
     nn.Module: The modified module with swapped parameters.
    """
    from torchao.prototype.mx_formats.grouped_mm.conversion_utils import _swap_params

    out = _swap_params(module, config=config, target_parameter_name=parameter_name)
    return out
