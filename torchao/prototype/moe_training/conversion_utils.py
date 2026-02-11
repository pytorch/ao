# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

logger: logging.Logger = logging.getLogger(__name__)


class FP8GroupedMMRecipe(Enum):
    """FP8 recipes for grouped matrix multiplication."""

    ROWWISE = "fp8_rowwise"


class MXFP8GroupedMMRecipe(Enum):
    """MXFP8 recipes for grouped matrix multiplication."""

    # TODO: add floor variants
    RCEIL = "rceil"
    RCEIL_WGRAD_WITH_HP = "rceil_wgrad_with_hp"
    EMULATED_RCEIL = "emulated"


class GroupedMMConfig(AOBaseConfig):
    """Base configuration for grouped matrix multiplication. Not intended to be used directly."""

    pass


@dataclass
class FP8GroupedMMConfig(GroupedMMConfig):
    """
    Configuration for FP8 grouped matrix multiplication.

    Currently, FP8 grouped matrix multiplication is only supported with a single recipe (FP8 rowwise),
    and the configuration options are hardcoded in the implementation.

    When more recipes and configuration options are added, this class will be expanded to include those options.
    """

    @classmethod
    def from_recipe(
        cls,
        recipe: FP8GroupedMMRecipe,
    ) -> "FP8GroupedMMConfig":
        """Factory method to create a FP8GroupedMMConfig from a FP8GroupedMMRecipe."""
        if recipe == FP8GroupedMMRecipe.ROWWISE:
            return cls()
        else:
            raise ValueError(f"Unsupported FP8 recipe: {recipe}")


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
    which performs dynamic float8 rowwise quantization on scaled grouped GEMM
    operands in both the forward and backward pass.

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
        if recipe == MXFP8GroupedMMRecipe.RCEIL:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=False,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        elif recipe == MXFP8GroupedMMRecipe.RCEIL_WGRAD_WITH_HP:
            return cls(
                kernel_preference=KernelPreference.AUTO,
                out_dtype=torch.bfloat16,
                wgrad_with_hp=True,
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
            )
        elif recipe == MXFP8GroupedMMRecipe.EMULATED_RCEIL:
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


# Need for dynamo non-strict trace mode
torch.utils._pytree.register_constant(MXFP8GroupedMMConfig)


@register_quantize_module_handler(MXFP8GroupedMMConfig)
def _moe_training_transform(
    module: nn.Module,
    config: GroupedMMConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Swaps `torch.nn.Parameter` data tensor with a ScaledGroupedMMTensor.

    Args:
        module: Module to modify.
        config: GroupedMMConfig which defines how to perform the MoE training transform.
        parameter_name: If specified, only transform this specific parameter. Otherwise transform all parameters.

    Returns:
     nn.Module: The modified module with swapped parameters.
    """

    out = _swap_params(module, config=config, target_parameter_name=parameter_name)
    return out


def _swap_params(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Optional[GroupedMMConfig] = None,
    target_parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Recurses through the nn.Module, recursively swapping the data tensor of
    each nn.Parameter with a ScaledGroupedMMTensor. Only applies if the module
    passed the module_filter_fn, if specified.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Parameter` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor

    if isinstance(module, nn.Parameter) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Parameter with children: {module}"
            )
        if not isinstance(module.data, ScaledGroupedMMTensor):
            new_data = ScaledGroupedMMTensor(module.data, config)
            return nn.Parameter(new_data, requires_grad=module.requires_grad)
        return module

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if module_filter_fn is None or module_filter_fn(module, cur_fqn):
            for param_name, param in module.named_parameters(recurse=False):
                if (
                    target_parameter_name is not None
                    and param_name != target_parameter_name
                ):
                    continue
                if not isinstance(param.data, ScaledGroupedMMTensor):
                    new_param = nn.Parameter(
                        ScaledGroupedMMTensor(param.data, config),
                        requires_grad=param.requires_grad,
                    )
                    setattr(module, param_name, new_param)
                    logger.info(
                        f"Swapped {cur_fqn}.{param_name} to ScaledGroupedMMTensor"
                    )

    post_order_traversal(root_module)
    return root_module
