# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.constants import DTYPE_TO_SHORT_STR
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.utils import register_as_pytree_constant


# Pre-made recipes for common configurations
class MXLinearRecipeName(Enum):
    MXFP4_EMULATED = "mxfp4_emulated"
    MXFP4_CUTLASS = "mxfp4_cutlass"


class MXFP8Dim0CastKernelChoice(Enum):
    """
    Defines which kernel to use for mxfp8 casting along dim0.
    """

    TRITON = "triton"
    TORCH = "torch"


class MXFP8Dim1CastKernelChoice(Enum):
    """
    Defines which kernel to use for mxfp8 casting along dim1.
    """

    TRITON = "triton"
    CUDA = "cuda"
    TORCH = "torch"


# register as pytree constant so we can use dynamo nonstrict trace in torchao.prototype.moe_training.ep
@register_as_pytree_constant
class ScaleCalculationMode(Enum):
    """
    Enum representing the different methods for calculating MX block scaling.
    There are four methods available:

    FLOOR: This method is recommended by the OCP MX Spec 1.0 and uses X = 2^floor(log2(max_abs(v))-max_exp).
           It result in overflow issues for large values and bad for gradient quantization.

    RCEIL: The method is to apply ceil to the ratio of max_abs(v) and max_pos.
           This method's detail is described in https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
           Section "Computing scaling and conversion factors for FP8 with UE8M0 scales"

    CEIL: This method avoids overflow issues, but small values may shift to 0 due to a large scaling factor.
           It uses X = 2^ceil(log2(max_abs(v))-max_exp).

    EVEN: This method is a trade-off between FLOOR and CEIL. It uses X = 2^(floor(log2(rounding(max_abs(v)))-max_exp)).
           It provides better accuracy for MX4 training compared to FLOOR and CEIL.
           Note: EVEN does not work with torch.compile yet:
           https://gist.github.com/vkuzo/1a04845cd503b1c75291aa1ea3bf79c4

    """

    FLOOR = "floor"
    RCEIL = "rceil"
    CEIL = "ceil"
    EVEN = "even"

    def __eq__(self, other):
        if isinstance(other, ScaleCalculationMode):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


def _validate_elem_dtype(elem_dtype):
    assert elem_dtype == torch.float4_e2m1fn_x2, (
        f"elem_dtype: expected torch.float4_e2m1fn_x2, got {elem_dtype}"
    )


def _validate_kernel_preference(kernel_preference, block_size, elem_dtype):
    if kernel_preference == KernelPreference.AUTO:
        assert elem_dtype == torch.float4_e2m1fn_x2, (
            f"unsupported {kernel_preference=}, {block_size=}, {elem_dtype=}"
        )
        assert block_size == 32, f"block_size must be 32, got {block_size}"
    else:
        assert kernel_preference == KernelPreference.EMULATED, (
            f"unsupported {kernel_preference=}, {block_size=}, {elem_dtype=}"
        )


@dataclass
class MXLinearConfig(AOBaseConfig):
    # block size for scaling, default is 32 to match
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
    # section 5.2
    block_size: int = 32

    # element dtype, used for activations, weights and gradients
    elem_dtype: Any = torch.float4_e2m1fn_x2

    # overrides for element dtype for weights and gradients
    # TODO(future PR): refactor to make this cleaner
    elem_dtype_weight_override: Optional[Any] = None
    elem_dtype_grad_output_override: Optional[Any] = None

    # defines the kernel preference, if the chosen kernel is not supported
    # on the given hardware an exception will be thrown
    kernel_preference: KernelPreference = KernelPreference.EMULATED

    scale_calculation_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR

    # kernel choices for mxfp8 casting (kept for compatibility with mx_mm autograd func)
    mxfp8_dim0_cast_kernel_choice: MXFP8Dim0CastKernelChoice = (
        MXFP8Dim0CastKernelChoice.TORCH
    )
    mxfp8_dim1_cast_kernel_choice: MXFP8Dim1CastKernelChoice = (
        MXFP8Dim1CastKernelChoice.TORCH
    )

    def __post_init__(self):
        _validate_elem_dtype(self.elem_dtype)
        _validate_kernel_preference(
            self.kernel_preference, self.block_size, self.elem_dtype
        )
        if self.elem_dtype_weight_override is not None:
            _validate_elem_dtype(self.elem_dtype_weight_override)
            assert self.kernel_preference == KernelPreference.EMULATED, "unsupported"
        if self.elem_dtype_grad_output_override is not None:
            _validate_elem_dtype(self.elem_dtype_grad_output_override)
            assert self.kernel_preference == KernelPreference.EMULATED, "unsupported"

    @staticmethod
    def from_recipe_name(
        recipe_name: Union[MXLinearRecipeName, str],
    ) -> "MXLinearConfig":
        """
        Input: `MXLinearRecipeName` value, or a string representing a `MXLinearRecipeName` value
        Output: a `MXLinearConfig` configured to implement the specified recipe
        """
        if type(recipe_name) == str:
            valid_names = [n.value for n in MXLinearRecipeName]
            assert recipe_name in valid_names, (
                f"recipe_name {recipe_name} not in valid names {valid_names}"
            )
            recipe_name = MXLinearRecipeName(recipe_name)

        if recipe_name is MXLinearRecipeName.MXFP4_EMULATED:
            return MXLinearConfig()
        elif recipe_name is MXLinearRecipeName.MXFP4_CUTLASS:
            return MXLinearConfig(
                kernel_preference=KernelPreference.AUTO,
            )
        else:
            raise AssertionError(f"unknown recipe_name {recipe_name}")

    def short_str(self) -> str:
        """
        Returns a concise representation of the current config.
        """
        s = f"bl_sz={self.block_size}, lp_dtype={DTYPE_TO_SHORT_STR[self.elem_dtype]}"
        if self.elem_dtype_weight_override is not None:
            s += (
                f", lp_w_override={DTYPE_TO_SHORT_STR[self.elem_dtype_weight_override]}"
            )
        if self.elem_dtype_grad_output_override is not None:
            s += f", lp_go_override={DTYPE_TO_SHORT_STR[self.elem_dtype_grad_output_override]}"
        s += f", kernel={self.kernel_preference.value}"
        if self.scale_calculation_mode != ScaleCalculationMode.FLOOR:
            s += f", scale_calculation_mode={self.scale_calculation_mode}"
        return s
