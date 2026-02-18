# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.config import (
    _validate_elem_dtype,
    _validate_kernel_preference,
)
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    QuantizeTensorToMXKwargs,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4QuantizeKernelChoice,
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)
from torchao.quantization.quant_api import _quantization_type
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import (
    is_sm_at_least_100,
    torch_version_at_least,
)


class NVFP4ObservedLinear(torch.nn.Linear):
    """A linear module with an observer for NVFP4 static quantization.

    During calibration, this module tracks the per-tensor absolute maximum (amax)
    of the input activations. After calibration, the amax is converted to a
    per_tensor_scale using per_tensor_amax_to_scale() during the convert step.

    The block-level dynamic quantization remains unchanged - only the global
    per_tensor_scale is determined statically.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.amax = torch.tensor(0.0, device=device)

    def forward(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            input_amax = torch.max(torch.abs(input))
            self.amax = torch.max(self.amax.to(input.device), input_amax)
        output = F.linear(input, self.weight, self.bias)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear) -> "NVFP4ObservedLinear":
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            bias=float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear


@dataclass
class MXDynamicActivationMXWeightConfig(AOBaseConfig):
    """
    MX Format Inference Quantization

    This module provides support for running inference with float8 quantization using MX formats.

    Requirements:
    - NVIDIA SM100+ hardware (Blackwell or newer) is required for execution
    - PyTorch 2.5+ for proper serialization support
    """

    block_size: int = 32

    # Dtypes for Input and Weights, supports Fp8 and Fp4 formats
    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn

    # Which kernel to run for mm
    kernel_preference: KernelPreference = KernelPreference.AUTO

    # How to calculate the block scales
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL

    def __post_init__(self):
        assert self.activation_dtype == self.weight_dtype, (
            "For now - we only support matching input/weight dtypes."
        )
        _validate_elem_dtype(self.activation_dtype)
        _validate_elem_dtype(self.weight_dtype)
        _validate_kernel_preference(
            self.kernel_preference, self.block_size, self.weight_dtype
        )


def _linear_extra_repr(self):
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={_quantization_type(self.weight)}"


@register_quantize_module_handler(MXDynamicActivationMXWeightConfig)
def _mx_inference_linear_transform(
    module: torch.nn.Module, config: MXDynamicActivationMXWeightConfig
):
    weight = module.weight

    assert weight.dtype == torch.bfloat16, (
        f"Only supporting bf16 out dtype for now, got {weight.dtype}"
    )
    act_quant_kwargs = QuantizeTensorToMXKwargs(
        elem_dtype=config.activation_dtype,
        block_size=config.block_size,
        kernel_preference=config.kernel_preference,
        is_swizzled_scales=True,
        scaling_mode=config.scaling_mode,
    )

    # Convert weight to MX Tensor
    quantized_weight = MXTensor.to_mx(
        weight,
        config.weight_dtype,
        block_size=config.block_size,
        kernel_preference=config.kernel_preference,
        act_quant_kwargs=act_quant_kwargs,
        is_swizzled_scales=True,
        scaling_mode=config.scaling_mode,
    )

    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


@dataclass
class NVFP4DynamicActivationNVFP4WeightConfig(AOBaseConfig):
    """
    NVIDIA FP4 (NVFP4) Inference Quantization Configuration

    This is a specialized configuration for NVIDIA's FP4 format.
    NVFP4 uses "double quantization" with two scale levels:
    - A global per_tensor_scale (float32)
    - Per-block scales (float8_e4m3fn, block_size=16), always dynamically calculated

    The activation per_tensor_scale can be determined in two ways:

    1. Dynamic per_tensor_scale (default, step=None, use_dynamic_per_tensor_scale=True):
        - Both weight and activation per_tensor_scale are computed at runtime
          from the tensor amax

    2. Static per_tensor_scale via observer flow (step="prepare"/"convert"):
        - Weight per_tensor_scale is computed from weight amax at convert time
        - Activation per_tensor_scale is determined statically during calibration:
          step="prepare" inserts observers, then after running calibration data,
          step="convert" extracts the observed amax and bakes the activation
          per_tensor_scale into the quantized weight tensor
        - At inference, the static activation per_tensor_scale is read from the
          weight tensor instead of being computed dynamically
        - Note: activation per-block scales are still computed dynamically at
          inference time

    Note: When step is specified, use_dynamic_per_tensor_scale is automatically
    set to False.

    Configuration parameters:
    - nvfp4_quantize_kernel_choice: NVFP4QuantizeKernelChoice, kernel choice for quantization (default: NVFP4QuantizeKernelChoice.TRITON)
    - use_dynamic_per_tensor_scale: bool, whether to dynamically compute per tensor scale (default: True)
    - step: Optional[QuantizationStep], the quantization step for observer-based flow
    - Data: float4_e2m1fn_x2
    - Scales: float8_e4m3fn
    - Block size: 16 along the reduction dim

    Note: Triton kernel only works with DYNAMIC mode and has constraints that input dimensions
    must satisfy M % 128 == 0 and K % 64 == 0. Will automatically fallback when constraints aren't met.
    """

    nvfp4_quantize_kernel_choice: NVFP4QuantizeKernelChoice = (
        NVFP4QuantizeKernelChoice.TRITON
    )
    use_dynamic_per_tensor_scale: bool = True
    step: Optional["QuantizationStep"] = None

    def __post_init__(self):
        if isinstance(self.step, str):
            self.step = QuantizationStep(self.step)
        # Validate PyTorch version
        if not torch_version_at_least("2.8.0"):
            raise RuntimeError(
                "NVFP4DynamicActivationNVFP4WeightConfig requires PyTorch 2.8 or later"
            )

        if self.step is not None:
            # Static quantization implies use_dynamic_per_tensor_scale=False
            self.use_dynamic_per_tensor_scale = False

        if self.nvfp4_quantize_kernel_choice == NVFP4QuantizeKernelChoice.FLASHINFER:
            if self.step is None and not self.use_dynamic_per_tensor_scale:
                raise ValueError(
                    "FLASHINFER kernel choice requires per_tensor_scale. "
                    "Use step='prepare'/'convert' for static quantization, "
                    "or set use_dynamic_per_tensor_scale=True."
                )


@register_quantize_module_handler(NVFP4DynamicActivationNVFP4WeightConfig)
def _nvfp4_inference_linear_transform(
    module: torch.nn.Linear, config: NVFP4DynamicActivationNVFP4WeightConfig
):
    """Quantization handler for NVFP4DynamicActivationNVFP4WeightConfig

    Behavior depends on the step:
    - PREPARE: Insert NVFP4ObservedLinear to collect activation statistics
    - CONVERT: Extract amax from observer, compute static per_tensor_scale, quantize
    - None (default): Original dynamic quantization behavior
    """
    weight = module.weight
    if weight.shape[-2] % 16 != 0 or weight.shape[-1] % 16 != 0:
        raise RuntimeError(
            f"NVFP4 only supports weight shape with last 2 dims divisible by 16, got {weight.shape}"
        )

    step = config.step
    if step == QuantizationStep.PREPARE or step == "prepare":
        return NVFP4ObservedLinear.from_float(module)

    elif step == QuantizationStep.CONVERT or step == "convert":
        if not isinstance(module, NVFP4ObservedLinear):
            return module

        # Compute activation per_tensor_scale from observed amax
        act_per_tensor_scale = per_tensor_amax_to_scale(module.amax)

        # Weight quantization (same as dynamic path)

        tensor_amax = torch.max(torch.abs(weight))
        weight_per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=False,
            nvfp4_quantize_kernel_choice=config.nvfp4_quantize_kernel_choice,
            is_swizzled_scales=True,
        )

        quantized_weight = NVFP4Tensor.to_nvfp4(
            weight,
            per_tensor_scale=weight_per_tensor_scale,
            act_per_tensor_scale=act_per_tensor_scale.detach(),
            is_swizzled_scales=True,
            nvfp4_quantize_kernel_choice=NVFP4QuantizeKernelChoice.TORCH,  # Always use traditional construction for weights
            act_quant_kwargs=act_quant_kwargs,
        )
        quantized_weight.nvfp4_quantize_kernel_choice = (
            config.nvfp4_quantize_kernel_choice
        )

        # Create new Linear (not observed) with quantized weight
        linear = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        linear.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
        linear.bias = module.bias
        linear.extra_repr = types.MethodType(_linear_extra_repr, linear)
        return linear

    elif step is None:
        # Dynamic quantization
        assert is_sm_at_least_100(), (
            "NVFP4 DYNAMIC mode is only supported on sm100+ machines"
        )

        per_tensor_scale = None
        if config.use_dynamic_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(weight))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=config.use_dynamic_per_tensor_scale,
            nvfp4_quantize_kernel_choice=config.nvfp4_quantize_kernel_choice,
            is_swizzled_scales=True,
        )

        quantized_weight = NVFP4Tensor.to_nvfp4(
            weight,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            nvfp4_quantize_kernel_choice=NVFP4QuantizeKernelChoice.TORCH,  # Always use traditional construction for weights
            act_quant_kwargs=act_quant_kwargs,
        )
        quantized_weight.nvfp4_quantize_kernel_choice = (
            config.nvfp4_quantize_kernel_choice
        )
        module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
        return module

    else:
        raise ValueError(
            f"Unexpected step: {step}. Expected one of {[s.value for s in QuantizationStep]} or None."
        )


@dataclass
class NVFP4WeightOnlyConfig(AOBaseConfig):
    use_dynamic_per_tensor_scale: bool = True

    def __post_init__(self):
        # Validate PyTorch version
        if not torch_version_at_least("2.8.0"):
            raise RuntimeError(
                "NVFP4DynamicActivationNVFP4WeightConfig requires PyTorch 2.8 or later"
            )


@register_quantize_module_handler(NVFP4WeightOnlyConfig)
def _nvfp4_weight_only_linear_transform(
    module: torch.nn.Linear, config: NVFP4WeightOnlyConfig
):
    """Quantization handler for NVFP4WeightOnlyConfig"""
    weight = module.weight

    if weight.shape[-2] % 16 != 0 or weight.shape[-1] % 16 != 0:
        raise RuntimeError(
            f"NVFP4 only supports weight shape with last 2 dims divisible by 16, got {weight.shape}"
        )

    per_tensor_scale = None
    if config.use_dynamic_per_tensor_scale:
        tensor_amax = torch.max(torch.abs(weight))
        per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

    quantized_weight = NVFP4Tensor.to_nvfp4(
        weight,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=None,
    )
    # Set triton preference after construction
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


torch.serialization.add_safe_globals(
    [
        MXTensor,
        NVFP4Tensor,
        QuantizeTensorToMXKwargs,
        QuantizeTensorToNVFP4Kwargs,
        ScaleCalculationMode,
    ]
)


import torch.nn as nn


def _auto_filter_for_nfp4(mod: nn.Module, fqn: str) -> bool:
    """Generic Filter fn for NVFP4 that is best practice for most models."""
    # Define any FQNs you want to exclude directly in the function
    filter_fqns = ["embedder", "embed", "embedding", "time_text_embed"]

    # Only support Linear modules
    if not isinstance(mod, nn.Linear):
        return False

    # If the fqn matches any filtered fqn, then we should not convert this module
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)
    if is_filtered_fqn:
        return False

    # All dims must be divisible by 16 due to float8 hardware requirements.
    N, K = mod.weight.shape
    dims_multiples_of_16 = K % 16 == 0 and N % 16 == 0
    if not dims_multiples_of_16:
        return False
    if N <= 64:
        print("skiping small linear layer")
        # TODO cublas doesn't like this one
        return False

    # Dims below these thresholds may result in worse performance
    if K <= 1024 and N <= 1024:
        print("skiping small linear layer")
        return False
    return True
