# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import types
from dataclasses import dataclass, field
from typing import Optional

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats import (
    MXGemmKernelChoice,
)
from torchao.prototype.mx_formats.config import (
    _validate_elem_dtype,
    _validate_gemm_kernel_choice,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.quantization.quant_api import to_linear_activation_quantized
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_100,
)


# Note: This API is extra prototype and will change in the future
@dataclass
class MXFPInferenceConfig(AOBaseConfig):
    """
    MX Format Inference Quantization

    This module provides support for running inference with float8 quantization using MX formats.
    The quantization flow works as follows:

    1. Weight Quantization:
    - In _mx_inference_linear_transform(), the module's weight is converted to an MXTensor
    - The weight is quantized to the specified dtype (float8_e4m3fn by default)
    - This happens when quantize_() is called with an MXFPInferenceConfig

    2. Activation Quantization:
    - A callable (_input_activation_quant_func_mxfp) is defined that will quantize
        activations during inference to the same dtype
    - This function is passed to to_linear_activation_quantized() along with the
        already-quantized weight

    3. Runtime Flow:
    - When the quantized module is called, the input goes through the LinearActivationQuantizedTensor
    - The input (activation) is quantized just-in-time using the provided function
    - The MX quantized activation and MX weight are used together in F.linear

    Requirements:
    - NVIDIA SM100+ hardware (Blackwell or newer) is required for execution
    - PyTorch 2.5+ for proper serialization support

    See also:
    - LinearActivationQuantizedTensor in torchao.quantization.quant_api
    - MXTensor in torchao.prototype.mx_formats.mx_tensor
    """

    block_size: int = 32

    # Dtypes for Input and Weights, supports Fp8 and Fp4 formats
    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn

    # Which kernel to run for mm
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.CUBLAS

    # Set some magic perf settings
    set_inductor_config: bool = False

    def __post_init__(self):
        assert self.activation_dtype == self.weight_dtype, (
            "For now - we only support matching input/weight dtypes."
        )
        _validate_elem_dtype(self.activation_dtype)
        _validate_elem_dtype(self.weight_dtype)
        _validate_gemm_kernel_choice(
            self.gemm_kernel_choice, self.block_size, self.weight_dtype
        )


def _linear_extra_repr(self):
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={repr(self.weight)}"


def _input_activation_quant_func_mxfp(
    x: torch.Tensor,
    activation_dtype: torch.dtype,
    block_size: int,
    scale: Optional[torch.Tensor] = None,
):
    """ """

    # TODO scale for static quant

    activation = MXTensor.to_mx(
        x,
        activation_dtype,
        block_size=block_size,
        gemm_kernel_choice=None,  # Get from weight
        pack_fp6=False,  # TODO
    )
    return activation


@register_quantize_module_handler(MXFPInferenceConfig)
def _mx_inference_linear_transform(
    module: torch.nn.Module, config: MXFPInferenceConfig
):
    # TODO Sm120 has slightly more restrictive reqs
    # TODO handle AMD
    assert is_sm_at_least_100(), "MXFP is only supported on sm100 machiens for now"
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    activation_dtype = config.activation_dtype
    weight_dtype = config.weight_dtype
    weight = module.weight

    assert weight.dtype == torch.bfloat16, (
        f"Only supporting bf16 out dtype for now, got {weight.dtype}"
    )

    # Convert weight to MX Tensor
    quantized_weight = MXTensor.to_mx(
        weight,
        weight_dtype,
        block_size=config.block_size,
        gemm_kernel_choice=config.gemm_kernel_choice,
        pack_fp6=False,  # TODO
    )

    input_quant_func = _input_activation_quant_func_mxfp
    input_quant_kwargs = {
        "block_size": config.block_size,
        "activation_dtype": activation_dtype,
        "scale": None,
    }

    quantized_weight = to_linear_activation_quantized(
        quantized_weight, input_quant_func, quant_kwargs=input_quant_kwargs
    )

    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


def _get_nvfp4_dtype():
    """Factory function for NVFP4 dtype defaults."""
    if not TORCH_VERSION_AT_LEAST_2_8:
        raise RuntimeError("NVFP4InferenceConfig requires PyTorch 2.8 or later")
    return torch.float4_e2m1fn_x2


@dataclass
class NVFP4InferenceConfig(AOBaseConfig):
    """
    NVIDIA FP4 (NVFP4) Inference Quantization Configuration

    This is a specialized configuration for NVIDIA's FP4 format with UE4M3 scales.
    It provides defaults optimized for NVFP4:
    - Data: float4_e2m1fn_x2
    - Scales: float8_e4m3fn (UE4M3)
    - Block size: 16 (required for NVFP4)
    - CUBLAS kernel (optimized for VEC16_UE4M3)
    """

    block_size: int = 16  # NVFP4 requires block size 16

    # NVFP4 uses FP4 data
    activation_dtype: torch.dtype = field(default_factory=_get_nvfp4_dtype)
    weight_dtype: torch.dtype = field(default_factory=_get_nvfp4_dtype)

    # NVFP4 uses E4M3 scales
    scale_dtype: torch.dtype = torch.float8_e4m3fn

    # CUBLAS is preferred for NVFP4 with VEC16_UE4M3 support
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.CUBLAS

    def __post_init__(self):
        # Validate NVFP4 constraints
        if not TORCH_VERSION_AT_LEAST_2_8:
            raise RuntimeError("NVFP4InferenceConfig requires PyTorch 2.8 or later")

        assert self.activation_dtype == torch.float4_e2m1fn_x2, (
            f"NVFP4 requires activation_dtype=float4_e2m1fn_x2, got {self.activation_dtype}"
        )
        assert self.weight_dtype == torch.float4_e2m1fn_x2, (
            f"NVFP4 requires weight_dtype=float4_e2m1fn_x2, got {self.weight_dtype}"
        )
        assert self.scale_dtype == torch.float8_e4m3fn, (
            f"NVFP4 requires scale_dtype=float8_e4m3fn, got {self.scale_dtype}"
        )
        assert self.block_size == 16, (
            f"NVFP4 requires block_size=16, got {self.block_size}"
        )


def _input_activation_quant_func_nvfp4(
    x: torch.Tensor,
    block_size: int = 16,
    scale: Optional[torch.Tensor] = None,
):
    """NVFP4-specific activation quantization function"""
    # TODO: scale for static quant
    activation = NVFP4Tensor.to_nvfp4(
        x,
        block_size=block_size,
    )
    return activation


@register_quantize_module_handler(NVFP4InferenceConfig)
def _nvfp4_inference_linear_transform(
    module: torch.nn.Module, config: NVFP4InferenceConfig
):
    """Quantization handler for NVFP4InferenceConfig"""
    assert is_sm_at_least_100(), "NVFP4 is only supported on sm100+ machines"
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    weight = module.weight
    assert weight.dtype == torch.bfloat16, (
        f"Only supporting bf16 out dtype for now, got {weight.dtype}"
    )

    # Convert weight to NVFP4 Tensor
    quantized_weight = NVFP4Tensor.to_nvfp4(
        weight,
        block_size=config.block_size,
    )

    input_quant_func = _input_activation_quant_func_nvfp4
    input_quant_kwargs = {
        "block_size": config.block_size,
        "scale": None,
    }

    quantized_weight = to_linear_activation_quantized(
        quantized_weight, input_quant_func, quant_kwargs=input_quant_kwargs
    )

    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals(
        [
            MXTensor,
            NVFP4Tensor,
            MXGemmKernelChoice,
            _input_activation_quant_func_mxfp,
            _input_activation_quant_func_nvfp4,
        ]
    )
