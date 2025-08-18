# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import types
from dataclasses import dataclass
from typing import Optional

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats import (
    MXGemmKernelChoice,
)
from torchao.prototype.mx_formats.config import (
    _validate_elem_dtype,
    _validate_gemm_kernel_choice,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4MMConfig,
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
)
from torchao.quantization.quant_api import to_linear_activation_quantized
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import (
    is_sm_at_least_100,
    torch_version_at_least,
)


# TODO The naming for these configs is a little weird, rename before moving to public API
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


@dataclass
class NVFP4InferenceConfig(AOBaseConfig):
    """
    NVIDIA FP4 (NVFP4) Inference Quantization Configuration

    This is a specialized configuration for NVIDIA's FP4 format.
    Configuration parameters:
    - mm_config: NVFP4MMConfig, which can be set to DYNAMIC or WEIGHT_ONLY (emulated mm in high precision)
    - use_triton_kernel: bool, whether to use fused triton kernel for activation scaling (default: False)
    - Data: float4_e2m1fn_x2
    - Scales: float8_e4m3fn
    - Block size: 16 along the reduction dim

    Note: Triton kernel only works with DYNAMIC mode and has constraints that input dimensions
    must satisfy M % 128 == 0 and K % 64 == 0. Will automatically fallback when constraints aren't met.
    """

    mm_config: NVFP4MMConfig = NVFP4MMConfig.DYNAMIC
    use_triton_kernel: bool = True

    def __post_init__(self):
        # Validate PyTorch version
        if not torch_version_at_least("2.8.0"):
            raise RuntimeError("NVFP4InferenceConfig requires PyTorch 2.8 or later")


@register_quantize_module_handler(NVFP4InferenceConfig)
def _nvfp4_inference_linear_transform(
    module: torch.nn.Linear, config: NVFP4InferenceConfig
):
    """Quantization handler for NVFP4InferenceConfig"""
    if config.mm_config == NVFP4MMConfig.DYNAMIC:
        assert is_sm_at_least_100(), (
            "NVFP4 DYNAMIC mode is only supported on sm100+ machines"
        )

    weight = module.weight

    if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
        raise RuntimeError(
            f"NVFP4 only supports weight shape divisible by 16, got {weight.shape}"
        )

    if module.bias is not None and weight.dtype == torch.float32:
        raise RuntimeError(
            "Bias is not supported when module weight is in fp32 (out_dtype=Float32). "
            "Please use bfloat16 or float16 weights, or remove the bias from the linear layer."
        )

    act_quant_kwargs = None
    if config.mm_config == NVFP4MMConfig.DYNAMIC:
        act_quant_kwargs = QuantizeTensorToNVFP4Kwargs()

    quantized_weight = NVFP4Tensor.to_nvfp4(
        weight,
        is_swizzled_scales=True,
        use_triton_kernel=False,  # Always use traditional construction for weights
        act_quant_kwargs=act_quant_kwargs,
    )
    # Set triton preference after construction
    quantized_weight.use_triton_kernel = config.use_triton_kernel
    module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


torch.serialization.add_safe_globals(
    [
        MXTensor,
        NVFP4Tensor,
        NVFP4MMConfig,
        MXGemmKernelChoice,
        _input_activation_quant_func_mxfp,
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
