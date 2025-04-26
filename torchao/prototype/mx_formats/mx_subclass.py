# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import types
from dataclasses import dataclass
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
from torchao.quantization.quant_api import to_linear_activation_quantized
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, is_sm_at_least_100


@dataclass
class MXFPConfig(AOBaseConfig):
    block_size: int = 32

    # Dtypes for Input and Weights
    activation_dtype: torch.dtype = torch.float8_e4m3fn
    weight_dtype: torch.dtype = torch.float8_e4m3fn

    # Which kernel to run for mm
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.CUBLAS

    # Set some magic perf settings
    set_inductor_config: bool = True

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


@register_quantize_module_handler(MXFPConfig)
def _mx_inference_linear_transform(module: torch.nn.Module, config: MXFPConfig):
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


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals(
        [MXTensor, MXGemmKernelChoice, _input_activation_quant_func_mxfp]
    )
