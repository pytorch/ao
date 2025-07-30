# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Explicit imports to improve import speed - import the most commonly used functions
from .quant_api import (
    quantize_,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int4_weight_only,
    int8_weight_only,
    uintx_weight_only,
    fpx_weight_only,
    float8_weight_only,
    float8_dynamic_activation_float8_weight,
    float8_static_activation_float8_weight,
    swap_conv2d_1x1_to_linear,
)
from .autoquant import autoquant

# Lazy loading for other modules
_LAZY_IMPORTS = {
    # From smoothquant
    "swap_linear_with_smooth_fq_linear": "smoothquant",
    "smooth_fq_linear_to_inference": "smoothquant",
    "SmoothFakeDynQuantMixin": "smoothquant",
    "SmoothFakeDynamicallyQuantizedLinear": "smoothquant",
    "set_smooth_fq_attribute": "smoothquant",
    # From quant_primitives
    "quantize_affine": "quant_primitives",
    "dequantize_affine": "quant_primitives",
    "choose_qprams_affine": "quant_primitives",
    "MappingType": "quant_primitives",
    "ZeroPointDomain": "quant_primitives",
    "safe_int_mm": "quant_primitives",
    # From utils
    "compute_error": "utils",
    # From unified
    "Quantizer": "unified",
    # From granularity
    "PerTensor": "granularity",
    "PerAxis": "granularity",
    "PerGroup": "granularity",
    # From GPTQ
    "Int4WeightOnlyGPTQQuantizer": "GPTQ",
    "Int4WeightOnlyQuantizer": "GPTQ",
    # From linear_activation modules
    "LinearActivationQuantizedTensor": "linear_activation_quantized_tensor",
    "to_linear_activation_quantized": "linear_activation_quantized_tensor",
    "to_weight_tensor_with_linear_activation_scale_metadata": "linear_activation_scale",
    # From autoquant
    "DEFAULT_AUTOQUANT_CLASS_LIST": "autoquant",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST": "autoquant", 
    "OTHER_AUTOQUANT_CLASS_LIST": "autoquant",
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        module = __import__(f"torchao.quantization.{module_name}", fromlist=[name])
        attr = getattr(module, name)
        # Cache for future access
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "swap_conv2d_1x1_to_linear"
    "safe_int_mm",
    "autoquant",
    "DEFAULT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST",
    "OTHER_AUTOQUANT_CLASS_LIST",
    "get_scale",
    "SmoothFakeDynQuantMixin",
    "SmoothFakeDynamicallyQuantizedLinear",
    "swap_linear_with_smooth_fq_linear",
    "smooth_fq_linear_to_inference",
    "set_smooth_fq_attribute",
    "compute_error",
    "Int4WeightOnlyGPTQQuantizer",
    "Int4WeightOnlyQuantizer",
    "quantize_affine",
    "dequantize_affine",
    "choose_qprams_affine",
    "quantize_",
    "int8_dynamic_activation_int4_weight",
    "int8_dynamic_activation_int8_weight",
    "int8_dynamic_activation_int8_semi_sparse_weight",
    "int4_weight_only",
    "int8_weight_only",
    "uintx_weight_only",
    "fpx_weight_only",
    "LinearActivationQuantizedTensor",
    "to_linear_activation_quantized",
    "to_weight_tensor_with_linear_activation_scale_metadata",
    "float8_weight_only",
    "float8_dynamic_activation_float8_weight",
    "float8_static_activation_float8_weight"
]
