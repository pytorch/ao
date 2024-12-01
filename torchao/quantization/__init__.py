# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchao.kernel import (
    int_scaled_matmul,
    safe_int_mm,
)

from .autoquant import (
    DEFAULT_AUTOQUANT_CLASS_LIST,
    DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST,
    DEFAULT_INT4_AUTOQUANT_CLASS_LIST,
    OTHER_AUTOQUANT_CLASS_LIST,
    autoquant,
)
from .GPTQ import (
    Int4WeightOnlyGPTQQuantizer,
    Int4WeightOnlyQuantizer,
    Int8DynActInt4WeightGPTQQuantizer,
    Int8DynActInt4WeightLinear,
    Int8DynActInt4WeightQuantizer,
)
from .granularity import (
    PerAxis,
    PerGroup,
    PerRow,
    PerTensor,
    PerToken,
)
from .linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)
from .linear_activation_scale import (
    to_weight_tensor_with_linear_activation_scale_metadata,
)
from .observer import (
    AffineQuantizedMinMaxObserver,
    AffineQuantizedObserverBase,
)
from .quant_api import (
    float8_dynamic_activation_float8_weight,
    float8_static_activation_float8_weight,
    float8_weight_only,
    fpx_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
    swap_conv2d_1x1_to_linear,
    uintx_weight_only,
)
from .quant_primitives import (
    MappingType,
    TorchAODType,
    ZeroPointDomain,
    choose_qparams_affine,
    choose_qparams_affine_floatx,
    choose_qparams_affine_with_min_max,
    choose_qparams_and_quantize_affine_hqq,
    dequantize_affine,
    dequantize_affine_floatx,
    fake_quantize_affine,
    fake_quantize_affine_cachemask,
    quantize_affine,
    quantize_affine_floatx,
)
from .smoothquant import (
    SmoothFakeDynamicallyQuantizedLinear,
    SmoothFakeDynQuantMixin,
    get_scale,
    set_smooth_fq_attribute,
    smooth_fq_linear_to_inference,
    swap_linear_with_smooth_fq_linear,
)
from .subclass import *  # noqa: F403
from .unified import Quantizer, TwoStepQuantizer
from .utils import (
    compute_error,
)
from .weight_only import WeightOnlyInt8QuantLinear

__all__ = [
    # top level API - auto
    "autoquant",
    "DEFAULT_AUTOQUANT_CLASS_LIST",
    "DEFAULT_INT4_AUTOQUANT_CLASS_LIST",
    "DEFAULT_FLOAT_AUTOQUANT_CLASS_LIST",
    "OTHER_AUTOQUANT_CLASS_LIST",
    # top level API - manual
    "quantize_",
    "int8_dynamic_activation_int4_weight",
    "int8_dynamic_activation_int8_weight",
    "int8_dynamic_activation_int8_semi_sparse_weight",
    "int4_weight_only",
    "int8_weight_only",
    "float8_weight_only",
    "float8_dynamic_activation_float8_weight",
    "float8_static_activation_float8_weight",
    "uintx_weight_only",
    "fpx_weight_only",
    # smooth quant - subject to change
    "swap_conv2d_1x1_to_linear",
    "get_scale",
    "SmoothFakeDynQuantMixin",
    "SmoothFakeDynamicallyQuantizedLinear",
    "swap_linear_with_smooth_fq_linear",
    "smooth_fq_linear_to_inference",
    "set_smooth_fq_attribute",
    "compute_error",
    # building blocks
    "to_linear_activation_quantized",
    "to_weight_tensor_with_linear_activation_scale_metadata",
    "AffineQuantizedMinMaxObserver",
    "AffineQuantizedObserverBase",
    # quant primitive ops
    "choose_qparams_affine",
    "choose_qparams_affine_with_min_max",
    "choose_qparams_affine_floatx",
    "quantize_affine",
    "quantize_affine_floatx",
    "dequantize_affine",
    "dequantize_affine_floatx",
    "choose_qparams_and_quantize_affine_hqq",
    "fake_quantize_affine",
    "fake_quantize_affine_cachemask",
    # operators/kernels
    "safe_int_mm",
    "int_scaled_matmul",
    # dataclasses and types
    "MappingType",
    "ZeroPointDomain",
    "TorchAODType",
    "PerTensor",
    "PerAxis",
    "PerGroup",
    "PerRow",
    "PerToken",
    "LinearActivationQuantizedTensor",
    "Int4WeightOnlyGPTQQuantizer",
    "Int4WeightOnlyQuantizer",
    "Int8DynActInt4WeightGPTQQuantizer",
    "Int8DynActInt4WeightQuantizer",
    "Int8DynActInt4WeightLinear",
    "WeightOnlyInt8QuantLinear",
    "TwoStepQuantizer",
    "Quantizer",
]
