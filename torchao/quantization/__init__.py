# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .smoothquant import *  # noqa: F403
from .quant_api import *  # noqa: F403
from .subclass import *  # noqa: F403
from .quant_primitives import *  # noqa: F403
from .utils import *  # noqa: F403
from .weight_only import *  # noqa: F403
from .unified import *
from .autoquant import *
from .granularity import *
from .linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)
from .linear_activation_scale import (
    to_weight_tensor_with_linear_activation_scale_metadata,
)

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
