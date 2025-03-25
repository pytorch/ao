# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TODO: delete this file.
# File is kept in torchao/experimental to avoid breaking existing code
import logging

logging.warning(
    "torchao.experimental.packed_linear_int8_dynamic_activation_intx_weight_layout.py is deprecated and will be removed.  Please use torchao.dtypes.uintx.packed_linear_int8_dynamic_activation_intx_weight_layout.py instead."
)
from torchao.dtypes.uintx.packed_linear_int8_dynamic_activation_intx_weight_layout import (
    PackedLinearInt8DynamicActivationIntxWeightLayout,
    Target,
    to_affine_quantized_packed_linear_int8_dynamic_activation_intx_weight,
)

to_affine_quantized_intx_experimental = (
    to_affine_quantized_packed_linear_int8_dynamic_activation_intx_weight
)
__all__ = [
    "to_affine_quantized_intx_experimental",
    "PackedLinearInt8DynamicActivationIntxWeightLayout",
    "Target",
]
