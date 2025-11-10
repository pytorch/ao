# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing from torchao.dtypes.floatx.floatx_tensor_core_layout is deprecated. "
    "Please use 'from torchao.prototype.dtypes.floatx.floatx_tensor_core_layout import ...' instead. "
    "This import path will be removed in a future torchao release. "
    "Please check issue: https://github.com/pytorch/ao/issues/2752 for more details. ",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public symbols from the new location for backward compatibility
from torchao.prototype.dtypes.floatx.floatx_tensor_core_layout import (
    FloatxTensorCoreAQTTensorImpl,
    FloatxTensorCoreLayout,
    _linear_f16_bf16_act_floatx_weight_check,
    _linear_f16_bf16_act_floatx_weight_impl,
    _pack_tc_floatx,
    _pack_tc_fp6,
    from_scaled_tc_floatx,
    pack_tc_floatx,
    to_scaled_tc_floatx,
    unpack_tc_floatx,
)

__all__ = [
    "FloatxTensorCoreAQTTensorImpl",
    "FloatxTensorCoreLayout",
    "_linear_f16_bf16_act_floatx_weight_check",
    "_linear_f16_bf16_act_floatx_weight_impl",
    "_pack_tc_floatx",
    "_pack_tc_fp6",
    "from_scaled_tc_floatx",
    "pack_tc_floatx",
    "to_scaled_tc_floatx",
    "unpack_tc_floatx",
]
