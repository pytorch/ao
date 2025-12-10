# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing from torchao.dtypes.uintx.dyn_int8_act_int4_wei_cpu_layout is deprecated. "
    "Please use 'from torchao.prototype.dtypes import Int8DynamicActInt4WeightCPULayout' instead. "
    "This import path will be removed in a future release of torchao. "
    "See https://github.com/pytorch/ao/issues/2752 for more details.",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.dtypes.uintx.dyn_int8_act_int4_wei_cpu_layout import (  # noqa: F401
    DA8W4CPUAQTTensorImpl,  # noqa: F401
    Int8DynamicActInt4WeightCPULayout,  # noqa: F401
    _aqt_is_int8,  # noqa: F401
    _aqt_is_uint4,  # noqa: F401
    _aqt_is_uint8,  # noqa: F401
    _linear_int8_act_int4_weight_cpu_check,  # noqa: F401
    _linear_int8_act_int4_weight_cpu_impl,  # noqa: F401
)
