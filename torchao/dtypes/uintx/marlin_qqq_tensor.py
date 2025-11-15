# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing from torchao.dtypes.uintx.marlin_qqq_tensor is deprecated. "
    "Please use 'from torchao.prototype.dtypes import MarlinQQQLayout, MarlinQQQTensor' instead. "
    "This import path will be removed in a future release of torchao. "
    "See https://github.com/pytorch/ao/issues/2752 for more details.",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.dtypes.uintx.marlin_qqq_tensor import (  # noqa: F401
    MarlinQQQAQTTensorImpl,  # noqa: F401
    MarlinQQQLayout,  # noqa: F401
    MarlinQQQTensor,  # noqa: F401
    _linear_int8_act_int4_weight_marlin_qqq_check,  # noqa: F401
    _linear_int8_act_int4_weight_marlin_qqq_impl,  # noqa: F401
    to_marlinqqq_quantized_intx,  # noqa: F401
)
