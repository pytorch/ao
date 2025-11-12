# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing from torchao.dtypes.uintx.gemlite_layout is deprecated. "
    "Please use 'from torchao.prototype.dtypes import GemlitePackedLayout' instead. "
    "This import path will be removed in a future release of torchao. "
    "See https://github.com/pytorch/ao/issues/2752 for more details.",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.dtypes.uintx.gemlite_layout import (  # noqa: F401
    GemliteAQTTensorImpl,  # noqa: F401
    GemlitePackedLayout,  # noqa: F401
    _linear_fp_act_int4_weight_gemlite_check,  # noqa: F401
    _linear_fp_act_int4_weight_gemlite_impl,  # noqa: F401
    _same_metadata,  # noqa: F401
    get_gemlite_aqt_kwargs,  # noqa: F401
    get_gemlite_quant_kwargs,  # noqa: F401
)
