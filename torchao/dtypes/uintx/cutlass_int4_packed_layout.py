# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing from torchao.dtypes is deprecated. "
    "Please use 'from torchao.prototype.dtypes import CutlassInt4PackedLayout' instead. "
    "This import path will be removed in torchao v0.16.0. "
    "Please check issue: https://github.com/pytorch/ao/issues/2752 for more details. ",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.dtypes.uintx.cutlass_int4_packed_layout import (  # noqa: F401
    CutlassInt4PackedLayout,  # noqa: F401
    Int4PackedTensorImpl,  # noqa: F401
    _linear_int4_act_int4_weight_cutlass_check,  # noqa: F401
    _linear_int4_act_int4_weight_cutlass_impl,  # noqa: F401
    _linear_int8_act_int4_weight_cutlass_check,  # noqa: F401
    _linear_int8_act_int4_weight_cutlass_impl,  # noqa: F401
)
