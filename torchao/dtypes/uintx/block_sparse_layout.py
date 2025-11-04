# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing BlockSparseLayout from torchao.dtypes is deprecated. "
    "Please use 'from torchao.prototype.dtypes import BlockSparseLayout' instead. "
    "This import path will be removed in torchao v0.16.0. "
    "Please check issue: https://github.com/pytorch/ao/issues/2752 for more details. ",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.dtypes.uintx.block_sparse_layout import (
    BlockSparseAQTTensorImpl,  # noqa: F401
    BlockSparseLayout,  # noqa: F401
    _linear_int8_act_int8_weight_block_sparse_check,  # noqa: F401
    _linear_int8_act_int8_weight_block_sparse_impl,  # noqa: F401
)
