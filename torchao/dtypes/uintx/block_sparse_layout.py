# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Backward compatibility stub - imports from the new location
import warnings

warnings.warn(
    "Importing BlockSparseLayout from torchao.dtypes is deprecated. "
    "Please use 'from torchao.prototype.sparsity import BlockSparseLayout' instead. "
    "This import path will be removed in torchao v0.16.0.",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.sparsity.block_sparse_layout import (
    BlockSparseLayout,
    BlockSparseAQTTensorImpl,
    _linear_int8_act_int8_weight_block_sparse_check,
    _linear_int8_act_int8_weight_block_sparse_impl,
)
