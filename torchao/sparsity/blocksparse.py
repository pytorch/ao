# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.warn(
    "torchao.sparsity.blocksparse is deprecated, please use torchao.prototype.sparsity.blocksparse instead. "
    "See https://github.com/pytorch/ao/issues/4230 for more details.",
    DeprecationWarning,
    stacklevel=2,
)

from torchao.prototype.sparsity.blocksparse import BlockSparseTensor  # noqa: F401
