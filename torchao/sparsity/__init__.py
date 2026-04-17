# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from .sparse_api import (
    apply_fake_sparsity,
    semi_sparse_weight,
    sparsify_,
)
from .supermask import SupermaskLinear
from .utils import PerChannelNormObserver  # noqa: F403
from .wanda import WandaSparsifier  # noqa: F403

__all__ = [
    "WandaSparsifier",
    "SupermaskLinear",
    "PerChannelNormObserver",
    "apply_fake_sparsity",
    "sparsify_",
    "semi_sparse_weight",
    "block_sparse_weight",
    "BlockSparseWeightConfig",
]


def __getattr__(name):
    if name in ("block_sparse_weight", "BlockSparseWeightConfig"):
        warnings.warn(
            f"Importing {name} from torchao.sparsity is deprecated, "
            "please use torchao.prototype.sparsity instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from torchao.prototype.sparsity.blocksparse_config import (
            BlockSparseWeightConfig as _BlockSparseWeightConfig,
        )

        if name == "BlockSparseWeightConfig":
            return _BlockSparseWeightConfig
        return _BlockSparseWeightConfig  # block_sparse_weight is alias
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
