# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Low-precision attention for inference.

Only supports forward pass — backward is not supported by the underlying backends.
"""

from torchao.prototype.attention.api import apply_low_precision_attention
from torchao.prototype.attention.config import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
)

__all__ = [
    "LowPrecisionAttentionConfig",
    "AttentionBackend",
    "apply_low_precision_attention",
]
