# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Low-precision attention for inference.

Only supports forward pass — backward is not supported by the underlying backends.
"""

from torchao.prototype.attention.api import (
    AttentionBackend,
    HadamardMode,
    apply_low_precision_attention,
)

__all__ = [
    "AttentionBackend",
    "HadamardMode",
    "apply_low_precision_attention",
]
