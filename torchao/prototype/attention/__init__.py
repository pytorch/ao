# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Low-precision attention for inference.

This module provides APIs for running attention with reduced precision
(e.g., FP8) for faster inference. It can be extended to support different
quantization strategies and different PyTorch core attention backends.

Note: Low-precision attention only supports inference (forward pass).
Backward pass is not supported by the underlying backends.

Note: apply_low_precision_attention replaces all F.scaled_dot_product_attention calls

Example::

    from torchao.prototype.attention import apply_low_precision_attention

    model = MyTransformer()
    model = apply_low_precision_attention(model)
    output = model(inputs)
"""

from torchao.prototype.attention.api import apply_low_precision_attention
from torchao.prototype.attention.config import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
)

__all__ = [
    # Config
    "LowPrecisionAttentionConfig",
    "AttentionBackend",
    # API
    "apply_low_precision_attention",
]
