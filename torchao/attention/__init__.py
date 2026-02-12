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

Example::

    from torchao.attention import (
        LowPrecisionAttentionConfig,
        apply_low_precision_attention,
    )

    model = MyTransformer()

    # Simple usage - auto-selects backend and uses basic FP8
    apply_low_precision_attention(model)

    # With options
    config = LowPrecisionAttentionConfig(fuse_rope=True)
    apply_low_precision_attention(model, config)

    model = torch.compile(model)
    output = model(inputs)
"""

from torchao.attention.api import apply_low_precision_attention
from torchao.attention.config import AttentionBackend, LowPrecisionAttentionConfig

__all__ = [
    # Config
    "LowPrecisionAttentionConfig",
    "AttentionBackend",
    # API
    "apply_low_precision_attention",
]
