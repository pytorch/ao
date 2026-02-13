# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 attention implementation using FA3 backend.

Use apply_low_precision_attention() from torchao.prototype.attention as the public API
for model-level wrapping, or use fp8_fa3_sdpa() directly for lower-level access.
"""

from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa
from torchao.prototype.attention.fp8_fa3.quantization import _fp8_sdpa_quantize

try:
    from torchao.prototype.attention.fp8_fa3.wrappers import (
        _fp8_fa3_attention_context,
        _wrap_model_with_fp8_fa3_attention,
    )
except ImportError:
    pass

__all__ = [
    "fp8_fa3_sdpa",
    "_fp8_sdpa_quantize",
    "_fp8_fa3_attention_context",
    "_wrap_model_with_fp8_fa3_attention",
]
