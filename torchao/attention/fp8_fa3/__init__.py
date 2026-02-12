# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Internal: FP8 attention implementation using FA3 backend.

Use apply_low_precision_attention() from torchao.attention as the public API.
"""

from torchao.attention.fp8_fa3.attention import _fp8_fa3_sdpa
from torchao.attention.fp8_fa3.quantization import _fp8_sdpa_quantize
from torchao.attention.fp8_fa3.wrappers import (
    _fp8_fa3_attention_context,
    _wrap_model_with_fp8_fa3_attention,
)

__all__ = [
    "_fp8_fa3_sdpa",
    "_fp8_sdpa_quantize",
    "_fp8_fa3_attention_context",
    "_wrap_model_with_fp8_fa3_attention",
]
