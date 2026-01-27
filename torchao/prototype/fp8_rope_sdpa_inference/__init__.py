# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 RoPE + SDPA Inference

This module provides utilities for fusing RoPE (Rotary Position Embedding)
and FP8 quantization with scaled dot-product attention
for inference optimization.
"""

from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_attention import (
    fp8_rope_sdpa_flux,
    fp8_rope_sdpa_flux_reference,
)
from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_quantization import (
    fp8_rope_quantize_func,
)
from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_utils import (
    fp8_rope_sdpa_context,
    wrap_module_with_fp8_rope_sdpa,
)

__all__ = [
    "fp8_rope_quantize_func",
    "fp8_rope_sdpa_context",
    "fp8_rope_sdpa_flux",
    "fp8_rope_sdpa_flux_reference",
    "wrap_module_with_fp8_rope_sdpa",
]
