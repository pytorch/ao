# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 RoPE + SDPA Inference

This module provides utilities for replacing RoPE (Rotary Position Embeddings) + SDPA
with a RoPE + FP8 quantization fused operation into FP8 SDPA using the FA3 backend for inference optimization.
"""

from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_attention import (
    fp8_rope_sdpa_flux,
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
    "wrap_module_with_fp8_rope_sdpa",
]
