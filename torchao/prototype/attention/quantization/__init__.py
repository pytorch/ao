# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared FP8 quantization kernels for low-precision attention backends.

This module provides backend-agnostic FP8 quantization for Q, K, V tensors,
used by both FA3 and FA4 backends.
"""

from torchao.prototype.attention.quantization.quantization import (
    _fp8_rope_sdpa_quantize,
    _fp8_sdpa_quantize,
)

__all__ = [
    "_fp8_sdpa_quantize",
    "_fp8_rope_sdpa_quantize",
]
