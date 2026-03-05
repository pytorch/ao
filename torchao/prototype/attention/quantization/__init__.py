# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared FP8 quantization kernels for low-precision attention."""

from torchao.prototype.attention.quantization.quantization import (
    _fp8_sdpa_quantize,
)

__all__ = [
    "_fp8_sdpa_quantize",
]
