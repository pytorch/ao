# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 attention using cuDNN per-tensor backend.
"""

from torchao.prototype.attention.fp8_cudnn.attention import (
    fp8_cudnn_rope_sdpa,
    fp8_cudnn_sdpa,
)
from torchao.prototype.attention.quantization import _fp8_per_tensor_sdpa_quantize

__all__ = [
    "fp8_cudnn_sdpa",
    "fp8_cudnn_rope_sdpa",
    "_fp8_per_tensor_sdpa_quantize",
]
