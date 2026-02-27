# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 attention implementation using FA4 backend.

Use apply_low_precision_attention() from torchao.prototype.attention as the public API.
For lower-level access, use fp8_fa4_sdpa() directly.
"""

from torchao.prototype.attention.fp8_fa4.attention import fp8_fa4_sdpa
from torchao.prototype.attention.quantization import _fp8_sdpa_quantize

__all__ = [
    "fp8_fa4_sdpa",
    "_fp8_sdpa_quantize",
]
