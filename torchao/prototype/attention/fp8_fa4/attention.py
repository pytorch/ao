# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 SDPA using FA4 backend.

When using these functions directly (not through apply_low_precision_attention),
you must activate FA4 yourself::

    activate_flash_attention_impl("FA4")
    try:
        out = fp8_fa4_sdpa(q, k, v, is_causal=True)
    finally:
        restore_flash_attention_impl()
"""

from functools import partial

from torchao.prototype.attention.shared_utils.attention import (
    _fp8_sdpa,
)

fp8_fa4_sdpa = partial(_fp8_sdpa, backend_name="FA4")
fp8_fa4_sdpa.__doc__ = _fp8_sdpa.__doc__
fp8_fa4_sdpa.__name__ = "fp8_fa4_sdpa"
fp8_fa4_sdpa.__qualname__ = "fp8_fa4_sdpa"
