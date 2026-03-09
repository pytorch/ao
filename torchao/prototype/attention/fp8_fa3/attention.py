# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 SDPA using FA3 backend.

When using these functions directly (not through apply_low_precision_attention),
you must activate FA3 yourself::

    activate_flash_attention_impl("FA3")
    try:
        out = fp8_fa3_sdpa(q, k, v, is_causal=True)
    finally:
        restore_flash_attention_impl()
"""

from functools import partial

from torchao.prototype.attention.shared_utils.attention import (
    _fp8_rope_sdpa,
    _fp8_sdpa,
)

fp8_fa3_sdpa = partial(_fp8_sdpa, backend_name="FA3")
fp8_fa3_sdpa.__doc__ = _fp8_sdpa.__doc__
fp8_fa3_sdpa.__name__ = "fp8_fa3_sdpa"
fp8_fa3_sdpa.__qualname__ = "fp8_fa3_sdpa"

fp8_fa3_rope_sdpa = partial(_fp8_rope_sdpa, backend_name="FA3")
fp8_fa3_rope_sdpa.__doc__ = _fp8_rope_sdpa.__doc__
fp8_fa3_rope_sdpa.__name__ = "fp8_fa3_rope_sdpa"
fp8_fa3_rope_sdpa.__qualname__ = "fp8_fa3_rope_sdpa"
