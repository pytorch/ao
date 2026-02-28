# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 scaled dot-product attention using FA4 backend.

This is a thin wrapper around the shared implementation in
``shared_utils/attention.py``.  It exists so that the FA4 backend has
a named entry point (``fp8_fa4_sdpa``) and backend-specific error messages.

.. important::

    When using this function directly (not through
    ``apply_low_precision_attention``), you **must** activate the FA4
    flash attention implementation yourself::

        from torch.nn.attention import (
            activate_flash_attention_impl,
            restore_flash_attention_impl,
        )

        activate_flash_attention_impl("FA4")
        try:
            out = fp8_fa4_sdpa(q, k, v, is_causal=True)
        finally:
            restore_flash_attention_impl()

    The high-level ``apply_low_precision_attention`` API handles this
    automatically.
"""

from functools import partial

from torchao.prototype.attention.shared_utils.attention import (
    _fp8_rope_sdpa,
    _fp8_sdpa,
)

fp8_fa4_sdpa = partial(_fp8_sdpa, backend_name="FA4")
fp8_fa4_sdpa.__doc__ = _fp8_sdpa.__doc__
fp8_fa4_sdpa.__name__ = "fp8_fa4_sdpa"
fp8_fa4_sdpa.__qualname__ = "fp8_fa4_sdpa"

fp8_fa4_rope_sdpa = partial(_fp8_rope_sdpa, backend_name="FA4")
fp8_fa4_rope_sdpa.__doc__ = _fp8_rope_sdpa.__doc__
fp8_fa4_rope_sdpa.__name__ = "fp8_fa4_rope_sdpa"
fp8_fa4_rope_sdpa.__qualname__ = "fp8_fa4_rope_sdpa"
