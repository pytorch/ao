# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Internal: Wrappers for applying FP8 FA3 attention to models.
"""

import warnings
from contextlib import contextmanager
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

_original_sdpa = F.scaled_dot_product_attention


@contextmanager
def _fp8_fa3_attention_context(
    config: Optional[LowPrecisionAttentionConfig] = None,
):
    """Context manager that enables FP8 FA3 attention within the context."""
    if config is None:
        config = LowPrecisionAttentionConfig()

    activate_flash_attention_impl("FA3")

    def fp8_wrapper(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if dropout_p != 0.0 or attn_mask is not None:
            warnings.warn(
                "Dropout and attention mask not supported for FP8 FA3. "
                "Falling back to regular SDPA."
            )
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        return fp8_fa3_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    F.scaled_dot_product_attention = fp8_wrapper

    try:
        yield
    finally:
        restore_flash_attention_impl()
        F.scaled_dot_product_attention = _original_sdpa


def _wrap_model_with_fp8_fa3_attention(
    model: nn.Module,
    config: Optional[LowPrecisionAttentionConfig] = None,
) -> nn.Module:
    """Wrap model so its forward pass uses FP8 FA3 attention."""
    if config is None:
        config = LowPrecisionAttentionConfig()

    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        with _fp8_fa3_attention_context(config):
            return original_forward(*args, **kwargs)

    model.forward = wrapped_forward
    return model
