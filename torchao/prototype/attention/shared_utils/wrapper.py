# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)


class _LowPrecisionAttentionWrapper(nn.Module):
    """Base wrapper. Proxies attribute access to the original module."""

    def __init__(self, orig_mod: nn.Module):
        super().__init__()
        self._orig_mod = orig_mod

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)


class _FP8FlashAttentionMonkeyPatchWrapper(_LowPrecisionAttentionWrapper):
    """Monkey-patch path wrapper. Replaces ``F.scaled_dot_product_attention``
    with the FP8 backend for the duration of each forward call.
    """

    def __init__(
        self, orig_mod: nn.Module, flash_impl_name: str, sdpa_patch_fn: Callable
    ):
        super().__init__(orig_mod)
        self._flash_impl_name = flash_impl_name
        self._sdpa_patch_fn = sdpa_patch_fn

    def forward(self, *args, **kwargs):
        activate_flash_attention_impl(self._flash_impl_name)
        try:
            original_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = self._sdpa_patch_fn
            try:
                return self._orig_mod(*args, **kwargs)
            finally:
                F.scaled_dot_product_attention = original_sdpa
        finally:
            restore_flash_attention_impl()


def _make_causal_aware_sdpa(
    fp8_sdpa_custom_op, strip_causal_mask: bool, hadamard: str = "NONE"
) -> Callable:
    """Bridge F.sdpa signature to the FP8 SDPA custom op.

    Calls the custom op so torch.compile sees an opaque node in the FX graph.
    """

    def _patched(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if strip_causal_mask and attn_mask is not None:
            attn_mask = None
            is_causal = True
        if attn_mask is not None:
            raise ValueError("attn_mask not supported for FP8 attention")
        if dropout_p != 0.0:
            raise ValueError(
                f"dropout_p must be 0.0 for FP8 attention, got {dropout_p}"
            )
        return fp8_sdpa_custom_op(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale if scale is not None else 0.0,
            enable_gqa=enable_gqa,
            hadamard=hadamard,
        )

    return _patched
