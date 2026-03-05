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


class _FP8FlashAttentionWrapper(_LowPrecisionAttentionWrapper):
    """Compile path wrapper. Activates the flash impl around the module forward."""

    def __init__(self, orig_mod: nn.Module, flash_impl_name: str):
        super().__init__(orig_mod)
        self._flash_impl_name = flash_impl_name

    def forward(self, *args, **kwargs):
        activate_flash_attention_impl(self._flash_impl_name)
        try:
            return self._orig_mod(*args, **kwargs)
        finally:
            restore_flash_attention_impl()


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


def _make_causal_aware_sdpa(fp8_sdpa_fn: Callable, strip_causal_mask: bool) -> Callable:
    """Wrap an FP8 SDPA function to strip materialized causal masks."""
    if strip_causal_mask:

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
            if attn_mask is not None:
                attn_mask = None
                is_causal = True
            return fp8_sdpa_fn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        return _patched
    return fp8_sdpa_fn
