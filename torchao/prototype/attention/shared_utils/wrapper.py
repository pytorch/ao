# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Wrapper classes for low-precision attention modules."""

from typing import Callable

import torch.nn as nn
import torch.nn.functional as F

from torchao.utils import torch_version_at_least

_TORCH_VERSION_AT_LEAST_2_11 = torch_version_at_least("2.11.0")

if _TORCH_VERSION_AT_LEAST_2_11:
    from torch.nn.attention import (
        activate_flash_attention_impl,
        restore_flash_attention_impl,
    )

_MIN_VERSION_ERROR = (
    "Low-precision attention requires PyTorch 2.11+. "
    "Please update your PyTorch version."
)


def _check_min_torch_version():
    if not _TORCH_VERSION_AT_LEAST_2_11:
        raise RuntimeError(_MIN_VERSION_ERROR)


class _LowPrecisionAttentionWrapper(nn.Module):
    """Base wrapper that registers ``_orig_mod`` and proxies attribute access."""

    def __init__(self, orig_mod: nn.Module):
        super().__init__()
        self._orig_mod = orig_mod

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)


class _FP8FlashAttentionMonkeyPatchWrapper(_LowPrecisionAttentionWrapper):
    """Monkey-patch wrapper that swaps SDPA with FP8 backend during forward."""

    def __init__(
        self,
        orig_mod: nn.Module,
        flash_impl_name: str,
        sdpa_patch_fn: Callable,
    ):
        _check_min_torch_version()
        super().__init__(orig_mod)
        object.__setattr__(self, "_flash_impl_name", flash_impl_name)
        object.__setattr__(self, "_sdpa_patch_fn", sdpa_patch_fn)

    def forward(self, *args, **kwargs):
        orig_mod = self._orig_mod
        flash_impl_name = object.__getattribute__(self, "_flash_impl_name")
        sdpa_patch_fn = object.__getattribute__(self, "_sdpa_patch_fn")

        activate_flash_attention_impl(flash_impl_name)
        try:
            original_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = sdpa_patch_fn
            try:
                return orig_mod(*args, **kwargs)
            finally:
                F.scaled_dot_product_attention = original_sdpa
        finally:
            restore_flash_attention_impl()


def _make_causal_aware_sdpa(fp8_sdpa_fn: Callable, strip_causal_mask: bool) -> Callable:
    """Wrap an FP8 SDPA function to strip materialized causal masks.

    When ``strip_causal_mask`` is True, strips any ``attn_mask`` and sets
    ``is_causal=True``. When False, passes the mask through unchanged.
    """
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
    else:
        return fp8_sdpa_fn
