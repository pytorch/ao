# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper classes for low-precision attention modules.

_LowPrecisionAttentionWrapper          (base: orig_mod proxy, isinstance target)
├── _FP8FlashAttentionCompiledWrapper   (compile path: @dynamo.disable, flash activation)
└── _FP8FlashAttentionMonkeyPatchWrapper (monkey-patch path: SDPA swap, flash activation)
"""

from typing import Callable

import torch
import torch._dynamo
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


class _FP8FlashAttentionCompiledWrapper(_LowPrecisionAttentionWrapper):
    """Compile path wrapper (``fuse_rope_using_torch_compile=True``).

    @torch._dynamo.disable prevents an outer torch.compile from re-tracing
    the internally-compiled graph.
    """

    def __init__(
        self,
        compiled_mod: nn.Module,
        orig_mod: nn.Module,
        flash_impl_name: str,
    ):
        super().__init__(orig_mod)
        # Stored outside _modules to avoid double-counting parameters.
        object.__setattr__(self, "_compiled_mod", compiled_mod)
        object.__setattr__(self, "_flash_impl_name", flash_impl_name)

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        compiled_mod = object.__getattribute__(self, "_compiled_mod")
        flash_impl_name = object.__getattribute__(self, "_flash_impl_name")

        activate_flash_attention_impl(flash_impl_name)
        try:
            return compiled_mod(*args, **kwargs)
        finally:
            restore_flash_attention_impl()


class _FP8FlashAttentionMonkeyPatchWrapper(_LowPrecisionAttentionWrapper):
    """Monkey-patch path wrapper (``fuse_rope_using_torch_compile=False``).

    Replaces ``F.scaled_dot_product_attention`` with the FP8 backend
    for the duration of each forward call.
    """

    def __init__(
        self,
        orig_mod: nn.Module,
        flash_impl_name: str,
        sdpa_patch_fn: Callable,
    ):
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

    When ``strip_causal_mask=True``, unconditionally strips any attn_mask
    and sets ``is_causal=True``. This is zero-cost at runtime.
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
