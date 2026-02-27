# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper classes for low-precision attention modules.

Class hierarchy::

    _LowPrecisionAttentionWrapper          (base: orig_mod proxy, isinstance target)
    ├── _FP8FlashAttentionCompiledWrapper   (compile path: @dynamo.disable, flash activation)
    └── _FP8FlashAttentionMonkeyPatchWrapper (monkey-patch path: SDPA swap, flash activation)

The base class is intentionally minimal — it handles ``_orig_mod``
registration and attribute proxying.  Backend-specific concerns (flash
activation, SDPA monkey-patching, dynamo guards) live in subclasses.
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

# ============================================================================
# Base wrapper
# ============================================================================


class _LowPrecisionAttentionWrapper(nn.Module):
    """Base wrapper for low-precision attention modules.

    Registers ``_orig_mod`` as a submodule so that ``to()``, ``cuda()``,
    ``eval()``, ``parameters()``, etc. propagate correctly through the
    standard ``nn.Module`` machinery.  Proxies attribute access to the
    original module so model-specific attributes (e.g., ``config``)
    remain accessible.

    This class is the ``isinstance`` check target used by
    ``apply_low_precision_attention`` to guard against double-wrapping.
    Subclasses implement ``forward`` with backend-specific logic.
    """

    def __init__(self, orig_mod: nn.Module):
        super().__init__()
        self._orig_mod = orig_mod

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ checks _parameters, _buffers, _modules.
        # If the attribute is not found there, proxy to the original module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)


# ============================================================================
# FP8 Flash Attention wrappers
# ============================================================================


class _FP8FlashAttentionCompiledWrapper(_LowPrecisionAttentionWrapper):
    """Wrapper for the compile path (``fuse_rope=True``).

    The inner module has already been compiled with a custom Inductor
    backend (the FP8 fusion pass).  ``@torch._dynamo.disable`` on
    ``forward`` prevents an outer ``torch.compile`` from re-tracing the
    internally-compiled graph.

    Flash attention is activated/restored around each forward call.
    """

    def __init__(
        self,
        compiled_mod: nn.Module,
        orig_mod: nn.Module,
        flash_impl_name: str,
    ):
        super().__init__(orig_mod)
        # Stored outside _modules to avoid double-counting parameters
        # (the compiled module wraps the same _orig_mod).
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
    """Wrapper for the monkey-patch path (``fuse_rope=False``).

    Replaces ``F.scaled_dot_product_attention`` with the FP8 backend
    function for the duration of each forward call.  The inner module
    is the original (uncompiled) model — the user may later call
    ``torch.compile`` on this wrapper or a parent module.

    Flash attention is activated/restored around each forward call.
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


# ============================================================================
# Causal mask stripping helper
# ============================================================================


def _make_causal_aware_sdpa(fp8_sdpa_fn: Callable, strip_causal_mask: bool) -> Callable:
    """Wrap an FP8 SDPA function to strip materialized causal masks.

    HuggingFace models (e.g. LLaMA) pass a materialized lower-triangular
    boolean ``attn_mask`` to ``F.scaled_dot_product_attention`` with
    ``is_causal=False``.  The FP8 SDPA functions don't support
    ``attn_mask``.

    When ``strip_causal_mask`` is ``True`` (determined once at setup time
    by ``detect_causal_mask``), the wrapper unconditionally strips any
    ``attn_mask`` and sets ``is_causal=True``.  This is zero-cost at
    runtime — no per-call tensor inspection.

    When ``strip_causal_mask`` is ``False``, the mask is passed through
    unchanged (and the FP8 function will raise if it receives one).
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
