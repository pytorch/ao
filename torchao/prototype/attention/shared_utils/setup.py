# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared backend setup logic for low-precision attention.

Provides ``setup_fp8_backend``, the parameterized core of every
backend-specific setup function (e.g., ``setup_fp8_fa3``).

Currently supports the monkey-patch path (``fuse_rope=False``), which wraps
the model so that ``F.scaled_dot_product_attention`` is replaced with
the FP8 backend at call time.  No ``torch.compile`` needed.
"""

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.wrapper import (
    _FP8FlashAttentionMonkeyPatchWrapper,
    _make_causal_aware_sdpa,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Causal mask detection (pre-flight probe)
# ============================================================================


def _is_lower_triangular_bool_mask(mask: torch.Tensor) -> bool:
    """Check if a real tensor is a bool, square lower-triangular (causal) mask.

    Compares the mask against ``torch.tril(torch.ones(...))``.  Returns
    ``True`` only when the mask is boolean, the last two dimensions are
    square, and every element matches the canonical causal pattern.
    """
    if mask.dtype != torch.bool:
        return False

    if mask.ndim < 2:
        return False

    q_len, kv_len = mask.shape[-2], mask.shape[-1]
    if q_len != kv_len:
        return False

    # Build reference causal mask on the same device / shape and compare.
    ref = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=mask.device))
    # Broadcast: mask may be [B, 1, Q, KV] or [B, H, Q, KV]; ref is [Q, KV].
    return torch.equal(mask.broadcast_to(mask.shape), ref.expand_as(mask))


def detect_causal_mask(
    model: nn.Module,
    sample_input_ids=None,
    flash_impl_name: str | None = None,
) -> bool:
    """Run one forward pass to detect whether the model uses causal masks.

    This pre-flight check monkey-patches ``F.scaled_dot_product_attention``
    to inspect every ``attn_mask`` argument.  It returns ``True`` when
    either of the following holds for **every** SDPA call during the
    forward pass:

    1. The call carries a materialized lower-triangular boolean causal
       mask (i.e. safe to strip and replace with ``is_causal=True``), or
    2. The call already uses ``is_causal=True`` with no mask tensor.

    Args:
        model: The model to probe.  Must expose ``model.config.vocab_size``
            for automatic dummy-input generation.
        sample_input_ids: Optional ``[B, S]`` int tensor of input IDs.
            If ``None``, a ``[1, 16]`` tensor of random token IDs is
            created automatically.
        flash_impl_name: Flash attention implementation to activate during
            the probe forward pass (e.g. ``"FA3"``).  Required
            on hardware where the default SDPA backend may not support the
            model.

    Returns:
        ``True`` if every SDPA call used causal attention (safe to strip
        masks that appear during compilation), ``False`` otherwise
        (conservatively disables mask stripping).
    """
    # Resolve device from model parameters.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return False

    # Build dummy input_ids if not provided.
    if sample_input_ids is None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is None:
            return False
        sample_input_ids = torch.randint(0, vocab_size, (1, 16), device=device)

    all_causal: list[bool] = []
    saw_any_sdpa = False

    original_sdpa = F.scaled_dot_product_attention

    def _hook(*args, **kwargs):
        nonlocal saw_any_sdpa
        saw_any_sdpa = True
        # Extract attn_mask (positional arg 3 or kwarg).
        attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask", None)
        is_causal = kwargs.get("is_causal", False) if len(args) <= 5 else args[5]

        if attn_mask is not None and not is_causal:
            all_causal.append(_is_lower_triangular_bool_mask(attn_mask))
        elif attn_mask is None and is_causal:
            # The framework skipped mask materialization and used
            # is_causal=True directly.  This is equivalent to a causal
            # mask and is safe to strip during compilation.
            all_causal.append(True)

        return original_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _hook
    if flash_impl_name is not None:
        activate_flash_attention_impl(flash_impl_name)
    try:
        with torch.no_grad():
            model(sample_input_ids)
    except Exception:
        logger.debug("detect_causal_mask: forward pass failed", exc_info=True)
        return False
    finally:
        F.scaled_dot_product_attention = original_sdpa
        if flash_impl_name is not None:
            restore_flash_attention_impl()

    if not saw_any_sdpa:
        # No SDPA calls observed at all.
        return False

    return all(all_causal)


# ============================================================================
# Backend setup
# ============================================================================


def setup_fp8_backend(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
    flash_impl_name: str,
    sdpa_fn: Callable,
) -> nn.Module:
    """Set up FP8 attention on *model* and wrap it.

    Args:
        model: The model to wrap.
        config: Low-precision attention configuration.
        flash_impl_name: Flash implementation name (e.g. ``"FA3"``).
        sdpa_fn: The backend-specific FP8 SDPA function (e.g.
            ``fp8_fa3_sdpa``).  Used for the monkey-patch path.

    Returns:
        A wrapped module with low-precision FP8 attention applied.
    """
    if config.use_hadamard == "qkv":
        raise NotImplementedError(
            "FP8 attention with Hadamard on QKV is not yet implemented."
        )
    elif config.use_hadamard == "v":
        raise NotImplementedError(
            "FP8 attention with Hadamard on V is not yet implemented."
        )

    if config.fuse_rope:
        raise NotImplementedError(
            "RoPE fusion (fuse_rope=True) is not yet implemented. "
            "Use fuse_rope=False (default) for the monkey-patch path."
        )

    strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)
    return _FP8FlashAttentionMonkeyPatchWrapper(
        model,
        flash_impl_name=flash_impl_name,
        sdpa_patch_fn=_make_causal_aware_sdpa(sdpa_fn, strip_causal_mask),
    )
