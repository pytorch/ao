# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
User-facing API for low-precision attention.

This module is the backend-agnostic entry point.  It validates inputs,
resolves the backend, and dispatches to the appropriate backend-specific
setup function (e.g., ``fp8_fa3.setup.setup_fp8_fa3``).
"""

from typing import Optional

import torch
import torch._dynamo
import torch.nn as nn

from torchao.prototype.attention.config import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
)
from torchao.prototype.attention.shared_utils.wrapper import (
    _LowPrecisionAttentionWrapper,
)
from torchao.prototype.attention.utils import (
    _check_backend_available,
    _get_available_backend,
)


def apply_low_precision_attention(
    model: nn.Module,
    config: Optional[LowPrecisionAttentionConfig] = None,
) -> nn.Module:
    """
    Apply low-precision attention to a model.

    Resolves the requested backend and delegates to the backend-specific
    setup function.  The returned wrapper manages attention backend
    activation internally — callers do **not** need to call
    ``activate_flash_attention_impl`` / ``restore_flash_attention_impl``.

    See ``LowPrecisionAttentionConfig`` for the available configuration
    options and their effects (e.g., ``fuse_rope_using_torch_compile``).

    Args:
        model: The model to apply low-precision attention to.
            Must not already be compiled with ``torch.compile``.
            KV caching should be disabled before calling this function
            (e.g., ``config.use_cache = False`` for HuggingFace models).
        config: Configuration for low-precision attention.
            If None, uses default config (auto backend selection).

    Returns:
        A wrapped module with low-precision attention applied.

    Raises:
        RuntimeError: If the model is already compiled or already
            wrapped.

    Example::

        from torchao.prototype.attention import apply_low_precision_attention

        model = MyTransformer()
        model = apply_low_precision_attention(model)
        output = model(inputs)  # flash activation managed internally
    """
    # Guard: already wrapped.
    if isinstance(model, _LowPrecisionAttentionWrapper):
        raise RuntimeError(
            "apply_low_precision_attention has already been applied to this module."
        )

    # Guard: already compiled.
    if isinstance(model, torch._dynamo.OptimizedModule):
        raise RuntimeError(
            "The module is already compiled with torch.compile. "
            "apply_low_precision_attention must be called on the "
            "original (uncompiled) module, before torch.compile."
        )

    if config is None:
        config = LowPrecisionAttentionConfig()

    if config.backend is None:
        backend = _get_available_backend()
    else:
        backend = config.backend
        _check_backend_available(backend)

    if backend == AttentionBackend.FP8_FA3:
        from torchao.prototype.attention.fp8_fa3.setup import setup_fp8_fa3

        return setup_fp8_fa3(model, config)

    raise ValueError(f"Unknown backend: {backend}")
