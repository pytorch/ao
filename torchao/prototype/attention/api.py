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

    Compiles the model with a custom backend that fuses
    RoPE + FP8 quantization + SDPA into optimized kernels.  The
    returned module is fully encapsulated: no global state is modified,
    and the caller does **not** need to call ``torch.compile`` or
    ``activate_flash_attention_impl`` separately.

    The returned wrapper creates a graph-break boundary, so if the
    caller later applies ``torch.compile`` to a parent model, the inner
    compilation (with the fusion pass) is preserved.

    Args:
        model: The model to apply low-precision attention to.
            Must not already be compiled with ``torch.compile``.
            For models that use KV caching (e.g., HuggingFace models
            with ``config.use_cache=True``), the caller should disable
            KV caching before calling this function.  The
            ``DynamicCache.update()`` calls insert ``torch.cat`` nodes
            that block the RoPE + SDPA fusion pass.
        config: Configuration for low-precision attention.
            If None, uses default config (auto backend selection).

    Returns:
        A wrapped module with low-precision attention compiled in.

    Raises:
        RuntimeError: If the model is already compiled or already
            wrapped.

    Example::

        from torchao.prototype.attention import apply_low_precision_attention

        model = MyTransformer()
        model = apply_low_precision_attention(model)
        output = model(inputs)  # First call triggers compilation
    """
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
        from torchao.prototype.attention.fp8_fa3.setup import (
            _LowPrecisionAttentionWrapper,
            setup_fp8_fa3,
        )

        # Guard: already wrapped.
        if isinstance(model, _LowPrecisionAttentionWrapper):
            raise RuntimeError(
                "apply_low_precision_attention has already been applied to this module."
            )

        return setup_fp8_fa3(model, config)

    raise ValueError(f"Unknown backend: {backend}")
