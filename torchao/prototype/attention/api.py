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

    Depending on the configuration, the model is either:
    - **Monkey-patch path** (``fuse_rope=False``, default): wraps the model
      so that ``F.scaled_dot_product_attention`` is replaced with the FP8
      backend at call time.  No ``torch.compile`` is needed.
    - **Compile path** (``fuse_rope=True``): compiles the model with a
      custom Inductor backend that fuses RoPE + FP8 quantization + SDPA
      into optimized kernels.

    In both cases, the returned wrapper manages flash attention activation
    internally — callers do **not** need to call
    ``activate_flash_attention_impl`` / ``restore_flash_attention_impl``.

    The returned wrapper creates a graph-break boundary, so if the
    caller later applies ``torch.compile`` to a parent model, the inner
    module is preserved.

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
