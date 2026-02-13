# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
User-facing API for low-precision attention.
"""

from typing import Optional

import torch.nn as nn

from torchao.prototype.attention.config import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
)
from torchao.prototype.attention.utils import (
    _check_backend_available,
    _check_config_supported,
    _get_available_backend,
)


def apply_low_precision_attention(
    model: nn.Module,
    config: Optional[LowPrecisionAttentionConfig] = None,
) -> nn.Module:
    """
    Apply low-precision attention to a model.

    Args:
        model: The model to modify (modified in-place).
        config: Configuration for low-precision attention.
            If None, uses default config (auto backend, basic FP8 quantization).

    Returns:
        The modified model.

    Example::

        from torchao.prototype.attention import (
            LowPrecisionAttentionConfig,
            apply_low_precision_attention,
        )

        model = MyTransformer()

        # Simple usage - auto-selects backend and uses basic FP8
        apply_low_precision_attention(model)

        # With options
        config = LowPrecisionAttentionConfig(fuse_rope=True)
        apply_low_precision_attention(model, config)

        model = torch.compile(model)
        output = model(inputs)
    """
    if config is None:
        config = LowPrecisionAttentionConfig()

    if config.backend is None:
        backend = _get_available_backend()
    else:
        backend = config.backend
        _check_backend_available(backend)

    _check_config_supported(config, backend)

    return _wrap_model(model, config, backend)


def _wrap_model(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
    backend: AttentionBackend,
) -> nn.Module:
    """Dispatch to appropriate backend implementation."""
    if backend == AttentionBackend.FP8_FA3:
        if config.use_hadamard == "qkv":
            raise NotImplementedError(
                "FP8 attention with Hadamard on QKV is not yet implemented."
            )
        elif config.use_hadamard == "v":
            raise NotImplementedError(
                "FP8 attention with Hadamard on V is not yet implemented."
            )
        elif config.fuse_rope:
            raise NotImplementedError(
                "FP8 attention with fused RoPE is not yet implemented."
            )
        else:
            from torchao.prototype.attention.fp8_fa3.wrappers import (
                _wrap_model_with_fp8_fa3_attention,
            )

            return _wrap_model_with_fp8_fa3_attention(model, config)

    raise ValueError(f"Unknown backend: {backend}")
