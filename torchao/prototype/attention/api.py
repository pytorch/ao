# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""User-facing API for low-precision attention."""

from typing import Optional

import torch
import torch._dynamo
import torch.nn as nn

from torchao.prototype.attention.config import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
)
from torchao.prototype.attention.shared_utils.setup import setup_fp8_backend
from torchao.prototype.attention.shared_utils.wrapper import (
    _LowPrecisionAttentionWrapper,
)
from torchao.prototype.attention.utils import (
    _check_backend_available,
    _get_available_backend,
)
from torchao.utils import torch_version_at_least


def apply_low_precision_attention(
    model: nn.Module,
    config: Optional[LowPrecisionAttentionConfig] = None,
) -> nn.Module:
    """Apply low-precision attention to a model.

    Must be called before ``torch.compile``. KV caching should be
    disabled before calling (e.g., ``config.use_cache = False`` for
    HuggingFace models).
    """
    if not torch_version_at_least("2.11.0"):
        raise RuntimeError("Low-precision attention requires PyTorch 2.11+.")
    if isinstance(model, _LowPrecisionAttentionWrapper):
        raise RuntimeError(
            "apply_low_precision_attention has already been applied to this module."
        )
    if isinstance(model, torch._dynamo.OptimizedModule):
        raise RuntimeError(
            "apply_low_precision_attention must be called before torch.compile."
        )

    if config is None:
        config = LowPrecisionAttentionConfig()

    if config.backend is None:
        backend = _get_available_backend()
    else:
        backend = config.backend
        _check_backend_available(backend)

    if backend == AttentionBackend.FP8_FA3:
        from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa
        from torchao.prototype.attention.fp8_fa3.fusion_pass import (
            compile_with_fp8_fusion,
        )

        return setup_fp8_backend(
            model,
            config,
            flash_impl_name="FA3",
            sdpa_fn=fp8_fa3_sdpa,
            compile_fn=compile_with_fp8_fusion,
        )

    raise ValueError(f"Unknown backend: {backend}")
