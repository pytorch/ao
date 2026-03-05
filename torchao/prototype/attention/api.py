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
    """Apply low-precision attention to a model.

    Resolves the backend and wraps the model so that attention backend
    activation is managed internally.
    """
    if isinstance(model, _LowPrecisionAttentionWrapper):
        raise RuntimeError("Model already has low-precision attention applied.")

    if isinstance(model, torch._dynamo.OptimizedModule):
        raise RuntimeError(
            "Module is already compiled. "
            "Call apply_low_precision_attention before torch.compile."
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
