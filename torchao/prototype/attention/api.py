# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""User-facing API for low-precision attention."""

from enum import Enum
from typing import Optional

import torch
import torch._dynamo
import torch.nn as nn

from torchao.prototype.attention.utils import _is_fa3_available, _is_hopper
from torchao.utils import torch_version_at_least

if torch_version_at_least("2.11.0"):
    from torchao.prototype.attention.shared_utils.setup import setup_fp8_backend
    from torchao.prototype.attention.shared_utils.wrapper import (
        _LowPrecisionAttentionWrapper,
    )
else:
    raise ImportError("Low-precision attention requires PyTorch 2.11+.")


class AttentionBackend(str, Enum):
    """Backend kernel for computing attention."""

    FP8_FA3 = "FP8_FA3"  # Requires SM90+ (Hopper)


def _get_available_backend() -> AttentionBackend:
    if not torch.cuda.is_available():
        raise RuntimeError("Low-precision attention requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if _is_hopper() and _is_fa3_available():
        return AttentionBackend.FP8_FA3
    raise RuntimeError(f"No compatible backend for SM{capability[0]}{capability[1]}.")


def _check_backend_available(backend: AttentionBackend) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{backend} backend requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if backend == AttentionBackend.FP8_FA3:
        if not _is_hopper():
            raise RuntimeError(
                f"FP8_FA3 requires Hopper (SM 9.x), got SM{capability[0]}{capability[1]}."
            )
        if not _is_fa3_available():
            raise RuntimeError(
                "FP8_FA3 requires the flash-attn package with FA3 support."
            )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def apply_low_precision_attention(
    model: nn.Module,
    backend: Optional[AttentionBackend] = None,
) -> nn.Module:
    """Apply low-precision attention to a model.

    Must be called before ``torch.compile``. KV caching should be
    disabled before calling (e.g., ``config.use_cache = False`` for
    HuggingFace models).

    This replaces ``F.scaled_dot_product_attention`` with an FP8 SDPA
    for eager execution and sets a global pre-grad pass so that
    ``torch.compile`` will automatically fuse RoPE where detected::

        model = apply_low_precision_attention(model)
        model = torch.compile(model)   # RoPE fusion happens automatically
    """
    if isinstance(model, _LowPrecisionAttentionWrapper):
        raise RuntimeError(
            "apply_low_precision_attention has already been applied to this module."
        )
    if isinstance(model, torch._dynamo.OptimizedModule):
        raise RuntimeError(
            "apply_low_precision_attention must be called before torch.compile."
        )

    if backend is None:
        backend = _get_available_backend()
    else:
        _check_backend_available(backend)

    if backend == AttentionBackend.FP8_FA3:
        return setup_fp8_backend(model, "FA3")

    raise ValueError(f"Unknown backend: {backend}")
