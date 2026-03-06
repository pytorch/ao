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

from torchao.prototype.attention.shared_utils.setup import setup_fp8_backend
from torchao.prototype.attention.shared_utils.wrapper import (
    _LowPrecisionAttentionWrapper,
)
from torchao.prototype.attention.utils import _is_fa3_available, _is_hopper
from torchao.utils import torch_version_at_least


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
    fuse_rope_using_torch_compile: bool = False,
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

    if backend is None:
        backend = _get_available_backend()
    else:
        _check_backend_available(backend)

    if backend == AttentionBackend.FP8_FA3:
        return setup_fp8_backend(model, "FA3", fuse_rope_using_torch_compile)

    raise ValueError(f"Unknown backend: {backend}")
