# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for low-precision attention.
"""

import importlib

import torch

from torchao.prototype.attention.config import AttentionBackend


def _is_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


def _is_fa3_available() -> bool:
    try:
        importlib.import_module("flash_attn_interface")
        return True
    except ModuleNotFoundError:
        return False


def _is_fa4_available() -> bool:
    try:
        importlib.import_module("flash_attn.cute.interface")
        return True
    except ModuleNotFoundError:
        return False


def _get_available_backend() -> AttentionBackend:
    """Get the best available backend for current hardware."""
    if not torch.cuda.is_available():
        raise RuntimeError("Low-precision attention requires CUDA.")

    capability = torch.cuda.get_device_capability()

    if _is_blackwell() and _is_fa4_available():
        return AttentionBackend.FP8_FA4
    if _is_hopper() and _is_fa3_available():
        return AttentionBackend.FP8_FA3
    if _is_hopper() and _is_fa4_available():
        return AttentionBackend.FP8_FA4

    raise RuntimeError(f"No compatible backend for SM{capability[0]}{capability[1]}.")


def _check_backend_available(backend: AttentionBackend) -> None:
    """Check if the specified backend is available on current hardware."""
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

    elif backend == AttentionBackend.FP8_FA4:
        if not (_is_hopper() or _is_blackwell()):
            raise RuntimeError(
                f"FP8_FA4 requires Hopper or Blackwell, got SM{capability[0]}{capability[1]}."
            )
        if not _is_fa4_available():
            raise RuntimeError(
                "FP8_FA4 requires the flash-attn package with FA4 support "
                "(flash_attn.cute.interface)."
            )

    else:
        raise ValueError(f"Unknown backend: {backend}")
