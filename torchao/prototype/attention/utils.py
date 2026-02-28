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
    """
    Check if the current CUDA device is Hopper (SM 9.x).
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _is_fa3_available() -> bool:
    """
    Check if the flash attention 3 library (flash_attn_interface) is installed.
    """
    try:
        importlib.import_module("flash_attn_interface")
        return True
    except ModuleNotFoundError:
        return False


def _get_available_backend() -> AttentionBackend:
    """
    Get the best available backend for current hardware.

    Returns:
        The best available backend.

    Raises:
        RuntimeError: If no compatible backend is available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Low-precision attention requires CUDA.")

    capability = torch.cuda.get_device_capability()

    # FA3 requires exactly Hopper (SM 9.x) and flash_attn_interface
    if _is_hopper() and _is_fa3_available():
        return AttentionBackend.FP8_FA3

    raise RuntimeError(f"No compatible backend for SM{capability[0]}{capability[1]}.")


def _check_backend_available(backend: AttentionBackend) -> None:
    """
    Check if the specified backend is available on current hardware.

    Args:
        backend: The backend to check.

    Raises:
        RuntimeError: If the backend is not available.
    """
    if backend == AttentionBackend.FP8_FA3:
        if not torch.cuda.is_available():
            raise RuntimeError("FP8_FA3 backend requires CUDA.")

        if not _is_hopper():
            capability = torch.cuda.get_device_capability()
            raise RuntimeError(
                f"FP8_FA3 backend requires Hopper (SM 9.x). "
                f"Current device: SM{capability[0]}{capability[1]}. "
            )

        if not _is_fa3_available():
            raise RuntimeError(
                "FP8_FA3 backend requires the flash-attn package with FA3 support. "
            )

    else:
        raise ValueError(f"Unknown backend: {backend}")
