# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for low-precision attention.
"""

import torch

from torchao.attention.config import AttentionBackend, LowPrecisionAttentionConfig


def _is_hopper() -> bool:
    """
    Check if the current CUDA device is Hopper (SM 9.x).
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


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

    # FA3 requires exactly Hopper (SM 9.x)
    if _is_hopper():
        return AttentionBackend.FP8_FA3
    else:
        raise RuntimeError(
            f"No compatible backend for SM{capability[0]}{capability[1]}."
        )


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
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _check_config_supported(
    config: LowPrecisionAttentionConfig, backend: AttentionBackend
) -> None:
    """
    Check if the config options are supported by the backend.

    Args:
        config: The configuration to check.
        backend: The backend to check against.

    Raises:
        ValueError: If an option is not supported by the backend.
    """
    if config.fuse_rope:
        if backend not in (AttentionBackend.FP8_FA3):
            raise ValueError("fuse_rope requires FP8_FA3 backend")
