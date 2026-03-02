# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration for low-precision attention.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


class AttentionBackend(str, Enum):
    """
    Backend kernel for computing attention.

    Different backends have different hardware requirements and capabilities.
    """

    FP8_FA3 = "FP8_FA3"
    """FlashAttention 3 via PyTorch core. Requires SM90+ (Hopper)."""

    FP8_FA4 = "FP8_FA4"
    """FlashAttention 4 via PyTorch core. Requires SM90+ (Hopper) or SM100+ (Blackwell)."""


@dataclass
class LowPrecisionAttentionConfig:
    """
    Configuration for low-precision attention inference.

    Note: Some low-precision attention only supports inference (forward pass).
    Backward pass may not be supported by the underlying backends.

    Args:
        backend: Attention backend to use. If None (default), automatically
            selected based on hardware capabilities.
        hadamard_mode: Apply Hadamard transform. Options:
            - None: No Hadamard transform (default)
            - "v": Apply Hadamard to V only
            - "qkv": Apply Hadamard to Q, K, and V
        fuse_rope_using_torch_compile: If True, fuse RoPE + quantization + SDPA into optimized
            kernels.  This uses ``torch.compile`` internally — see
            ``shared_utils/custom_ops.py`` for details.
            If False (default), attention is replaced at call time without
            compilation.
    """

    backend: Optional[AttentionBackend] = None
    hadamard_mode: Optional[Literal["v", "qkv"]] = None
    fuse_rope_using_torch_compile: bool = False

    def __post_init__(self):
        # Validate hadamard_mode value
        if self.hadamard_mode is not None and self.hadamard_mode not in ("v", "qkv"):
            raise ValueError(
                f"hadamard_mode must be None, 'v', or 'qkv', got {self.hadamard_mode!r}"
            )
