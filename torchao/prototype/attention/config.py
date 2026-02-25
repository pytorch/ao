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

    FP8_FA3 = "fa3"
    """FlashAttention 3 via PyTorch core. Requires SM90+ (Hopper)."""


@dataclass
class LowPrecisionAttentionConfig:
    """
    Configuration for low-precision attention inference.

    Note: Some low-precision attention only supports inference (forward pass).
    Backward pass may not be supported by the underlying backends.

    Args:
        backend: Attention backend to use. If None (default), automatically
            selected based on hardware capabilities.
        use_hadamard: Apply Hadamard transform. Options:
            - None: No Hadamard transform (default)
            - "v": Apply Hadamard to V only
            - "qkv": Apply Hadamard to Q, K, and V
        fuse_rope: If True (default), the compilation pass fuses RoPE +
            quantization + SDPA into a single kernel.  If False,
            only SDPA is replaced with its low-precision equivalent;
            RoPE and transpose ops remain in the graph and are compiled
            normally by Inductor.
    """

    backend: Optional[AttentionBackend] = None
    use_hadamard: Optional[Literal["v", "qkv"]] = None
    fuse_rope: bool = True

    def __post_init__(self):
        # Validate use_hadamard value
        if self.use_hadamard is not None and self.use_hadamard not in ("v", "qkv"):
            raise ValueError(
                f"use_hadamard must be None, 'v', or 'qkv', got {self.use_hadamard!r}"
            )
