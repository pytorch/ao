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
    """Backend kernel for computing attention."""

    FP8_FA3 = "FP8_FA3"
    """FlashAttention 3 via PyTorch core. Requires SM90+ (Hopper)."""


@dataclass
class LowPrecisionAttentionConfig:
    """Configuration for low-precision attention inference."""

    backend: Optional[AttentionBackend] = None
    hadamard_mode: Optional[Literal["v", "qkv"]] = None
    fuse_rope_using_torch_compile: bool = False

    def __post_init__(self):
        if self.hadamard_mode is not None and self.hadamard_mode not in ("v", "qkv"):
            raise ValueError(
                f"hadamard_mode must be None, 'v', or 'qkv', got {self.hadamard_mode!r}"
            )
