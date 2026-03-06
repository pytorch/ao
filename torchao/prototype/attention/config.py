# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class AttentionBackend(str, Enum):
    """Backend kernel for computing attention."""

    FP8_FA3 = "FP8_FA3"  # Requires SM90+ (Hopper)
    FP8_FA4 = "FP8_FA4"  # Requires SM90+ (Hopper) or SM100+ (Blackwell)
