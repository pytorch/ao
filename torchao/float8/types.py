# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Common types for float8 quantization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from torchao.quantization.granularity import PerRow, PerTensor

# Define FP8Granularity type alias to break circular import dependencies
FP8Granularity = Union["PerTensor", "PerRow"]
