# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.prototype.moe_training.blockwise_fp8.grouped_mm import (
    fp8_blockwise_grouped_mm,
)

__all__ = [
    "fp8_blockwise_grouped_mm",
]
