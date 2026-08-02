# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 FA3 backend setup."""

import torch.nn as nn

from torchao.prototype.attention.shared_utils.setup import setup_fp8_backend


def setup_fp8_fa3(
    model: nn.Module,
    hadamard: str = "NONE",
) -> nn.Module:
    """Set up FP8 FA3 attention on *model* and wrap it."""
    return setup_fp8_backend(
        model,
        flash_impl_name="FA3",
        hadamard=hadamard,
    )
