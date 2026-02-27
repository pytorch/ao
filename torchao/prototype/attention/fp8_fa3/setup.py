# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 FA3 backend setup.

Thin wrapper around the shared ``setup_fp8_backend``, binding the FA3
attention function.
"""

import torch.nn as nn

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.setup import setup_fp8_backend


def setup_fp8_fa3(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
) -> nn.Module:
    """Set up FP8 FA3 attention on *model* and wrap it."""
    from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

    return setup_fp8_backend(
        model,
        config,
        flash_impl_name="FA3",
        sdpa_fn=fp8_fa3_sdpa,
    )
