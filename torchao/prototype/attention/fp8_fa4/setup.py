# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 FA4 backend setup.

Thin wrapper around the shared ``setup_fp8_backend``, binding the FA4
attention function and FA4 compile helper.
"""

import torch.nn as nn

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.setup import setup_fp8_backend


def setup_fp8_fa4(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
) -> nn.Module:
    """Set up FP8 FA4 attention on *model* and wrap it."""
    from torchao.prototype.attention.fp8_fa4.attention import fp8_fa4_sdpa
    from torchao.prototype.attention.fp8_fa4.fusion_pass import (
        compile_with_fp8_fusion,
    )

    return setup_fp8_backend(
        model,
        config,
        flash_impl_name="FA4",
        sdpa_fn=fp8_fa4_sdpa,
        compile_fn=compile_with_fp8_fusion,
    )
