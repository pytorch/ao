# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared backend setup logic for low-precision attention.

Routes between two paths depending on ``config.fuse_rope_using_torch_compile``:
- Monkey-patch path (default): wraps the model so that
  ``F.scaled_dot_product_attention`` is replaced at call time.
- Compile path: compiles the model with the RoPE + FP8 fusion pass.
"""

from typing import Callable

import torch.nn as nn

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.wrapper import (
    _FP8FlashAttentionCompiledWrapper,
    _FP8FlashAttentionMonkeyPatchWrapper,
    _make_causal_aware_sdpa,
)


def setup_fp8_backend(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
    flash_impl_name: str,
    sdpa_fn: Callable,
    compile_fn: Callable,
) -> nn.Module:
    """Set up FP8 attention on *model* and wrap it."""
    if config.hadamard_mode == "qkv":
        raise NotImplementedError(
            "FP8 attention with Hadamard on QKV is not yet implemented."
        )
    elif config.hadamard_mode == "v":
        raise NotImplementedError(
            "FP8 attention with Hadamard on V is not yet implemented."
        )

    if config.fuse_rope_using_torch_compile:
        compiled = compile_fn(model, config)
        return _FP8FlashAttentionCompiledWrapper(
            compiled, model, flash_impl_name=flash_impl_name
        )
    else:
        from torchao.prototype.attention.shared_utils.fusion_utils import (
            detect_causal_mask,
        )

        strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)
        return _FP8FlashAttentionMonkeyPatchWrapper(
            model,
            flash_impl_name=flash_impl_name,
            sdpa_patch_fn=_make_causal_aware_sdpa(sdpa_fn, strip_causal_mask),
        )
