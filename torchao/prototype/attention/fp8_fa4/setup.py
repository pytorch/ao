# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 FA4 backend setup: compilation, wrapping, and causal-mask pre-flight.

This module contains all FP8-FA4-specific logic for compiling a model with
the RoPE + FP8 fusion pass and wrapping it for inference.  The public entry
point is ``setup_fp8_fa4``, called by the backend-agnostic dispatcher in
``torchao.prototype.attention.api``.
"""

from functools import partial

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn

from torchao.prototype.attention.api import _LowPrecisionAttentionWrapper
from torchao.prototype.attention.config import LowPrecisionAttentionConfig


def setup_fp8_fa4(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
) -> nn.Module:
    """Compile *model* with the RoPE + FP8 fusion pass (FA4) and wrap it."""
    if config.use_hadamard == "qkv":
        raise NotImplementedError(
            "FP8 attention with Hadamard on QKV is not yet implemented."
        )
    elif config.use_hadamard == "v":
        raise NotImplementedError(
            "FP8 attention with Hadamard on V is not yet implemented."
        )

    from torch._inductor.compile_fx import compile_fx

    from torchao.prototype.attention.fp8_fa4.fusion_pass import (
        rope_sdpa_fusion_pass,
    )
    from torchao.prototype.attention.fusion_utils import detect_causal_mask

    strip_causal_mask = detect_causal_mask(model)

    pass_fn = partial(
        rope_sdpa_fusion_pass,
        fuse_rope=config.fuse_rope,
        strip_causal_mask=strip_causal_mask,
    )

    def fp8_attention_backend(gm, example_inputs):
        """Custom Inductor backend that applies the RoPE + FP8 fusion pass."""
        old_pass = inductor_config.pre_grad_custom_pass
        inductor_config.pre_grad_custom_pass = pass_fn
        try:
            return compile_fx(gm, example_inputs)
        finally:
            inductor_config.pre_grad_custom_pass = old_pass

    # Clear stale Dynamo caches to ensure fresh compilation.
    torch._dynamo.reset()

    # Compile with our custom backend (fusion pass is baked in).
    compiled = torch.compile(model, backend=fp8_attention_backend)

    return _LowPrecisionAttentionWrapper(compiled, model, flash_impl_name="FA4")
