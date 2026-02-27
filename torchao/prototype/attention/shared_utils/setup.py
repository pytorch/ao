# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared backend setup logic for low-precision attention.

Provides ``setup_fp8_backend``, the parameterized core of every
backend-specific setup function (e.g., ``setup_fp8_fa3``).

Routes between two paths depending on ``config.fuse_rope``:
- **Monkey-patch path** (default): wraps the model so that
  ``F.scaled_dot_product_attention`` is replaced with the FP8
  backend at call time.  No ``torch.compile`` needed.
- **Compile path** (``fuse_rope=True``): compiles the model with the
  RoPE + FP8 fusion pass via Inductor.
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
    """Set up FP8 attention on *model* and wrap it.

    Args:
        model: The model to wrap.
        config: Low-precision attention configuration.
        flash_impl_name: Flash implementation name (e.g. ``"FA3"``).
        sdpa_fn: The backend-specific FP8 SDPA function (e.g.
            ``fp8_fa3_sdpa``).  Used for the monkey-patch path.
        compile_fn: A callable ``(model, config) -> compiled_module`` that
            compiles the model with the backend-specific fusion pass.
            Used for the compile path.

    Returns:
        A wrapped module with low-precision FP8 attention applied.
    """
    if config.use_hadamard == "qkv":
        raise NotImplementedError(
            "FP8 attention with Hadamard on QKV is not yet implemented."
        )
    elif config.use_hadamard == "v":
        raise NotImplementedError(
            "FP8 attention with Hadamard on V is not yet implemented."
        )

    if config.fuse_rope:
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
