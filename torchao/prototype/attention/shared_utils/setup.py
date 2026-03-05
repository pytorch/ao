# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Callable

import torch.nn as nn

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.fusion_utils import detect_causal_mask
from torchao.prototype.attention.shared_utils.wrapper import (
    _FP8FlashAttentionMonkeyPatchWrapper,
    _FP8FlashAttentionWrapper,
    _make_causal_aware_sdpa,
)


def setup_fp8_backend(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
    flash_impl_name: str,
    sdpa_fn: Callable,
    backend_fn: Callable,
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
        compile_backend = backend_fn(model, config)
        wrapper = _FP8FlashAttentionWrapper(model, flash_impl_name=flash_impl_name)
        wrapper.compile_backend = compile_backend
        warnings.warn(
            "fuse_rope_using_torch_compile=True: you must call "
            "torch.compile(model, backend=model.compile_backend) for the "
            "RoPE + FP8 fusion to take effect. Without it the model runs "
            "eagerly with no fusion. "
            "Note: this path uses torch._inductor.config.pre_grad_custom_pass, "
            "an unstable internal API that may change across PyTorch versions.",
            UserWarning,
            stacklevel=3,
        )
        return wrapper

    strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)
    return _FP8FlashAttentionMonkeyPatchWrapper(
        model,
        flash_impl_name=flash_impl_name,
        sdpa_patch_fn=_make_causal_aware_sdpa(sdpa_fn, strip_causal_mask),
    )
