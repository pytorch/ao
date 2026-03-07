# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch.nn as nn

from torchao.prototype.attention.shared_utils.fusion_utils import detect_causal_mask
from torchao.prototype.attention.shared_utils.wrapper import (
    _FP8FlashAttentionMonkeyPatchWrapper,
    _FP8FlashAttentionWrapper,
    _make_causal_aware_sdpa,
)


def setup_fp8_backend(
    model: nn.Module,
    flash_impl_name: str,
    fuse_rope_using_torch_compile: bool,
) -> nn.Module:
    if flash_impl_name == "FA3":
        from torchao.prototype.attention.fp8_fa3.attention import (
            fp8_fa3_sdpa as sdpa_fn,
        )
        from torchao.prototype.attention.fp8_fa3.fusion_pass import make_fp8_backend
    else:
        raise ValueError(f"Unknown flash_impl_name: {flash_impl_name}")

    if fuse_rope_using_torch_compile:
        wrapper = _FP8FlashAttentionWrapper(model, flash_impl_name=flash_impl_name)
        wrapper.compile_backend = make_fp8_backend(model, fuse_rope_using_torch_compile)
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
