# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchao.prototype.attention.shared_utils.wrapper import (
    _FP8FlashAttentionMonkeyPatchWrapper,
    _make_causal_aware_sdpa,
)


def setup_fp8_backend(
    model: nn.Module,
    flash_impl_name: str,
    fuse_rope_using_torch_compile: bool,
) -> nn.Module:
    if fuse_rope_using_torch_compile:
        raise NotImplementedError(
            "fuse_rope_using_torch_compile requires the RoPE fusion path, "
            "which is not available in this version."
        )
    if flash_impl_name == "FA3":
        from torchao.prototype.attention.fp8_fa3.attention import (
            fp8_fa3_sdpa as sdpa_fn,
        )
    else:
        raise ValueError(f"Unknown flash_impl_name: {flash_impl_name}")

    return _FP8FlashAttentionMonkeyPatchWrapper(
        model,
        flash_impl_name=flash_impl_name,
        sdpa_patch_fn=_make_causal_aware_sdpa(sdpa_fn, strip_causal_mask=False),
    )
