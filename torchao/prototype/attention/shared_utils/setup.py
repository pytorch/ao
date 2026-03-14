# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch._inductor.config as inductor_config
import torch.nn as nn

from torchao.prototype.attention.shared_utils.fusion_utils import (
    detect_causal_mask,
    rope_sdpa_fusion_pass,
)
from torchao.prototype.attention.shared_utils.wrapper import (
    _FP8FlashAttentionMonkeyPatchWrapper,
    _make_causal_aware_sdpa,
)


def setup_fp8_backend(
    model: nn.Module,
    flash_impl_name: str,
) -> nn.Module:
    if flash_impl_name == "FA3":
        from torchao.prototype.attention.fp8_fa3.attention import _ops
    else:
        raise ValueError(f"Unknown flash_impl_name: {flash_impl_name}")

    strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)

    inductor_config.pre_grad_custom_pass = partial(
        rope_sdpa_fusion_pass,
        rope_sdpa_op=_ops.rope_sdpa_op,
        fp8_sdpa_op=_ops.fp8_sdpa_op,
        backend_name=flash_impl_name,
    )

    return _FP8FlashAttentionMonkeyPatchWrapper(
        model,
        flash_impl_name=flash_impl_name,
        sdpa_patch_fn=_make_causal_aware_sdpa(_ops.fp8_sdpa_op, strip_causal_mask),
    )
