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
    hadamard: str = "NONE",
) -> nn.Module:
    if flash_impl_name == "FA3":
        from torchao.prototype.attention.fp8_fa3.attention import _ops
    elif flash_impl_name == "FA4":
        from torchao.prototype.attention.fp8_fa4.attention import _ops
    elif flash_impl_name == "CUDNN":
        from torchao.prototype.attention.fp8_cudnn.attention import _ops
    else:
        raise ValueError(f"Unknown flash_impl_name: {flash_impl_name}")

    # cuDNN doesn't use flash attention impl for causal mask detection
    causal_flash_impl = flash_impl_name if flash_impl_name != "CUDNN" else None
    strip_causal_mask = detect_causal_mask(model, flash_impl_name=causal_flash_impl)

    inductor_config.pre_grad_custom_pass = partial(
        rope_sdpa_fusion_pass,
        rope_sdpa_op=_ops.rope_sdpa_op,
        fp8_sdpa_op=_ops.fp8_sdpa_op,
        backend_name=flash_impl_name,
    )

    # cuDNN doesn't need flash attention impl activation
    wrapper_flash_impl = flash_impl_name if flash_impl_name != "CUDNN" else None

    return _FP8FlashAttentionMonkeyPatchWrapper(
        model,
        flash_impl_name=wrapper_flash_impl,
        sdpa_patch_fn=_make_causal_aware_sdpa(
            _ops.fp8_sdpa_op, strip_causal_mask, hadamard=hadamard
        ),
    )
