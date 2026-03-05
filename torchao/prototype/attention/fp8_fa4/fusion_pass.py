# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FA4-specific FX graph fusion pass and compile helper.
"""

from torchao.prototype.attention.fp8_fa4.attention import (
    fp8_fa4_rope_sdpa,
    fp8_fa4_sdpa,
)
from torchao.prototype.attention.shared_utils.custom_ops import (
    make_compile_fn,
    make_fusion_pass,
    register_fp8_attention_ops,
)

_ops = register_fp8_attention_ops(
    backend_name="fa4",
    rope_sdpa_fn=fp8_fa4_rope_sdpa,
    sdpa_fn=fp8_fa4_sdpa,
)

rope_sdpa_fusion_pass = make_fusion_pass(_ops, backend_name="FA4", max_head_dim=256)
compile_with_fp8_fusion = make_compile_fn(rope_sdpa_fusion_pass, flash_impl_name="FA4")
