# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torchao.prototype.attention.fp8_fa4.attention import (
    fp8_fa4_rope_sdpa,
    fp8_fa4_sdpa,
)
from torchao.prototype.attention.shared_utils.custom_ops import (
    make_backend_fn,
    register_fp8_attention_ops,
)

_ops = register_fp8_attention_ops(
    backend_name="fa4",
    rope_sdpa_fn=fp8_fa4_rope_sdpa,
    sdpa_fn=fp8_fa4_sdpa,
)

make_fp8_backend = make_backend_fn(_ops, backend_name="FA4", flash_impl_name="FA4")
