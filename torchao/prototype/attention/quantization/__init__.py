# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils._triton import has_triton

if has_triton():
    from torchao.prototype.attention.quantization.triton_hadamard_qkv_quantization import (
        triton_fp8_hadamard_sdpa_quantize as _fp8_hadamard_sdpa_quantize,
    )
    from torchao.prototype.attention.quantization.triton_hadamard_rope_qkv_quantization import (
        triton_fp8_hadamard_rope_sdpa_quantize as _fp8_hadamard_rope_sdpa_quantize,
    )
    from torchao.prototype.attention.quantization.triton_hadamard_utils import (
        inverse_hadamard_transform as _inverse_hadamard_transform,
    )
    from torchao.prototype.attention.quantization.triton_qkv_quantization import (
        triton_fp8_sdpa_quantize as _fp8_sdpa_quantize,
    )
    from torchao.prototype.attention.quantization.triton_rope_qkv_quantization import (
        triton_fp8_rope_sdpa_quantize as _fp8_rope_sdpa_quantize,
    )
else:

    def _triton_required(*args, **kwargs):
        raise ImportError(
            "torchao.prototype.attention quantization kernels require triton, "
            "which is not installed (it is only available on Linux with CUDA)."
        )

    _fp8_hadamard_sdpa_quantize = _triton_required
    _fp8_hadamard_rope_sdpa_quantize = _triton_required
    _inverse_hadamard_transform = _triton_required
    _fp8_sdpa_quantize = _triton_required
    _fp8_rope_sdpa_quantize = _triton_required

__all__ = [
    "_fp8_sdpa_quantize",
    "_fp8_rope_sdpa_quantize",
    "_fp8_hadamard_sdpa_quantize",
    "_fp8_hadamard_rope_sdpa_quantize",
    "_inverse_hadamard_transform",
]
