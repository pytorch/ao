# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 quantization for attention inputs.
"""

from typing import Tuple

import torch


def _fp8_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize Q, K, V to FP8 with per-head scaling."""
    if q.dim() != 4:
        raise ValueError(f"Expected 4D tensor for q, got {q.dim()}D")
    if k.dim() != 4:
        raise ValueError(f"Expected 4D tensor for k, got {k.dim()}D")
    if v.dim() != 4:
        raise ValueError(f"Expected 4D tensor for v, got {v.dim()}D")
    if k.shape != v.shape:
        raise ValueError(f"K and V shape mismatch: {k.shape} vs {v.shape}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"Batch size mismatch: {q.shape[0]} vs {k.shape[0]}")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"Q head count ({q.shape[1]}) must be a multiple of K head count ({k.shape[1]})"
        )
    if q.shape[3] != k.shape[3]:
        raise ValueError(f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}")

    from torchao.prototype.attention.quantization.triton_qkv_quantization import (
        triton_fp8_sdpa_quantize,
    )

    return triton_fp8_sdpa_quantize(q, k, v)
