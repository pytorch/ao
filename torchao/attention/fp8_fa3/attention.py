# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Internal: FP8 scaled dot-product attention using FA3 backend.
"""

from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
)

from torchao.attention.fp8_fa3.quantization import _fp8_sdpa_quantize


def _fp8_fa3_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """FP8 SDPA using FA3 backend. Quantizes Q, K, V to FP8 before attention."""
    if attn_mask is not None:
        raise ValueError("attn_mask not supported for FP8 FA3")
    if dropout_p != 0.0:
        raise ValueError(f"dropout_p must be 0.0 for FP8 FA3, got {dropout_p}")

    input_dtype = query.dtype

    q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = _fp8_sdpa_quantize(
        query, key, value
    )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = _scaled_dot_product_attention_quantized(
            q_fp8,
            k_fp8,
            v_fp8,
            is_causal=is_causal,
            scale=scale,
            q_descale=descale_q,
            k_descale=descale_k,
            v_descale=descale_v,
        )

    return out.to(input_dtype)
