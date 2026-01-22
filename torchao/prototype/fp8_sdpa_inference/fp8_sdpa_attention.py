# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 Scaled Dot Product Attention wrapper.
Uses PyTorch's FA3 backend for fp8 attention computation.
"""

from typing import Optional

import torch
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_quantization import (
    fp8_sdpa_quantize_func,
)


def fp8_sdpa_parallel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    num_chunks: Optional[int] = None,
) -> torch.Tensor:
    """
    Functional interface for FP8 SDPA with parallelized quantization.

    Splits quantization work across multiple thread blocks for better SM utilization.
    This version is faster when B*H is small relative to the number of SMs on the GPU.

    Args:
        query, key, value: Input tensors of shape (B, H, S, D)
        attn_mask: Not supported, must be None
        dropout_p: Must be 0.0
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for attention
        num_chunks: Number of chunks to split the S dimension into.
                    If None, automatically selects based on GPU SM count.
    """
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported for FP8 SDPA")
    if dropout_p != 0.0:
        raise ValueError("dropout_p must be 0.0 for FP8 SDPA")

    # Parallelized quantization of Q, K, V
    q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = fp8_sdpa_quantize_func(
        query, key, value, num_chunks=num_chunks
    )

    # Call PyTorch's fp8 SDPA
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

    return out
