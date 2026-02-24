# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 scaled dot-product attention using FA4 backend.
"""

from typing import Optional

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

_has_quantized_sdpa = False
try:
    from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
        _scaled_dot_product_attention_quantized,
    )

    _has_quantized_sdpa = True
except ImportError:
    pass

from torchao.prototype.attention.quantization import (
    _fp8_rope_sdpa_quantize,
    _fp8_sdpa_quantize,
)


def fp8_fa4_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """FP8 SDPA using FA4 backend. Quantizes Q, K, V to FP8 before attention."""
    if not _has_quantized_sdpa:
        raise RuntimeError(
            "fp8_fa4_sdpa requires a PyTorch version with "
            "torch.nn.attention.experimental._scaled_dot_product_attention_quantized. "
            "Please upgrade PyTorch."
        )
    if attn_mask is not None:
        raise ValueError("attn_mask not supported for FP8 FA4")
    if dropout_p != 0.0:
        raise ValueError(f"dropout_p must be 0.0 for FP8 FA4, got {dropout_p}")

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


def fp8_fa4_rope_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    rope_interleaved: bool = False,
) -> torch.Tensor:
    """FP8 SDPA with fused RoPE using FA4 backend.

    Applies RoPE to Q and K, quantizes Q, K, V to FP8, and runs attention
    in a single fused operation. Also transforms the layout from
    [B, S, H, D] to [B, H, S, D] as part of the fusion.

    Args:
        query: Query tensor of shape [B, S, H, D] in bf16/fp16.
        key: Key tensor of shape [B, S, H, D] in bf16/fp16.
        value: Value tensor of shape [B, S, H, D] in bf16/fp16.
        cos: Cosine frequencies for RoPE, shape [S, D].
        sin: Sine frequencies for RoPE, shape [S, D].
        attn_mask: Not supported, must be None.
        dropout_p: Not supported, must be 0.0.
        is_causal: Whether to apply causal masking.
        scale: Scaling factor for attention. If None, uses 1/sqrt(D).
        enable_gqa: Whether to enable grouped query attention.
        rope_interleaved: If True, uses interleaved RoPE (paired elements adjacent).
            If False, uses NeoX half-split RoPE. Default: False.

    Returns:
        Attention output of shape [B, H, S, D] in the input dtype.
    """
    if not _has_quantized_sdpa:
        raise RuntimeError(
            "fp8_fa4_rope_sdpa requires a PyTorch version with "
            "torch.nn.attention.experimental._scaled_dot_product_attention_quantized. "
            "Please upgrade PyTorch."
        )
    if attn_mask is not None:
        raise ValueError("attn_mask not supported for FP8 FA4")
    if dropout_p != 0.0:
        raise ValueError(f"dropout_p must be 0.0 for FP8 FA4, got {dropout_p}")

    input_dtype = query.dtype

    q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = _fp8_rope_sdpa_quantize(
        query, key, value, cos, sin,
        rope_interleaved=rope_interleaved,
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
