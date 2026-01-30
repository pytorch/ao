# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 RoPE + SDPA wrapper.

This module provides a functional interface for fused:
- RoPE (Rotary Position Embeddings)
- FP8 quantization
- Scaled dot-product attention

The fused operation reduces memory traffic by applying RoPE and quantization
in a single pass before calling FP8 SDPA.
"""

from typing import Optional, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
)

from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_quantization import (
    fp8_rope_quantize_func,
)

# =============================================================================
# Variant for FLUX-style inputs (sequence_dim=1)
# FLUX uses [B, S, H, D] format internally, need to handle the transpose
# =============================================================================


def fp8_rope_sdpa_flux(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    num_chunks: Optional[int] = None,
) -> torch.Tensor:
    """
    FP8 RoPE + SDPA for FLUX-style inputs.

    FLUX uses [B, S, H, D] tensor format with sequence_dim=1 for RoPE.
    This function uses the Triton kernels provided in fp8_rope_sdpa_quantization for RoPE + quantization.

    Args:
        query: Query tensor of shape (B, S, H, D)
        key: Key tensor of shape (B, S, H, D)
        value: Value tensor of shape (B, S, H, D)
        freqs_cis: Tuple of (cos, sin) tensors for RoPE, each of shape (S, D)
        attn_mask: Not supported, must be None
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for attention
        num_chunks: Number of chunks for parallelized quantization

    Returns:
        Attention output tensor of shape (B, S, H, D)
    """
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported for FP8 RoPE SDPA")

    B, S, H, D = query.shape
    cos, sin = freqs_cis

    # Fused RoPE + quantization using Triton kernel
    # Input: [B, S, H, D], Output: [B, H, S, D] (ready for SDPA)
    q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = fp8_rope_quantize_func(
        query, key, value, cos, sin, num_chunks
    )

    # Call PyTorch's FP8 SDPA
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = _scaled_dot_product_attention_quantized(
            q_fp8,
            k_fp8,
            v_fp8,
            is_causal=is_causal,
            scale=scale,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    # Transpose back to [B, S, H, D]
    out = out.transpose(1, 2)

    return out
