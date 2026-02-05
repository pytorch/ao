# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 RoPE + SDPA wrapper.

This module provides a functional interface for fused:
- RoPE (Rotary Position Embeddings)
- Hadamard transform (optional)
- FP8 quantization
- Scaled dot-product attention

The fused operation reduces memory traffic by applying RoPE and quantization
in a single pass before calling FP8 SDPA.

Supports multiple Hadamard transform modes to improve FP8 quantization quality:
- "none": No Hadamard transform
- "qkv": Hadamard on Q, K, and V (requires inverse Hadamard on output for V)
- "v_only": Hadamard on V only (requires inverse Hadamard on output)
"""

from typing import Literal, Optional, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
)

from torchao.prototype.fp8_rope_sdpa_inference.fp8_hadamard_rope_sdpa_quantization import (
    fp8_hadamard_rope_quantize_func,
    inverse_hadamard_transform,
)
from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_quantization_helion import (
    fp8_rope_quantize_func,
)
from torchao.prototype.fp8_rope_sdpa_inference.fp8_v_hadamard_rope_sdpa_quantization import (
    fp8_v_hadamard_rope_quantize_func,
)

# Type alias for hadamard mode
HadamardMode = Literal["none", "qkv", "v_only"]


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
    hadamard_mode: HadamardMode = "none",
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
        hadamard_mode: Hadamard transform mode for FP8 quantization quality:
            - "none": No Hadamard transform (default)
            - "qkv": Apply Hadamard to Q, K, and V before quantization
            - "v_only": Apply Hadamard to V only before quantization
            Both "qkv" and "v_only" require inverse Hadamard on output.

    Returns:
        Attention output tensor of shape (B, S, H, D)
    """
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported for FP8 RoPE SDPA")

    B, S, H, D = query.shape
    cos, sin = freqs_cis

    # Fused RoPE + quantization using Triton kernel
    # Input: [B, S, H, D], Output: [B, H, S, D] (ready for SDPA)
    if hadamard_mode == "qkv":
        # Use RoPE + Hadamard (on Q, K, and V) + FP8 quantization
        q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = (
            fp8_hadamard_rope_quantize_func(query, key, value, cos, sin, num_chunks)
        )
    elif hadamard_mode == "v_only":
        # Use RoPE + Hadamard (on V only) + FP8 quantization
        q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = (
            fp8_v_hadamard_rope_quantize_func(query, key, value, cos, sin, num_chunks)
        )
    else:
        # Use RoPE + FP8 quantization (no Hadamard)
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

    # Apply inverse Hadamard to recover correct attention output
    # This is needed because V was Hadamard-transformed before attention
    if hadamard_mode in ("qkv", "v_only"):
        out = inverse_hadamard_transform(out)

    return out
