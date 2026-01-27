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

The fused kernel reduces memory traffic by applying RoPE and quantization
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
    This function uses a fused Triton kernel for RoPE + quantization.

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

    # Ensure cos/sin are on the correct device and contiguous
    cos = cos.to(query.device).contiguous()
    sin = sin.to(query.device).contiguous()

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


# =============================================================================
# Python reference implementation (for testing/comparison)
# =============================================================================


def fp8_rope_sdpa_flux_reference(
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
    Reference Python implementation of FP8 RoPE + SDPA for FLUX-style inputs.

    This is the non-fused version used for testing and comparison.
    Uses pure PyTorch operations for RoPE and quantization.

    Args:
        query: Query tensor of shape (B, S, H, D)
        key: Key tensor of shape (B, S, H, D)
        value: Value tensor of shape (B, S, H, D)
        freqs_cis: Tuple of (cos, sin) tensors for RoPE, each of shape (S, D)
        attn_mask: Not supported, must be None
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for attention
        num_chunks: Ignored (for API compatibility)

    Returns:
        Attention output tensor of shape (B, S, H, D)
    """
    if attn_mask is not None:
        raise ValueError("attn_mask is not supported for FP8 RoPE SDPA")

    B, S, H, D = query.shape
    cos, sin = freqs_cis

    # Broadcast cos/sin for (B, S, H, D) format (sequence_dim=1)
    cos = cos[None, :, None, :]  # (1, S, 1, D)
    sin = sin[None, :, None, :]  # (1, S, 1, D)
    cos, sin = cos.to(query.device), sin.to(query.device)

    # Apply RoPE to Q and K (FLUX uses use_real_unbind_dim=-1)
    q_real, q_imag = query.reshape(*query.shape[:-1], -1, 2).unbind(-1)
    q_rotated = torch.stack([-q_imag, q_real], dim=-1).flatten(3)
    query_rope = (query.float() * cos + q_rotated.float() * sin).to(query.dtype)

    k_real, k_imag = key.reshape(*key.shape[:-1], -1, 2).unbind(-1)
    k_rotated = torch.stack([-k_imag, k_real], dim=-1).flatten(3)
    key_rope = (key.float() * cos + k_rotated.float() * sin).to(key.dtype)

    # Transpose to [B, H, S, D] for SDPA
    query_rope = query_rope.transpose(1, 2)
    key_rope = key_rope.transpose(1, 2)
    value_t = value.transpose(1, 2)

    # Quantize to FP8 (per-head scaling)
    FP8_MAX = 448.0
    eps = 1e-12

    # Compute max values in fp32 for precision
    q_max = query_rope.float().abs().amax(dim=(2, 3), keepdim=True)
    k_max = key_rope.float().abs().amax(dim=(2, 3), keepdim=True)
    v_max = value_t.float().abs().amax(dim=(2, 3), keepdim=True)

    # Compute scales with conditional logic
    q_scale = torch.where(q_max > eps, FP8_MAX / q_max, torch.ones_like(q_max))
    k_scale = torch.where(k_max > eps, FP8_MAX / k_max, torch.ones_like(k_max))
    v_scale = torch.where(v_max > eps, FP8_MAX / v_max, torch.ones_like(v_max))

    # Quantize in fp32 before casting to fp8
    q_fp8 = (query_rope.float() * q_scale).to(torch.float8_e4m3fn)
    k_fp8 = (key_rope.float() * k_scale).to(torch.float8_e4m3fn)
    v_fp8 = (value_t.float() * v_scale).to(torch.float8_e4m3fn)

    # Compute descales
    q_descale = (
        torch.where(q_max > eps, q_max / FP8_MAX, torch.ones_like(q_max))
        .squeeze(-1)
        .squeeze(-1)
    )
    k_descale = (
        torch.where(k_max > eps, k_max / FP8_MAX, torch.ones_like(k_max))
        .squeeze(-1)
        .squeeze(-1)
    )
    v_descale = (
        torch.where(v_max > eps, v_max / FP8_MAX, torch.ones_like(v_max))
        .squeeze(-1)
        .squeeze(-1)
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
