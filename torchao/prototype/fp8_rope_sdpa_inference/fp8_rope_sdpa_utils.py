# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for wrapping modules with FP8 RoPE + SDPA fusion.

This module provides FLUX-specific attention processors that fuse:
- RoPE (Rotary Position Embedding)
- FP8 quantization
- Scaled dot-product attention

The baseline FluxAttnProcessor code is adapted from diffusers.
"""

from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.fp8_rope_sdpa_inference.fp8_rope_sdpa_attention import (
    fp8_rope_sdpa_flux,
)
from torchao.prototype.fp8_sdpa_inference.fp8_sdpa_attention import (
    fp8_sdpa_parallel,
)


# =============================================================================
# Context manager for FP8 RoPE + SDPA
# =============================================================================


@contextmanager
def fp8_rope_sdpa_context():
    """
    Context manager that enables FA3 backend for FP8 SDPA.

    Use this to wrap forward passes that use FP8 RoPE + SDPA attention.
    The FA3 backend is activated once at entry and restored at exit.

    Example:
        >>> with torch.no_grad():
        ...     with fp8_rope_sdpa_context():
        ...         # Transformer uses FP8 RoPE + SDPA
        ...         output = transformer(inputs, ...)
    """
    activate_flash_attention_impl("FA3")
    try:
        yield
    finally:
        restore_flash_attention_impl()


# =============================================================================
# RoPE implementation (from diffusers.models.embeddings)
# =============================================================================


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query or key 'x' tensors
    using the provided frequency tensor 'freqs_cis'. The input tensors are reshaped
    as complex numbers, and the frequency tensor is reshaped for broadcasting
    compatibility.

    Args:
        x: Query or key tensor to apply rotary embeddings. [B, H, S, D] or [B, S, H, D]
        freqs_cis: Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)
        use_real: Whether to use real-valued computation (True for FLUX)
        use_real_unbind_dim: Dimension for unbinding real/imag parts (-1 for FLUX)
        sequence_dim: Dimension for sequence (1 for FLUX, 2 for others)

    Returns:
        Tensor with rotary embeddings applied.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# =============================================================================
# QKV projection helpers (from diffusers.models.transformers.transformer_flux)
# =============================================================================


def _get_projections(attn, hidden_states, encoder_hidden_states=None):
    """Get Q, K, V projections using separate linear layers."""
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn, hidden_states, encoder_hidden_states=None):
    """Get Q, K, V projections using fused linear layer."""
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(
            encoder_hidden_states
        ).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn, hidden_states, encoder_hidden_states=None):
    """Get Q, K, V projections, using fused or separate layers based on config."""
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


# =============================================================================
# Baseline FluxAttnProcessor (from diffusers.models.transformers.transformer_flux)
# This is the original implementation we'll modify for FP8 RoPE + SDPA fusion.
# =============================================================================


class FluxAttnProcessor:
    """
    Baseline FLUX attention processor.

    This is the original implementation from diffusers that we use as reference.
    The flow is:
        1. Linear projections (to_q, to_k, to_v)
        2. Reshape to multi-head format
        3. RMSNorm on Q and K (norm_q, norm_k)
        4. Concatenate encoder hidden states (for cross-attention)
        5. Apply RoPE (rotary position embeddings)
        6. Scaled dot-product attention
        7. Output projection
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. "
                "Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn,  # FluxAttention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Step 1: Get Q, K, V projections
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        # Step 2: Reshape to multi-head format [B, S, H, D]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Step 3: Apply RMSNorm on Q and K
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Step 4: Handle encoder hidden states (cross-attention)
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Step 5: Apply RoPE (rotary position embeddings)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Step 6: Scaled dot-product attention
        # TODO: Replace this with FP8 RoPE + SDPA fusion
        # Currently uses F.scaled_dot_product_attention via dispatch
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),  # [B, H, S, D]
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(1, 2)  # [B, S, H, D]
        hidden_states = hidden_states.flatten(2, 3)  # [B, S, H*D]
        hidden_states = hidden_states.to(query.dtype)

        # Step 7: Output projection
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


# =============================================================================
# FP8 RoPE + SDPA fused attention processor
# TODO: Implement the fused version
# =============================================================================


class FP8RoPESDPAFluxAttnProcessor:
    """
    FLUX attention processor with fused FP8 RoPE + SDPA.

    This processor replaces the RoPE + SDPA steps with a single fused kernel
    that performs:
        1. RoPE (rotary position embeddings)
        2. FP8 quantization
        3. Scaled dot-product attention in FP8
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. "
                "Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn,  # FluxAttention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Step 1: Get Q, K, V projections
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        # Step 2: Reshape to multi-head format [B, S, H, D]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Step 3: Apply RMSNorm on Q and K
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Step 4: Handle encoder hidden states (cross-attention)
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Step 5 + 6: Fused RoPE + FP8 quantization + SDPA
        if image_rotary_emb is not None:
            # Use fused FP8 RoPE + SDPA
            hidden_states = fp8_rope_sdpa_flux(
                query,
                key,
                value,
                freqs_cis=image_rotary_emb,
                attn_mask=None,  # attention_mask not supported for FP8
                is_causal=False,
                scale=None,
            )
        else:
            # No RoPE needed, use FP8 SDPA directly
            hidden_states = fp8_sdpa_parallel(
                query.transpose(1, 2),  # [B, H, S, D]
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            )
            hidden_states = hidden_states.transpose(1, 2)  # [B, S, H, D]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Step 7: Output projection
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


# =============================================================================
# Module wrapping utility
# =============================================================================


def wrap_module_with_fp8_rope_sdpa(module: nn.Module) -> nn.Module:
    """
    Wrap a module to use fused FP8 RoPE + SDPA.

    This function transforms attention layers in the module to use a fused
    kernel that combines:
    - RoPE (Rotary Position Embedding)
    - FP8 quantization
    - Scaled dot-product attention

    The module's forward pass is wrapped with fp8_rope_sdpa_context() to
    activate the FA3 backend once per forward pass.

    Args:
        module: The module to wrap (e.g., a FLUX transformer)

    Returns:
        The wrapped module with fused FP8 RoPE + SDPA attention
    """
    # Find all FluxAttention modules and replace their processor
    fp8_processor = FP8RoPESDPAFluxAttnProcessor()

    for name, submodule in module.named_modules():
        # Check if this is a FluxAttention module (has a processor attribute)
        if hasattr(submodule, "processor") and hasattr(submodule, "to_q"):
            # Check if it's a FLUX-style attention (has norm_q, norm_k)
            if hasattr(submodule, "norm_q") and hasattr(submodule, "norm_k"):
                submodule.processor = fp8_processor

    # Wrap forward to use FA3 context (activates once per forward pass)
    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        with fp8_rope_sdpa_context():
            return original_forward(*args, **kwargs)

    module.forward = wrapped_forward

    return module
