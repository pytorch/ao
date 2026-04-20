# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared FP8 scaled dot-product attention implementation.

Backend-specific modules (``fp8_fa3/attention.py``, etc.) provide thin
named wrappers around these functions via ``functools.partial``.
"""

from typing import Optional

import torch

from torchao.utils import torch_version_at_least

_TORCH_VERSION_AT_LEAST_2_11 = torch_version_at_least("2.11.0")

if _TORCH_VERSION_AT_LEAST_2_11:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
        _scaled_dot_product_attention_quantized,
    )

from torchao.prototype.attention.quantization import (
    _fp8_hadamard_rope_sdpa_quantize,
    _fp8_hadamard_sdpa_quantize,
    _fp8_rope_sdpa_quantize,
    _fp8_sdpa_quantize,
    _inverse_hadamard_transform,
)


def _fp8_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    hadamard: str = "NONE",
    *,
    backend_name: str = "FP8",
) -> torch.Tensor:
    """FP8 SDPA shared by all backends.

    The correct flash attention implementation (e.g. FA3) must be
    activated before calling this function. The high-level
    ``apply_low_precision_attention`` API handles this automatically.

    Input/output layout: [B, H, S, D].
    """
    if not _TORCH_VERSION_AT_LEAST_2_11:
        raise RuntimeError("Low-precision attention requires PyTorch 2.11+.")
    if attn_mask is not None:
        raise ValueError(f"attn_mask not supported for FP8 {backend_name}")
    if dropout_p != 0.0:
        raise ValueError(
            f"dropout_p must be 0.0 for FP8 {backend_name}, got {dropout_p}"
        )

    input_dtype = query.dtype
    use_hadamard = hadamard != "NONE"

    if use_hadamard:
        q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = (
            _fp8_hadamard_sdpa_quantize(
                query, key, value, v_only=(hadamard == "V_ONLY")
            )
        )
    else:
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

    out = out.to(input_dtype)
    if use_hadamard:
        out = _inverse_hadamard_transform(out)
    return out


def _fp8_rope_sdpa(
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
    hadamard: str = "NONE",
    *,
    backend_name: str = "FP8",
) -> torch.Tensor:
    """Fused RoPE + FP8 SDPA shared by all backends.

    Input layout: [B, S, H, D] (pre-transpose). The fused quantization
    kernel handles the transpose to [B, H, S, D] internally.
    Output layout: [B, H, S, D].
    """
    if not _TORCH_VERSION_AT_LEAST_2_11:
        raise RuntimeError("Low-precision attention requires PyTorch 2.11+.")
    if attn_mask is not None:
        raise ValueError(f"attn_mask not supported for FP8 {backend_name}")
    if dropout_p != 0.0:
        raise ValueError(
            f"dropout_p must be 0.0 for FP8 {backend_name}, got {dropout_p}"
        )

    input_dtype = query.dtype
    use_hadamard = hadamard != "NONE"

    cos = cos.to(query.device)
    sin = sin.to(query.device)

    if use_hadamard:
        q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = (
            _fp8_hadamard_rope_sdpa_quantize(
                query,
                key,
                value,
                cos,
                sin,
                rope_interleaved=rope_interleaved,
                v_only=(hadamard == "V_ONLY"),
            )
        )
    else:
        q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = _fp8_rope_sdpa_quantize(
            query, key, value, cos, sin, rope_interleaved=rope_interleaved
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

    out = out.to(input_dtype)
    if use_hadamard:
        out = _inverse_hadamard_transform(out)
    return out
