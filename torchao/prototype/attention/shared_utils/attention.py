# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared FP8 scaled dot-product attention implementation.

This module contains the backend-agnostic FP8 SDPA logic.  The actual
backend (e.g., FA3) is selected by the caller via
``activate_flash_attention_impl`` *before* calling these functions.

Backend-specific modules (``fp8_fa3/attention.py``, etc.) provide thin
named wrappers around these functions.
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

_MIN_VERSION_ERROR = (
    "Low-precision attention requires PyTorch 2.11+. "
    "Please update your PyTorch version."
)

from torchao.prototype.attention.quantization import (
    _fp8_sdpa_quantize,
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
    *,
    backend_name: str = "FP8",
) -> torch.Tensor:
    """FP8 SDPA implementation shared by all backends.

    Quantizes Q, K, V to FP8 with per-head scaling, then runs
    ``_scaled_dot_product_attention_quantized`` under the
    ``SDPBackend.FLASH_ATTENTION`` kernel selector.

    .. important::

        The correct flash attention implementation (e.g., FA3) must
        be activated **before** calling this function::

            from torch.nn.attention import (
                activate_flash_attention_impl,
                restore_flash_attention_impl,
            )

            activate_flash_attention_impl("FA3")
            try:
                out = _fp8_sdpa(q, k, v, is_causal=True)
            finally:
                restore_flash_attention_impl()

        The high-level API (``apply_low_precision_attention``) handles
        this automatically.  This requirement only applies when calling
        this function (or the backend-specific wrappers like
        ``fp8_fa3_sdpa``) directly.

    Args:
        query: Query tensor of shape [B, H, S, D] in bf16/fp16.
        key: Key tensor of shape [B, H, S, D] in bf16/fp16.
        value: Value tensor of shape [B, H, S, D] in bf16/fp16.
        attn_mask: Not supported, must be None.
        dropout_p: Not supported, must be 0.0.
        is_causal: Whether to apply causal masking.
        scale: Scaling factor for attention. If None, uses 1/sqrt(D).
        enable_gqa: Whether to enable grouped query attention.
        backend_name: Name of the backend for error messages.

    Returns:
        Attention output of shape [B, H, S, D] in the input dtype.
    """
    if not _TORCH_VERSION_AT_LEAST_2_11:
        raise RuntimeError(_MIN_VERSION_ERROR)
    if attn_mask is not None:
        raise ValueError(f"attn_mask not supported for FP8 {backend_name}")
    if dropout_p != 0.0:
        raise ValueError(
            f"dropout_p must be 0.0 for FP8 {backend_name}, got {dropout_p}"
        )

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
