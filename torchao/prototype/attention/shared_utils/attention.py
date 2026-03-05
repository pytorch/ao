# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared FP8 scaled dot-product attention implementation."""

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

    Quantizes Q, K, V to FP8 with per-head scaling, then calls
    ``_scaled_dot_product_attention_quantized`` under ``SDPBackend.FLASH_ATTENTION``.
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

    # Ensure Triton kernels launch on the correct device.
    # In the monkey-patch path (no torch.compile), accelerate's hooks move
    # tensors to the correct device but don't call torch.cuda.set_device().
    # Triton dispatches based on current_device(), not tensor device, so
    # without this guard the kernel launches on the wrong GPU's stream.
    _prev_device = torch.cuda.current_device()
    if query.device.index is not None and query.device.index != _prev_device:
        torch.cuda.set_device(query.device)

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

    # Restore previous device to avoid side effects on the caller.
    if query.device.index is not None and query.device.index != _prev_device:
        torch.cuda.set_device(_prev_device)

    return out.to(input_dtype)
