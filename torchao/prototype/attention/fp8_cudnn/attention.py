# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 SDPA using cuDNN per-tensor backend.

Unlike the FA3 backend, cuDNN per-tensor does not require
activate_flash_attention_impl or an sdpa_kernel context.
"""

from functools import partial
from typing import Optional

import torch

from torchao.utils import torch_version_at_least

_TORCH_VERSION_AT_LEAST_2_11 = torch_version_at_least("2.11.0")

if _TORCH_VERSION_AT_LEAST_2_11:
    from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
        DescaleType,
        _scaled_dot_product_attention_quantized,
    )

from torchao.prototype.attention.quantization import (
    _fp8_per_tensor_rope_sdpa_quantize,
    _fp8_per_tensor_sdpa_quantize,
)
from torchao.prototype.attention.shared_utils.custom_ops import (
    register_fp8_attention_ops,
)


def _fp8_cudnn_sdpa(
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
    backend_name: str = "CUDNN",
) -> torch.Tensor:
    """FP8 SDPA using cuDNN per-tensor backend.

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
    if hadamard != "NONE":
        raise ValueError(
            f"Hadamard transform not yet supported for {backend_name} backend"
        )

    q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = (
        _fp8_per_tensor_sdpa_quantize(query, key, value)
    )

    return _scaled_dot_product_attention_quantized(
        q_fp8,
        k_fp8,
        v_fp8,
        is_causal=is_causal,
        scale=scale,
        q_descale=descale_q,
        k_descale=descale_k,
        v_descale=descale_v,
        q_descale_type=DescaleType.PER_TENSOR,
        k_descale_type=DescaleType.PER_TENSOR,
        v_descale_type=DescaleType.PER_TENSOR,
    )


def _fp8_cudnn_rope_sdpa(
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
    backend_name: str = "CUDNN",
) -> torch.Tensor:
    """Fused RoPE + FP8 SDPA using cuDNN per-tensor backend.

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
    if hadamard != "NONE":
        raise ValueError(
            f"Hadamard transform not yet supported for {backend_name} backend"
        )

    cos = cos.to(query.device)
    sin = sin.to(query.device)

    q_fp8, k_fp8, v_fp8, descale_q, descale_k, descale_v = (
        _fp8_per_tensor_rope_sdpa_quantize(
            query, key, value, cos, sin, rope_interleaved=rope_interleaved
        )
    )

    return _scaled_dot_product_attention_quantized(
        q_fp8,
        k_fp8,
        v_fp8,
        is_causal=is_causal,
        scale=scale,
        q_descale=descale_q,
        k_descale=descale_k,
        v_descale=descale_v,
        q_descale_type=DescaleType.PER_TENSOR,
        k_descale_type=DescaleType.PER_TENSOR,
        v_descale_type=DescaleType.PER_TENSOR,
    )


fp8_cudnn_sdpa = partial(_fp8_cudnn_sdpa, backend_name="CUDNN")
fp8_cudnn_sdpa.__doc__ = _fp8_cudnn_sdpa.__doc__
fp8_cudnn_sdpa.__name__ = "fp8_cudnn_sdpa"
fp8_cudnn_sdpa.__qualname__ = "fp8_cudnn_sdpa"

fp8_cudnn_rope_sdpa = partial(_fp8_cudnn_rope_sdpa, backend_name="CUDNN")
fp8_cudnn_rope_sdpa.__doc__ = _fp8_cudnn_rope_sdpa.__doc__
fp8_cudnn_rope_sdpa.__name__ = "fp8_cudnn_rope_sdpa"
fp8_cudnn_rope_sdpa.__qualname__ = "fp8_cudnn_rope_sdpa"

_ops = register_fp8_attention_ops(
    backend_name="cudnn",
    rope_sdpa_fn=fp8_cudnn_rope_sdpa,
    sdpa_fn=fp8_cudnn_sdpa,
)
