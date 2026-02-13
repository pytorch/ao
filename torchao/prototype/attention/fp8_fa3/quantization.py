# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Internal: FP8 quantization for attention inputs.
"""

from typing import Tuple

import torch

from torchao.quantization.quant_primitives import (
    _choose_scale_float8,
    _quantize_affine_float8,
)


def _quantize_per_head(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 with per-head scaling. Returns (tensor_fp8, descale)."""
    B, H, S, D = tensor.shape
    block_size = [1, 1, S, D]

    descale = _choose_scale_float8(
        tensor,
        block_size=block_size,
        float8_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
    )

    tensor_fp8 = _quantize_affine_float8(
        tensor,
        scale=descale,
        float8_dtype=torch.float8_e4m3fn,
    )

    descale = descale.squeeze(-1).squeeze(-1)
    return tensor_fp8, descale


def _fp8_sdpa_quantize(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize Q, K, V to FP8 with per-head scaling."""
    if q.dim() != 4:
        raise ValueError(f"Expected 4D tensor for q, got {q.dim()}D")
    if k.dim() != 4:
        raise ValueError(f"Expected 4D tensor for k, got {k.dim()}D")
    if v.dim() != 4:
        raise ValueError(f"Expected 4D tensor for v, got {v.dim()}D")
    if k.shape != v.shape:
        raise ValueError(f"K and V shape mismatch: {k.shape} vs {v.shape}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"Batch size mismatch: {q.shape[0]} vs {k.shape[0]}")
    if q.shape[1] != k.shape[1]:
        raise ValueError(f"Head count mismatch: {q.shape[1]} vs {k.shape[1]}")
    if q.shape[3] != k.shape[3]:
        raise ValueError(f"Head dim mismatch: {q.shape[3]} vs {k.shape[3]}")

    if torch.compiler.is_compiling():
        # Under torch.compile, use the PyTorch primitives path which the
        # compiler can trace and optimize.
        q_fp8, q_descale = _quantize_per_head(q)
        k_fp8, k_descale = _quantize_per_head(k)
        v_fp8, v_descale = _quantize_per_head(v)
        return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale
    else:
        # In eager mode, use fused Triton kernels for better performance.
        from torchao.prototype.attention.fp8_fa3.triton_qkv_quantization import (
            triton_fp8_sdpa_quantize,
        )

        return triton_fp8_sdpa_quantize(q, k, v)
