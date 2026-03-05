# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
TransformerEngine (TE) backend for MXFP8 grouped GEMM in MoE training.

Provides custom ops (te_moe::gemm_fwd, te_moe::gemm_dgrad, te_moe::gemm_wgrad)
that use TE's MXFP8Quantizer for block-scaled FP8 quantization (block_size=32,
E8M0 scales) and general_grouped_gemm for the CUDA GEMM kernel.

These ops are invoked when ``kernel_preference=KernelPreference.TE`` is set on
an ``MXFP8TrainingOpConfig``. They plug into the existing ``_MXFP8GroupedMM``
autograd function as an alternative to the torchao-native quantization +
CUTLASS path.

torch.compile compatible: each custom op has a fake (meta) implementation so
dynamo can trace through without executing TE C++ kernels.

Note: ``_offs_to_m_splits()`` calls ``offs.tolist()`` which triggers a GPU→CPU
sync. This is unavoidable because TE's ``general_grouped_gemm`` requires
``m_splits`` as a ``List[int]``. The result is cached on the tensor object so
the sync happens once per MoE layer per step (in fwd; dgrad/wgrad reuse it).
"""

import logging
from typing import Any, List, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

try:
    from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex
except ImportError as e:
    raise ImportError(
        "TransformerEngine is required for KernelPreference.TE but is not installed. "
        "Install from: https://github.com/NVIDIA/TransformerEngine"
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Offset conversion
# ──────────────────────────────────────────────────────────────────────────────


def _offs_to_m_splits(offs: torch.Tensor) -> List[int]:
    """Convert cumulative offsets tensor to per-group sizes (m_splits).

    offs is [n1, n1+n2, n1+n2+n3, ...] (cumulative, no leading zero).
    Returns [n1, n2, n3, ...].

    NOTE: calls offs.tolist() → GPU→CPU sync on first call. Result is
    cached on the tensor so that dgrad/wgrad reuse it without re-syncing.
    """
    cached = getattr(offs, "_cached_m_splits", None)
    if cached is not None:
        return cached
    if offs.numel() == 0:
        return []
    offs_list = offs.tolist()
    splits = [offs_list[0]]
    for i in range(1, len(offs_list)):
        splits.append(offs_list[i] - offs_list[i - 1])
    offs._cached_m_splits = splits
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# MXFP8 helpers
# ──────────────────────────────────────────────────────────────────────────────

_MXFP8_BLOCK = 32


def _ceil_to_block(n: int) -> int:
    return (n + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK * _MXFP8_BLOCK


def _pad_for_mxfp8(
    tensor: torch.Tensor, m_splits: List[int]
) -> Tuple[torch.Tensor, List[int]]:
    """Pad each expert chunk so its row count is divisible by 32.

    Uses TE's fused CUDA kernel (tex.fused_multi_row_padding) which copies
    real rows and zero-fills padding rows in a single kernel launch.
    """
    padded_splits = [_ceil_to_block(m) for m in m_splits]
    if padded_splits == m_splits:
        return tensor.contiguous(), m_splits

    padded_total = sum(padded_splits)
    K = tensor.shape[-1]
    padded = torch.empty(padded_total, K, dtype=tensor.dtype, device=tensor.device)
    tex.fused_multi_row_padding(
        tensor.contiguous().view(-1, K), padded, m_splits, padded_splits
    )
    return padded, padded_splits


def _unpad_mxfp8_output(
    padded_out: torch.Tensor,
    m_splits: List[int],
    padded_splits: List[int],
    out: torch.Tensor,
) -> None:
    """Extract real rows from padded GEMM output, discarding padding rows.

    Uses TE's fused CUDA kernel (tex.fused_multi_row_unpadding) for a
    single-kernel-launch scatter.
    """
    if padded_splits == m_splits:
        return
    total_tokens = sum(m_splits)
    tex.fused_multi_row_unpadding(
        padded_out, out[:total_tokens], padded_splits, m_splits
    )


def _mxfp8_quantize_inputs(
    A: torch.Tensor, m_splits: List[int], num_experts: int
) -> Tuple[Any, List[int]]:
    """Quantize input activations per expert group using MXFP8 (rowwise).

    Pads each expert chunk to a multiple of 32 rows.
    Returns (quantized_list, padded_splits).
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    padded_A, padded_splits = _pad_for_mxfp8(A, m_splits)
    quantizers = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
        q.internal = True
        q.optimize_for_gemm = True
        quantizers.append(q)
    return tex.split_quantize(padded_A, padded_splits, quantizers), padded_splits


def _mxfp8_quantize_weights(B_t: torch.Tensor, num_experts: int) -> List[Any]:
    """Quantize weights for forward pass (TN layout).

    B_t: [E, K, N] → transpose to [E, N, K] and quantize rowwise.
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    quantizers = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
        q.internal = True
        q.optimize_for_gemm = True
        quantizers.append(q)
    B_t_T = B_t.transpose(-2, -1).contiguous()  # [E, N, K]
    return [q.quantize_impl(B_t_T[i]) for i, q in enumerate(quantizers)]


def _mxfp8_quantize_weights_dgrad(B_t: torch.Tensor, num_experts: int) -> List[Any]:
    """Quantize weights for DGRAD (NN layout, columnwise usage).

    Transposes [E, K, N] → [E, N, K], quantizes with both rowwise+columnwise,
    then switches to columnwise-only usage matching TE's dgrad convention.
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    quantizers = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
        q.internal = True
        q.optimize_for_gemm = True
        quantizers.append(q)
    B_t_T = B_t.transpose(-2, -1).contiguous()  # [E, N, K]
    result = [q.quantize_impl(B_t_T[i]) for i, q in enumerate(quantizers)]
    for r in result:
        r.update_usage(rowwise_usage=False, columnwise_usage=True)
    return result


def _mxfp8_quantize_wgrad(
    A: torch.Tensor,
    grad_output: torch.Tensor,
    m_splits: List[int],
    num_experts: int,
) -> Tuple[Any, Any, List[int]]:
    """Quantize inputs (columnwise) and grads (rowwise) for WGRAD.

    NT-layout GEMM (wgrad = dOut^T @ X) requires both operands to have
    columnwise data available. Quantize with BOTH rowwise+columnwise,
    then switch inputs to columnwise-only.

    Returns (inputs_fp8, grads_fp8, padded_splits).
    """
    fp8_dtype = tex.DType.kFloat8E4M3
    padded_A, padded_splits = _pad_for_mxfp8(A, m_splits)
    padded_grad, _ = _pad_for_mxfp8(grad_output, m_splits)

    input_qs = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
        q.internal = True
        q.optimize_for_gemm = True
        input_qs.append(q)
    grad_qs = []
    for _ in range(num_experts):
        q = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
        q.internal = True
        q.optimize_for_gemm = True
        grad_qs.append(q)

    inputs_fp8 = tex.split_quantize(padded_A, padded_splits, input_qs)
    grads_fp8 = tex.split_quantize(padded_grad, padded_splits, grad_qs)
    for im in inputs_fp8:
        im.update_usage(rowwise_usage=False, columnwise_usage=True)
    return inputs_fp8, grads_fp8, padded_splits


# ──────────────────────────────────────────────────────────────────────────────
# Custom ops — torch.library registration for torch.compile compatibility
# ──────────────────────────────────────────────────────────────────────────────


@torch.library.custom_op("te_moe::gemm_fwd", mutates_args=())
def te_gemm_fwd(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    use_fp8: bool,
) -> torch.Tensor:
    """Forward grouped GEMM: out[i] = input[i] @ weight[i]^T  (TN layout).

    Args:
        A: [total_tokens, K] input activations.
        B_t: [E, K, N] weights (transposed from GroupedExperts convention).
        offs: [E] int32 cumulative offsets.
        out_dtype: output dtype (e.g. torch.bfloat16).
        use_fp8: If True, MXFP8 quantization; if False, BF16 direct.
    """
    m_splits = _offs_to_m_splits(offs)
    num_experts = len(m_splits)
    total_tokens = sum(m_splits)
    N = B_t.shape[-1]

    out = torch.empty(A.shape[0], N, dtype=out_dtype, device=A.device)
    A_used = A[:total_tokens] if A.shape[0] > total_tokens else A

    if use_fp8:
        inputmats_fp8, padded_splits = _mxfp8_quantize_inputs(
            A_used, m_splits, num_experts
        )
        weights_fp8 = _mxfp8_quantize_weights(B_t, num_experts)
        needs_unpad = padded_splits != m_splits
        if needs_unpad:
            padded_total = sum(padded_splits)
            padded_out = torch.empty(
                padded_total, N, dtype=out_dtype, device=A.device
            )
            gemm_out = padded_out
        else:
            gemm_out = out[:total_tokens]
        general_grouped_gemm(
            weights_fp8,
            inputmats_fp8,
            [gemm_out],
            [None] * num_experts,
            out_dtype,
            single_output=True,
            m_splits=padded_splits,
        )
        if needs_unpad:
            _unpad_mxfp8_output(padded_out, m_splits, padded_splits, out)
    else:
        B_t_T = B_t.transpose(-2, -1).contiguous()  # [E, N, K]
        weights = [B_t_T[i] for i in range(num_experts)]
        inputs = list(A_used.contiguous().split(m_splits))
        general_grouped_gemm(
            weights,
            inputs,
            [out],
            [None] * num_experts,
            out_dtype,
            single_output=True,
            m_splits=m_splits,
        )

    return out


@te_gemm_fwd.register_fake
def _te_gemm_fwd_fake(A, B_t, offs, out_dtype, use_fp8):
    return torch.empty(A.shape[0], B_t.shape[-1], dtype=out_dtype, device=A.device)


@torch.library.custom_op("te_moe::gemm_dgrad", mutates_args=())
def te_gemm_dgrad(
    grad_output: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    use_fp8: bool,
) -> torch.Tensor:
    """DGRAD: grad_A[i] = grad_out[i] @ weight[i]  (NN layout, columnwise weights)."""
    m_splits = _offs_to_m_splits(offs)
    num_experts = len(m_splits)
    total_tokens = sum(m_splits)
    K = B_t.shape[-2]

    grad_A = torch.empty(
        grad_output.shape[0], K, dtype=out_dtype, device=grad_output.device
    )
    grad_used = (
        grad_output[:total_tokens]
        if grad_output.shape[0] > total_tokens
        else grad_output
    )

    if use_fp8:
        grad_fp8, padded_splits = _mxfp8_quantize_inputs(
            grad_used, m_splits, num_experts
        )
        weights_col_fp8 = _mxfp8_quantize_weights_dgrad(B_t, num_experts)
        needs_unpad = padded_splits != m_splits
        if needs_unpad:
            padded_total = sum(padded_splits)
            padded_grad_A = torch.empty(
                padded_total, K, dtype=out_dtype, device=grad_output.device
            )
            gemm_out = padded_grad_A
        else:
            gemm_out = grad_A[:total_tokens]
        general_grouped_gemm(
            weights_col_fp8,
            grad_fp8,
            [gemm_out],
            [None] * num_experts,
            out_dtype,
            single_output=True,
            layout="NN",
            m_splits=padded_splits,
            grad=True,
            use_split_accumulator=True,
        )
        if needs_unpad:
            _unpad_mxfp8_output(padded_grad_A, m_splits, padded_splits, grad_A)
    else:
        bt_weights = [B_t[i].contiguous() for i in range(num_experts)]
        grad_splits = list(grad_used.contiguous().split(m_splits))
        general_grouped_gemm(
            bt_weights,
            grad_splits,
            [grad_A],
            [None] * num_experts,
            out_dtype,
            single_output=True,
            m_splits=m_splits,
            grad=True,
        )

    return grad_A


@te_gemm_dgrad.register_fake
def _te_gemm_dgrad_fake(grad_output, B_t, offs, out_dtype, use_fp8):
    K = B_t.shape[-2]
    return torch.empty(
        grad_output.shape[0], K, dtype=out_dtype, device=grad_output.device
    )


@torch.library.custom_op("te_moe::gemm_wgrad", mutates_args=())
def te_gemm_wgrad(
    A: torch.Tensor,
    grad_output: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    use_fp8: bool,
) -> torch.Tensor:
    """WGRAD: wgrad[i] = grad_out[i]^T @ A[i]  (NT layout), returns [E, K, N]."""
    m_splits = _offs_to_m_splits(offs)
    num_experts = len(m_splits)
    total_tokens = sum(m_splits)
    N = grad_output.shape[-1]
    K = A.shape[-1]

    wgrad_list = [
        torch.empty(N, K, dtype=out_dtype, device=A.device)
        for _ in range(num_experts)
    ]

    A_used = A[:total_tokens] if A.shape[0] > total_tokens else A
    grad_used = (
        grad_output[:total_tokens]
        if grad_output.shape[0] > total_tokens
        else grad_output
    )

    if use_fp8:
        inputs_fp8, grads_fp8, padded_splits = _mxfp8_quantize_wgrad(
            A_used, grad_used, m_splits, num_experts
        )
        general_grouped_gemm(
            inputs_fp8,
            grads_fp8,
            wgrad_list,
            [None] * num_experts,
            out_dtype,
            layout="NT",
            m_splits=padded_splits,
            grad=True,
            use_split_accumulator=True,
        )
    else:
        input_splits = list(A_used.contiguous().split(m_splits))
        grad_splits = list(grad_used.contiguous().split(m_splits))
        general_grouped_gemm(
            input_splits,
            grad_splits,
            wgrad_list,
            [None] * num_experts,
            out_dtype,
            layout="NT",
            m_splits=m_splits,
            grad=True,
        )

    return torch.stack([w.t() for w in wgrad_list], dim=0)


@te_gemm_wgrad.register_fake
def _te_gemm_wgrad_fake(A, grad_output, offs, out_dtype, use_fp8):
    num_experts = offs.shape[0]
    K = A.shape[-1]
    N = grad_output.shape[-1]
    return torch.empty(num_experts, K, N, dtype=out_dtype, device=A.device)
