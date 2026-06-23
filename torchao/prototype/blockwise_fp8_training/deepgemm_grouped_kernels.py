# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

import torch

from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    DeepGemmGroupedOffsetPlan,
    DeepGemmKGroupedQuantMetadata,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
    _triton_fp8_blockwise_act_quant_k_grouped_deepgemm_with_group_sizes,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _is_row_major,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)

# H100 tuning on the DeepSeek-V3 MoE shapes showed direct flat quantization wins
# for the wide K=4096/7168 operands, while the existing TorchAO quantizer plus
# flatten is faster for the common N=2048 operand. Keep the threshold here with
# the DeepGEMM backend policy so future PyTorch grouped FP8 support can replace
# the whole path without leaking this heuristic into the public MoE op.
_DEEPGEMM_DIRECT_K_GROUPED_QUANT_MIN_DIM = 4096


class DeepGemmKGroupedLayout(str, Enum):
    """How a K-grouped operand is represented before DeepGEMM dispatch."""

    FLAT = "flat"
    TORCHAO_TRANSPOSED_LHS = "torchao_transposed_lhs"
    TORCHAO_RHS = "torchao_rhs"


@dataclass(frozen=True)
class DeepGemmKGroupedOperand:
    """A quantized K-grouped operand plus enough layout info to dispatch it.

    Args:
        data: FP8 data buffer, either already flat in DeepGEMM order or in a
            TorchAO intermediate layout.
        scale: Float32 inverse scales for the operand.
        dim: Logical non-token dimension used by DeepGEMM's K-grouped API.
        layout: Current representation of ``data``.
    """

    data: torch.Tensor
    scale: torch.Tensor
    dim: int
    layout: DeepGemmKGroupedLayout


@dataclass(frozen=True)
class DeepGemmWgradPlan:
    """Quantized operands for DeepGEMM's K-grouped weight-gradient kernel."""

    lhs: DeepGemmKGroupedOperand
    rhs: DeepGemmKGroupedOperand


@dataclass(frozen=True)
class DeepGemmCapabilities:
    """Cached import result and training symbols for the optional dependency."""

    module: object | None
    import_error: ImportError | None
    m_grouped_gemm: object | None
    k_grouped_gemm: object | None


def _deepgemm_import_error(exc: Exception) -> ImportError:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "deep_gemm":
        return ImportError(
            "DeepGEMM backend selected for FP8 blockwise MoE grouped GEMM, "
            "but optional dependency `deep_gemm` is not installed. Install "
            "DeepGEMM from https://github.com/deepseek-ai/DeepGEMM or select "
            "KernelPreference.EMULATED."
        )
    if isinstance(exc, ModuleNotFoundError):
        return ImportError(
            "DeepGEMM backend selected for FP8 blockwise MoE grouped GEMM, "
            "but the installed `deep_gemm` package failed to import one of "
            f"its dependencies: {exc}. Reinstall DeepGEMM against the active "
            "PyTorch and CUDA environment, or select KernelPreference.EMULATED."
        )
    return ImportError(
        "DeepGEMM backend selected for FP8 blockwise MoE grouped GEMM, "
        "but the installed `deep_gemm` package could not be imported. This "
        "usually means DeepGEMM was built against a different PyTorch/CUDA "
        f"ABI than the one currently imported. Original error: {exc}. "
        "Reinstall DeepGEMM against the active PyTorch and CUDA environment, "
        "or select KernelPreference.EMULATED."
    )


def _first_deepgemm_symbol(module: object, *names: str):
    for name in names:
        symbol = getattr(module, name, None)
        if symbol is not None:
            return symbol
    return None


@lru_cache(maxsize=1)
def _get_deepgemm_capabilities() -> DeepGemmCapabilities:
    try:
        module = importlib.import_module("deep_gemm")
    except (ImportError, OSError) as exc:
        return DeepGemmCapabilities(
            module=None,
            import_error=_deepgemm_import_error(exc),
            m_grouped_gemm=None,
            k_grouped_gemm=None,
        )

    return DeepGemmCapabilities(
        module=module,
        import_error=None,
        m_grouped_gemm=_first_deepgemm_symbol(
            module,
            "m_grouped_fp8_gemm_nt_contiguous",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
        ),
        k_grouped_gemm=getattr(module, "k_grouped_fp8_gemm_nt_contiguous", None),
    )


def _clear_deepgemm_capability_cache() -> None:
    _get_deepgemm_capabilities.cache_clear()


def _require_deep_gemm():
    capabilities = _get_deepgemm_capabilities()
    if capabilities.import_error is not None:
        raise capabilities.import_error
    assert capabilities.module is not None
    return capabilities.module


def is_deep_gemm_available() -> bool:
    return _get_deepgemm_capabilities().module is not None


def _is_cuda_sm90_or_newer(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    major, _ = torch.cuda.get_device_capability(x.device)
    return major >= 9


def can_use_deepgemm_grouped_training(
    a: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> bool:
    capabilities = _get_deepgemm_capabilities()
    return (
        block_size == 128
        and out_dtype == torch.bfloat16
        and capabilities.m_grouped_gemm is not None
        and capabilities.k_grouped_gemm is not None
        and _is_cuda_sm90_or_newer(a)
    )


def _flatten_k_grouped_transposed_lhs(
    a_t: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    parts = []
    start = 0
    for group_size in group_sizes:
        end = start + group_size
        # TorchAO's wgrad LHS quantizer returns logical (N_out, M_tokens).
        # DeepGEMM's K-grouped K-major buffer stores one expert at a time as
        # (N_out, expert_tokens). Slice the token dimension per expert before
        # flattening so each expert's K range is contiguous in DeepGEMM order.
        parts.append(a_t[:, start:end].contiguous().view(-1))
        start = end
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _flatten_k_grouped_rhs(
    b: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    parts = []
    start = 0
    for group_size in group_sizes:
        end = start + group_size
        # TorchAO's wgrad RHS quantizer returns logical (M_tokens, K_in) in a
        # column-major physical layout for torch._grouped_mm. DeepGEMM's
        # K-grouped K-major buffer stores each expert as (K_in, expert_tokens),
        # so transpose the logical slice and make that DeepGEMM flat order.
        parts.append(b[start:end].transpose(-2, -1).contiguous().view(-1))
        start = end
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _deepgemm_flat_k_grouped_operand(
    data: torch.Tensor,
    scale: torch.Tensor,
) -> DeepGemmKGroupedOperand:
    assert data.ndim == 1, "DeepGEMM flat K-grouped data must be 1D"
    assert scale.ndim == 2, "DeepGEMM flat K-grouped scale must be 2D"
    return DeepGemmKGroupedOperand(
        data=data,
        scale=scale,
        dim=scale.shape[0],
        layout=DeepGemmKGroupedLayout.FLAT,
    )


def _torchao_transposed_lhs_operand(
    data: torch.Tensor,
    scale: torch.Tensor,
) -> DeepGemmKGroupedOperand:
    assert data.ndim == 2, "TorchAO transposed-LHS data must be 2D"
    assert scale.ndim == 2, "TorchAO transposed-LHS scale must be 2D"
    return DeepGemmKGroupedOperand(
        data=data,
        scale=scale,
        dim=data.shape[0],
        layout=DeepGemmKGroupedLayout.TORCHAO_TRANSPOSED_LHS,
    )


def _torchao_rhs_operand(
    data: torch.Tensor,
    scale: torch.Tensor,
) -> DeepGemmKGroupedOperand:
    assert data.ndim == 2, "TorchAO RHS data must be 2D"
    assert scale.ndim == 2, "TorchAO RHS scale must be 2D"
    return DeepGemmKGroupedOperand(
        data=data,
        scale=scale,
        dim=data.shape[-1],
        layout=DeepGemmKGroupedLayout.TORCHAO_RHS,
    )


def _should_quantize_k_grouped_directly(dim: int) -> bool:
    return dim >= _DEEPGEMM_DIRECT_K_GROUPED_QUANT_MIN_DIM


def _quantize_wgrad_lhs(
    x: torch.Tensor,
    group_end_offsets: torch.Tensor,
    group_sizes: list[int],
    block_size: int,
    dtype: torch.dtype,
    metadata: DeepGemmKGroupedQuantMetadata | None,
) -> DeepGemmKGroupedOperand:
    if _should_quantize_k_grouped_directly(x.shape[-1]):
        q, scale = _triton_fp8_blockwise_act_quant_k_grouped_deepgemm_with_group_sizes(
            x.contiguous(),
            group_end_offsets,
            group_sizes,
            block_size=block_size,
            dtype=dtype,
            metadata=metadata,
        )
        # Direct quantization writes the DeepGEMM input contract directly:
        # flat per-expert (dim, tokens) data with (dim, token_blocks) scales.
        return _deepgemm_flat_k_grouped_operand(q, scale)

    q, scale = triton_fp8_blockwise_act_quant_transposed_lhs(
        x.contiguous(),
        block_size=block_size,
        dtype=dtype,
    )
    # For narrower LHS dimensions, TorchAO's transposed-LHS quantizer is
    # faster; the DeepGEMM launcher later flattens (dim, all_tokens) into
    # per-expert (dim, expert_tokens) chunks.
    return _torchao_transposed_lhs_operand(q, scale)


def _quantize_wgrad_rhs(
    x: torch.Tensor,
    group_end_offsets: torch.Tensor,
    group_sizes: list[int],
    block_size: int,
    dtype: torch.dtype,
    metadata: DeepGemmKGroupedQuantMetadata | None,
) -> DeepGemmKGroupedOperand:
    if _should_quantize_k_grouped_directly(x.shape[-1]):
        q, scale = _triton_fp8_blockwise_act_quant_k_grouped_deepgemm_with_group_sizes(
            x.contiguous(),
            group_end_offsets,
            group_sizes,
            block_size=block_size,
            dtype=dtype,
            metadata=metadata,
        )
        # Direct quantization writes the DeepGEMM input contract directly:
        # flat per-expert (dim, tokens) data with (dim, token_blocks) scales.
        return _deepgemm_flat_k_grouped_operand(q, scale)

    q, scale = triton_fp8_blockwise_act_quant_rhs(
        x.contiguous(),
        block_size=block_size,
        dtype=dtype,
    )
    # For narrower RHS dimensions, keep TorchAO's grouped-mm RHS contract:
    # logical (tokens, dim) data in column-major physical layout with
    # (token_blocks, dim) scales, then convert at DeepGEMM launch time.
    return _torchao_rhs_operand(q, scale)


def prepare_deepgemm_wgrad_plan(
    padded_grad_output: torch.Tensor,
    padded_a: torch.Tensor,
    offset_plan: DeepGemmGroupedOffsetPlan,
    block_size: int,
    dtype: torch.dtype,
) -> Optional[DeepGemmWgradPlan]:
    if not offset_plan.groups_are_block_aligned(block_size):
        # DeepGEMM's K-grouped FP8 kernel requires token counts to be aligned
        # because the token dimension is the GEMM K axis. Ragged no-padding
        # callers should keep using the emulated wgrad path for correctness.
        return None
    group_sizes = offset_plan.group_sizes
    lhs_needs_direct_quant_metadata = _should_quantize_k_grouped_directly(
        padded_grad_output.shape[-1]
    )
    rhs_needs_direct_quant_metadata = _should_quantize_k_grouped_directly(
        padded_a.shape[-1]
    )
    lhs_metadata = (
        offset_plan.k_quant_metadata(block_size, padded_grad_output.shape[-1])
        if lhs_needs_direct_quant_metadata
        else None
    )
    rhs_metadata = (
        offset_plan.k_quant_metadata(block_size, padded_a.shape[-1])
        if rhs_needs_direct_quant_metadata
        else None
    )
    return DeepGemmWgradPlan(
        lhs=_quantize_wgrad_lhs(
            padded_grad_output,
            offset_plan.group_end_offsets,
            group_sizes,
            block_size,
            dtype,
            lhs_metadata,
        ),
        rhs=_quantize_wgrad_rhs(
            padded_a,
            offset_plan.group_end_offsets,
            group_sizes,
            block_size,
            dtype,
            rhs_metadata,
        ),
    )


def deepgemm_blockwise_scaled_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
    *,
    offset_plan: DeepGemmGroupedOffsetPlan,
) -> torch.Tensor:
    assert block_size == 128, (
        "DeepGEMM FP8 blockwise grouped GEMM requires block_size=128"
    )
    assert _is_row_major(a), "deepgemm_blockwise_scaled_grouped_mm expected row-major A"
    assert offs is not None and offs.dtype == torch.int32, "offs must be int32"
    assert out_dtype == torch.bfloat16, (
        "DeepGEMM FP8 blockwise grouped GEMM currently supports bfloat16 output"
    )
    assert _scaling_type_value(scale_recipe_a) == _scaling_type_value(
        BLOCKWISE_1X128_SCALING_TYPE
    ), "DeepGEMM FP8 blockwise grouped GEMM expects 1x128 LHS scales"
    assert _scaling_type_value(scale_recipe_b) == _scaling_type_value(
        BLOCKWISE_128X128_SCALING_TYPE
    ), "DeepGEMM FP8 blockwise grouped GEMM expects 128x128 RHS scales"

    _require_deep_gemm()
    capabilities = _get_deepgemm_capabilities()
    if not a.is_cuda:
        raise NotImplementedError(
            "DeepGEMM FP8 blockwise grouped GEMM requires CUDA tensors. "
            "Select KernelPreference.EMULATED for non-CUDA execution."
        )
    assert b.stride(-1) == 1, (
        "deepgemm_blockwise_scaled_grouped_mm expected DeepGEMM RHS layout with K-major B"
    )
    assert a.shape[-1] == b.shape[-1], (
        f"shape {a.shape} and {b.shape} are not compatible"
    )

    grouped_layout = offset_plan.m_grouped_layout(a.shape[0])
    # DeepGEMM RHS is (..., N_out, K_contract); the output keeps TorchAO's
    # grouped GEMM shape (M, N_out).
    out = torch.empty(
        (a.shape[0], b.shape[-2]),
        dtype=out_dtype,
        device=a.device,
    )

    grouped_gemm = capabilities.m_grouped_gemm
    if grouped_gemm is None:
        raise ImportError(
            "Installed `deep_gemm` does not expose "
            "`m_grouped_fp8_gemm_nt_contiguous` or "
            "`m_grouped_fp8_fp4_gemm_nt_contiguous`. Install a DeepGEMM "
            "version with contiguous M-grouped FP8 GEMM support."
        )

    grouped_gemm(
        (a, a_s),
        (b, b_s),
        out,
        grouped_layout,
        recipe_a=(1, block_size),
        recipe_b=(block_size, block_size),
        disable_ue8m0_cast=True,
    )
    return out


def _prepare_k_grouped_lhs_operand(
    operand: DeepGemmKGroupedOperand,
    group_sizes: list[int],
    total_tokens: int,
    valid_scale_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if operand.layout == DeepGemmKGroupedLayout.FLAT:
        # Direct quantization already produced DeepGEMM's flat K-major
        # contract. Trim scale columns to valid padded rows and ignore any
        # upper-bound padding allocated for CUDA graph friendliness.
        return operand.data, operand.scale[:, :valid_scale_blocks]

    assert operand.layout == DeepGemmKGroupedLayout.TORCHAO_TRANSPOSED_LHS, (
        f"unsupported DeepGEMM wgrad LHS layout: {operand.layout}"
    )
    assert operand.data.shape[-1] >= total_tokens, (
        f"shape {operand.data.shape} and offs={total_tokens} are not compatible"
    )
    data = _flatten_k_grouped_transposed_lhs(operand.data, group_sizes)
    scale = operand.scale[:, :valid_scale_blocks]
    return data, scale


def _prepare_k_grouped_rhs_operand(
    operand: DeepGemmKGroupedOperand,
    group_sizes: list[int],
    total_tokens: int,
    valid_scale_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if operand.layout == DeepGemmKGroupedLayout.FLAT:
        # Direct quantization already produced DeepGEMM's flat K-major
        # contract. Trim scale columns to valid padded rows and ignore any
        # upper-bound padding allocated for CUDA graph friendliness.
        return operand.data, operand.scale[:, :valid_scale_blocks]

    assert operand.layout == DeepGemmKGroupedLayout.TORCHAO_RHS, (
        f"unsupported DeepGEMM wgrad RHS layout: {operand.layout}"
    )
    assert operand.data.shape[-2] >= total_tokens, (
        f"shape {operand.data.shape} and offs={total_tokens} are not compatible"
    )
    data = _flatten_k_grouped_rhs(operand.data, group_sizes)
    scale = operand.scale[:valid_scale_blocks]
    # TorchAO RHS 1x128 scales are stored as (M_blocks, K_in). DeepGEMM's
    # K-grouped scale contract is (K_in, M_blocks), matching its flattened
    # per-expert (K_in, expert_tokens) RHS data.
    scale = scale.transpose(-2, -1)
    return data, scale


def deepgemm_blockwise_scaled_grouped_mm_wgrad(
    lhs: DeepGemmKGroupedOperand,
    rhs: DeepGemmKGroupedOperand,
    offset_plan: DeepGemmGroupedOffsetPlan,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    assert block_size == 128, (
        "DeepGEMM FP8 blockwise grouped wgrad requires block_size=128"
    )
    assert lhs.layout in (
        DeepGemmKGroupedLayout.FLAT,
        DeepGemmKGroupedLayout.TORCHAO_TRANSPOSED_LHS,
    ), f"unsupported DeepGEMM wgrad LHS layout: {lhs.layout}"
    assert rhs.layout in (
        DeepGemmKGroupedLayout.FLAT,
        DeepGemmKGroupedLayout.TORCHAO_RHS,
    ), f"unsupported DeepGEMM wgrad RHS layout: {rhs.layout}"
    assert out_dtype in (torch.bfloat16, torch.float32), (
        "DeepGEMM FP8 blockwise grouped wgrad supports bfloat16 or float32 output"
    )

    _require_deep_gemm()
    capabilities = _get_deepgemm_capabilities()
    if not lhs.data.is_cuda:
        raise NotImplementedError(
            "DeepGEMM FP8 blockwise grouped wgrad requires CUDA tensors. "
            "Select KernelPreference.EMULATED for non-CUDA execution."
        )

    total_tokens = offset_plan.total_tokens
    group_sizes = offset_plan.group_sizes
    if not offset_plan.groups_are_block_aligned(block_size):
        raise NotImplementedError(
            "DeepGEMM K-grouped FP8 wgrad requires every expert token count "
            f"to be divisible by {block_size}. Enable "
            "pad_token_groups_for_grouped_mm or select KernelPreference.EMULATED."
        )

    k_grouped_gemm = capabilities.k_grouped_gemm
    if k_grouped_gemm is None:
        raise ImportError(
            "Installed `deep_gemm` does not expose "
            "`k_grouped_fp8_gemm_nt_contiguous`. Install a DeepGEMM version "
            "with contiguous K-grouped FP8 GEMM support."
        )

    valid_scale_blocks = offset_plan.valid_scale_blocks(block_size)
    a, a_s = _prepare_k_grouped_lhs_operand(
        lhs,
        group_sizes,
        total_tokens,
        valid_scale_blocks,
    )
    b, b_s = _prepare_k_grouped_rhs_operand(
        rhs,
        group_sizes,
        total_tokens,
        valid_scale_blocks,
    )
    assert a.numel() == total_tokens * lhs.dim, (
        f"shape {a.shape} and LHS scales {a_s.shape} are not compatible"
    )
    assert b.numel() == total_tokens * rhs.dim, (
        f"shape {b.shape} and RHS scales {b_s.shape} are not compatible"
    )
    ks_tensor = offset_plan.ks_tensor

    # SM90 DeepGEMM K-grouped FP8 always runs with accumulation: the kernel
    # hard-asserts `c.has_value()` and a float output (see
    # sm90_k_grouped_fp8_gemm_1d1d in DeepGEMM), computing `d = c + A@B`. `c`
    # (`accum`) is read and `d` (`out_fp32`) is written, so they must be
    # distinct buffers and `c` must be zero-seeded since we do not accumulate
    # across calls. Both FP32 allocations are therefore required by the kernel
    # contract. Cast after the launch to preserve TorchAO's public grouped-mm
    # output dtype.
    out_fp32 = torch.empty(
        (len(group_sizes), lhs.dim, rhs.dim),
        dtype=torch.float32,
        device=lhs.data.device,
    )
    accum = torch.zeros_like(out_fp32)

    k_grouped_gemm(
        (a, a_s),
        (b, b_s),
        out_fp32,
        group_sizes,
        ks_tensor,
        accum,
        recipe=(1, 1, block_size),
    )
    return out_fp32 if out_dtype == torch.float32 else out_fp32.to(out_dtype)
