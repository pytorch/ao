# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
from dataclasses import dataclass
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
)


@dataclass(frozen=True)
class DeepGemmKGroupedOperand:
    """A quantized operand in DeepGEMM's flat K-grouped layout.

    Args:
        data: Flat FP8 data with each expert stored as ``(dim, expert_tokens)``.
        scale: Float32 inverse scales for the operand.
        dim: Logical non-token dimension used by DeepGEMM's K-grouped API.
    """

    data: torch.Tensor
    scale: torch.Tensor
    dim: int


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


def _quantize_wgrad_operand(
    x: torch.Tensor,
    group_end_offsets: torch.Tensor,
    group_sizes: list[int],
    block_size: int,
    dtype: torch.dtype,
    metadata: DeepGemmKGroupedQuantMetadata,
) -> DeepGemmKGroupedOperand:
    q, scale = _triton_fp8_blockwise_act_quant_k_grouped_deepgemm_with_group_sizes(
        x.contiguous(),
        group_end_offsets,
        group_sizes,
        block_size=block_size,
        dtype=dtype,
        metadata=metadata,
    )
    assert q.ndim == 1, "DeepGEMM flat K-grouped data must be 1D"
    assert scale.ndim == 2, "DeepGEMM flat K-grouped scale must be 2D"
    return DeepGemmKGroupedOperand(q, scale, scale.shape[0])


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
    lhs_metadata = offset_plan.k_quant_metadata(
        block_size, padded_grad_output.shape[-1]
    )
    rhs_metadata = offset_plan.k_quant_metadata(block_size, padded_a.shape[-1])
    return DeepGemmWgradPlan(
        lhs=_quantize_wgrad_operand(
            padded_grad_output,
            offset_plan.group_end_offsets,
            group_sizes,
            block_size,
            dtype,
            lhs_metadata,
        ),
        rhs=_quantize_wgrad_operand(
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
    a, a_s = lhs.data, lhs.scale[:, :valid_scale_blocks]
    b, b_s = rhs.data, rhs.scale[:, :valid_scale_blocks]
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
