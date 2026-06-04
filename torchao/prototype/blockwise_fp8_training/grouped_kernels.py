# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    FP8_E4M3_DTYPES,
    _is_column_major,
    _is_row_major,
    _prepare_blockwise_scaled_mm_rhs_scale,
    _scaling_type_value,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)
from torchao.utils import ceil_div


def _prepare_grouped_128x128_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        return _prepare_blockwise_scaled_mm_rhs_scale(
            scale, BLOCKWISE_128X128_SCALING_TYPE
        )
    assert scale.ndim == 3, "expected 2D or 3D scale tensor"
    padded_k_blocks = ceil_div(scale.shape[-2], 4) * 4
    if padded_k_blocks == scale.shape[-2]:
        return scale

    padded = scale.new_full(
        (scale.shape[0], padded_k_blocks, scale.shape[-1]), 1.0
    ).as_strided(
        (scale.shape[0], padded_k_blocks, scale.shape[-1]),
        (padded_k_blocks * scale.shape[-1], 1, padded_k_blocks),
    )
    padded[:, : scale.shape[-2], :] = scale
    return padded


def _prepare_grouped_rhs_scale(
    scale: torch.Tensor,
    scale_recipe,
) -> torch.Tensor:
    if _scaling_type_value(scale_recipe) != _scaling_type_value(
        BLOCKWISE_128X128_SCALING_TYPE
    ):
        return scale
    return _prepare_grouped_128x128_scale(scale)


@torch.library.custom_op(
    "torchao::triton_fp8_blockwise_weight_quant_grouped_transposed_rhs",
    mutates_args=(),
)
def triton_fp8_blockwise_weight_quant_grouped_transposed_rhs(
    weight_t: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize expert weights for output = A @ weight_t.

    Input is (E, K, N), normally the transposed expert weight tensor used by
    grouped GEMM. Output data is (E, K, N) in per-expert column-major layout,
    with 128x128 reciprocal scales of shape (E, K // 128, N // 128).
    """
    assert weight_t.ndim == 3, "weight_t must be 3D"
    assert _is_column_major(weight_t), "weight_t must be per-expert column-major"
    assert dtype in FP8_E4M3_DTYPES, f"dtype must be one of {FP8_E4M3_DTYPES}"
    E, K, N = weight_t.shape
    assert K % block_size == 0 and N % block_size == 0, (
        f"weight_t K and N must be divisible by block_size={block_size}"
    )

    q_out = torch.empty_strided(
        (E, K, N),
        (K * N, 1, K),
        dtype=dtype,
        device=weight_t.device,
    )
    scale_out = torch.empty_strided(
        (E, K // block_size, N // block_size),
        ((K // block_size) * (N // block_size), 1, K // block_size),
        dtype=torch.float32,
        device=weight_t.device,
    )
    # NOTE: intentionally done per expert for first pass functionality
    # we use a known correct dense quantization kernel
    # this will be replaced with a native grouped quant kernel without
    # changing the MoE frontend
    for expert_idx in range(E):
        expert_weight = weight_t[expert_idx].transpose(-2, -1).contiguous()
        q, scale = triton_fp8_blockwise_weight_quant_transposed_rhs(
            expert_weight,
            block_size=block_size,
            dtype=dtype,
        )
        q_out[expert_idx].copy_(q)
        scale_out[expert_idx].copy_(scale)
    return q_out, scale_out


@triton_fp8_blockwise_weight_quant_grouped_transposed_rhs.register_fake
def _(
    weight_t: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    E, K, N = weight_t.shape
    q_out = torch.empty_strided(
        (E, K, N),
        (K * N, 1, K),
        dtype=dtype,
        device=weight_t.device,
    )
    scale_out = torch.empty_strided(
        (E, K // block_size, N // block_size),
        ((K // block_size) * (N // block_size), 1, K // block_size),
        dtype=torch.float32,
        device=weight_t.device,
    )
    return q_out, scale_out


@torch.library.custom_op(
    "torchao::triton_fp8_blockwise_weight_quant_grouped_rhs",
    mutates_args=(),
)
def triton_fp8_blockwise_weight_quant_grouped_rhs(
    weight_t: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize expert weights for dgrad = grad_output @ weight.

    Input is (E, K, N), normally the transposed expert weight tensor saved from
    forward. Output data is (E, N, K) in per-expert column-major layout, with
    128x128 reciprocal scales of shape (E, N // 128, K // 128).
    """
    assert weight_t.ndim == 3, "weight_t must be 3D"
    assert _is_column_major(weight_t), "weight_t must be per-expert column-major"
    assert dtype in FP8_E4M3_DTYPES, f"dtype must be one of {FP8_E4M3_DTYPES}"
    E, K, N = weight_t.shape
    assert K % block_size == 0 and N % block_size == 0, (
        f"weight_t K and N must be divisible by block_size={block_size}"
    )

    q_out = torch.empty_strided(
        (E, N, K),
        (N * K, 1, N),
        dtype=dtype,
        device=weight_t.device,
    )
    scale_out = torch.empty_strided(
        (E, N // block_size, K // block_size),
        ((N // block_size) * (K // block_size), 1, N // block_size),
        dtype=torch.float32,
        device=weight_t.device,
    )
    # NOTE: intentionally done per expert for first pass functionality
    # we use a known correct dense quantization kernel
    # this will be replaced with a native grouped quant kernel without
    # changing the MoE frontend
    for expert_idx in range(E):
        expert_weight = weight_t[expert_idx].transpose(-2, -1).contiguous()
        q, scale = triton_fp8_blockwise_weight_quant_rhs(
            expert_weight,
            block_size=block_size,
            dtype=dtype,
        )
        q_out[expert_idx].copy_(q)
        scale_out[expert_idx].copy_(scale)
    return q_out, scale_out


@triton_fp8_blockwise_weight_quant_grouped_rhs.register_fake
def _(
    weight_t: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    E, K, N = weight_t.shape
    q_out = torch.empty_strided(
        (E, N, K),
        (N * K, 1, N),
        dtype=dtype,
        device=weight_t.device,
    )
    scale_out = torch.empty_strided(
        (E, N // block_size, K // block_size),
        ((N // block_size) * (K // block_size), 1, N // block_size),
        dtype=torch.float32,
        device=weight_t.device,
    )
    return q_out, scale_out


# NOTE: only for emulated backend, we can remove once we have native scaled grouped gemm
def _expand_blockwise_scale(
    q_data: torch.Tensor,
    scale: torch.Tensor,
    scale_recipe: int,
    block_size: int,
) -> torch.Tensor:
    if _scaling_type_value(scale_recipe) == _scaling_type_value(
        BLOCKWISE_128X128_SCALING_TYPE
    ):
        scale = scale[..., : ceil_div(q_data.shape[-2], block_size), :]
        return scale.repeat_interleave(block_size, dim=-2).repeat_interleave(
            block_size, dim=-1
        )

    assert _scaling_type_value(scale_recipe) == _scaling_type_value(
        BLOCKWISE_1X128_SCALING_TYPE
    ), f"unsupported scale recipe: {scale_recipe}"
    if scale.shape[-2] == q_data.shape[-2]:
        return scale.repeat_interleave(block_size, dim=-1)
    if scale.shape[-1] == q_data.shape[-1]:
        return scale.repeat_interleave(block_size, dim=-2)
    raise AssertionError(
        f"Could not infer 1x128 scale orientation for q_data={q_data.shape}, scale={scale.shape}"
    )


# NOTE: only for emulated backend, we can remove once we have native scaled grouped gemm
# for correctness not performance
def _emulated_blockwise_scaled_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    a_scale = _expand_blockwise_scale(a, a_s, scale_recipe_a, block_size)
    b_scale = _expand_blockwise_scale(b, b_s, scale_recipe_b, block_size)
    a_hp = a.to(torch.bfloat16) * a_scale.to(torch.bfloat16)
    b_hp = b.to(torch.bfloat16) * b_scale.to(torch.bfloat16)
    return torch._grouped_mm(a_hp, b_hp, offs=offs, out_dtype=out_dtype)


@torch.library.custom_op("torchao::blockwise_scaled_grouped_mm", mutates_args=())
def blockwise_scaled_grouped_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    assert _is_row_major(a), "blockwise_scaled_grouped_mm expected row-major A"
    assert _is_column_major(b), "blockwise_scaled_grouped_mm expected column-major B"
    assert offs is not None and offs.dtype == torch.int32, "offs must be int32"
    b_s = _prepare_grouped_rhs_scale(b_s, scale_recipe_b)
    # TODO(future): hook up F.scaled_grouped_mm once it supports float blockwise.
    return _emulated_blockwise_scaled_grouped_mm(
        a,
        b,
        a_s,
        scale_recipe_a,
        b_s,
        scale_recipe_b,
        offs,
        out_dtype,
        block_size,
    )


@blockwise_scaled_grouped_mm.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    scale_recipe_a: int,
    b_s: torch.Tensor,
    scale_recipe_b: int,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int = 128,
) -> torch.Tensor:
    if b.ndim == 3:
        return a.new_empty((a.shape[0], b.shape[-1]), dtype=out_dtype)
    return a.new_empty((offs.numel(), a.shape[0], b.shape[-1]), dtype=out_dtype)
