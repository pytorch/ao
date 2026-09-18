# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _is_column_major,
    _is_row_major,
    _prepare_blockwise_scaled_mm_rhs_scale,
    _scaling_type_value,
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


# Expand blockwise scales to match q_data's elementwise shape.
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


def _emulated_blockwise_scaled_grouped_mm_impl(
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


@torch.library.custom_op(
    "torchao::emulated_blockwise_scaled_grouped_mm", mutates_args=()
)
def emulated_blockwise_scaled_grouped_mm(
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
    assert _is_row_major(a), "emulated_blockwise_scaled_grouped_mm expected row-major A"
    assert _is_column_major(b), (
        "emulated_blockwise_scaled_grouped_mm expected column-major B"
    )
    assert offs is not None and offs.dtype == torch.int32, "offs must be int32"
    b_s = _prepare_grouped_rhs_scale(b_s, scale_recipe_b)
    # TODO(future): hook up F.scaled_grouped_mm once it supports float blockwise.
    return _emulated_blockwise_scaled_grouped_mm_impl(
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


@emulated_blockwise_scaled_grouped_mm.register_fake
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
