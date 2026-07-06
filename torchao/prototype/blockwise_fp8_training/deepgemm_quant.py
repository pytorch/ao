# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    DeepGemmKGroupedQuantMetadata,
    build_deepgemm_k_grouped_quant_metadata,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    group_sizes_from_offsets as _group_sizes_from_offsets,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    EPS,
    FP8_E4M3_DTYPES,
    quant_kernel_configs_with_groups,
)


def _should_use_traceable_triton_launch() -> bool:
    return torch.compiler.is_compiling()


def _triton_launcher(kernel, traceable: bool):
    return wrap_triton(kernel) if traceable else kernel


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["D"])
@triton.jit
def triton_fp8_blockwise_act_quant_k_grouped_deepgemm_kernel(
    x_ptr,
    x_stride_m,
    x_stride_d,
    group_end_offsets_ptr,
    q_ptr,
    s_ptr,
    s_stride_d,
    s_stride_block,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_group = tl.program_id(axis=0)
    pid_group_block = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)

    group_end = tl.load(group_end_offsets_ptr + pid_group)
    group_start = tl.load(
        group_end_offsets_ptr + pid_group - 1,
        mask=pid_group > 0,
        other=0,
    )
    group_size = group_end - group_start

    offs_m = pid_group_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = pid_d * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    valid_group_block = pid_group_block * BLOCK_SIZE < group_size

    x_offsets = (group_start + offs_m[:, None]) * x_stride_m + offs_d[
        None, :
    ] * x_stride_d
    # Group sizes are asserted to be multiples of BLOCK_SIZE before launch.
    # Within a valid group block every row is live, so only mask off invalid
    # blocks from shorter experts and the tail of the D dimension.
    x_mask = valid_group_block & (offs_d[None, :] < D)
    x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    # Compute one 1x128 scale per logical column over this expert's token
    # block, matching the existing transposed-LHS/RHS wgrad quantizers.
    amax = tl.clamp(tl.max(tl.abs(x), axis=0), min=EPS, max=float("inf")).to(tl.float64)
    scale = (FP8_MAX / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=-FP8_MAX, max=FP8_MAX).to(q_ptr.dtype.element_ty)

    # DeepGEMM K-grouped K-major buffers are concatenated per expert. For an
    # expert with `group_size` tokens and logical dimension D, its flat slice is
    # shaped as (D, group_size): q[group_start * D + d * group_size + local_m].
    q_offsets = group_start * D + offs_d[:, None] * group_size + offs_m[None, :]
    q_mask = valid_group_block & (offs_d[:, None] < D)
    tl.store(q_ptr + q_offsets, y.trans(1, 0), mask=q_mask)

    # Scales are concatenated along token-blocks in the same expert order:
    # (D, total_valid_token_blocks).
    block_idx = group_start // BLOCK_SIZE + pid_group_block
    s_offsets = offs_d * s_stride_d + block_idx * s_stride_block
    s_mask = (offs_d < D) & valid_group_block
    tl.store(s_ptr + s_offsets, tl.div_rn(1.0, scale), mask=s_mask)


@triton.autotune(configs=quant_kernel_configs_with_groups, key=["D"])
@triton.jit
def triton_fp8_blockwise_act_quant_k_grouped_compact_deepgemm_kernel(
    x_ptr,
    q_offset_by_block_ptr,
    group_size_by_block_ptr,
    q_ptr,
    s_ptr,
    D: tl.constexpr,
    VALID_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_block = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)

    q_offset = tl.load(q_offset_by_block_ptr + pid_block)
    group_size = tl.load(group_size_by_block_ptr + pid_block)
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_d = pid_d * NUM_GROUPS + tl.arange(0, NUM_GROUPS)

    # Dense block metadata keeps the kernel shape independent of expert count
    # and routing skew: one program per valid token block, no empty expert
    # blocks, and no source-level specialization for a particular topology.
    x_offsets = (pid_block * BLOCK_SIZE + offs_m[:, None]) * D + offs_d[None, :]
    x = tl.load(x_ptr + x_offsets)

    amax = tl.clamp(tl.max(tl.abs(x), axis=0), min=EPS, max=float("inf")).to(tl.float64)
    scale = (FP8_MAX / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=-FP8_MAX, max=FP8_MAX).to(q_ptr.dtype.element_ty)

    # `q_offset` is the per-block base in DeepGEMM's flat per-expert
    # (D, expert_tokens) output. Precomputing it keeps this compact kernel
    # metadata-driven without paying a group_start * D multiply in every tile.
    q_offsets = q_offset + offs_d[:, None] * group_size + offs_m[None, :]
    # The compact path only launches when D % 128 == 0; every NUM_GROUPS value
    # divides 128, so offs_d never runs past D and these stores need no D mask.
    tl.store(q_ptr + q_offsets, y.trans(1, 0))

    s_offsets = offs_d * VALID_BLOCKS + pid_block
    tl.store(s_ptr + s_offsets, tl.div_rn(1.0, scale))


@torch.library.custom_op(
    "torchao::triton_fp8_blockwise_act_quant_k_grouped_deepgemm",
    mutates_args=(),
)
def triton_fp8_blockwise_act_quant_k_grouped_deepgemm(
    x: torch.Tensor,
    group_end_offsets: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations directly for DeepGEMM K-grouped wgrad.

    Input is logical (M_tokens, D), already padded/grouped by expert. Output is
    a flat K-major DeepGEMM buffer: for each expert, data is stored as
    (D, expert_tokens), then concatenated with the next expert. Scales are
    stored as (D, total_valid_token_blocks).

    The fake implementation treats the number of valid tokens as dynamic
    because it cannot read ``group_end_offsets`` values from FakeTensors.
    """
    group_sizes = _group_sizes_from_offsets(group_end_offsets)
    return _triton_fp8_blockwise_act_quant_k_grouped_deepgemm_with_group_sizes(
        x,
        group_end_offsets,
        group_sizes,
        block_size,
        dtype,
    )


def _triton_fp8_blockwise_act_quant_k_grouped_deepgemm_with_group_sizes(
    x: torch.Tensor,
    group_end_offsets: torch.Tensor,
    group_sizes: list[int],
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
    traceable: bool | None = None,
    metadata: DeepGemmKGroupedQuantMetadata | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, "x must be 2D"
    assert x.is_contiguous(), "x must be contiguous"
    assert group_end_offsets is not None and group_end_offsets.dtype == torch.int32, (
        "group_end_offsets must be int32"
    )
    assert dtype in FP8_E4M3_DTYPES, f"dtype must be one of {FP8_E4M3_DTYPES}"
    valid_tokens = sum(group_sizes)
    assert group_end_offsets.numel() == len(group_sizes), (
        "group_sizes must have one entry per group"
    )
    assert x.shape[0] >= valid_tokens, (
        f"x has {x.shape[0]} rows but offsets require {valid_tokens}"
    )
    assert all(group_size % block_size == 0 for group_size in group_sizes), (
        "DeepGEMM K-grouped activation quantization requires every group size "
        f"to be divisible by block_size={block_size}"
    )

    D = x.shape[1]
    valid_blocks = valid_tokens // block_size
    q_out = torch.empty((valid_tokens * D,), dtype=dtype, device=x.device)
    scale_out = torch.empty((D, valid_blocks), dtype=torch.float32, device=x.device)
    max_group_blocks = max(group_sizes) // block_size if group_sizes else 0
    traceable = (
        _should_use_traceable_triton_launch() if traceable is None else traceable
    )

    if group_sizes and D % 128 == 0:
        if metadata is None:
            metadata = build_deepgemm_k_grouped_quant_metadata(
                group_end_offsets,
                group_sizes,
                block_size,
                D,
            )
        assert metadata.dim == D, (
            f"DeepGEMM K-grouped metadata was built for dim={metadata.dim}, "
            f"but input dim is {D}"
        )

        def compact_grid(meta):
            return (
                valid_blocks,
                triton.cdiv(D, meta["NUM_GROUPS"]),
            )

        _triton_launcher(
            triton_fp8_blockwise_act_quant_k_grouped_compact_deepgemm_kernel,
            traceable=traceable,
        )[compact_grid](
            x,
            metadata.q_offset_by_block,
            metadata.group_size_by_block,
            q_out,
            scale_out,
            D=D,
            VALID_BLOCKS=valid_blocks,
            BLOCK_SIZE=block_size,
            EPS=EPS,
            FP8_MAX=torch.finfo(dtype).max,
        )
        return q_out, scale_out

    def grid(meta):
        return (
            group_end_offsets.numel(),
            max_group_blocks,
            triton.cdiv(D, meta["NUM_GROUPS"]),
        )

    _triton_launcher(
        triton_fp8_blockwise_act_quant_k_grouped_deepgemm_kernel,
        traceable=traceable,
    )[grid](
        x,
        x.stride(0),
        x.stride(1),
        group_end_offsets,
        q_out,
        scale_out,
        scale_out.stride(0),
        scale_out.stride(1),
        D=D,
        BLOCK_SIZE=block_size,
        EPS=EPS,
        FP8_MAX=torch.finfo(dtype).max,
    )
    return q_out, scale_out


@triton_fp8_blockwise_act_quant_k_grouped_deepgemm.register_fake
def _(
    x: torch.Tensor,
    group_end_offsets: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ctx = torch.library.get_ctx()
    max_valid_tokens = x.shape[0] if type(x.shape[0]) is int else None
    valid_tokens = (
        ctx.new_dynamic_size(min=0, max=max_valid_tokens)
        if max_valid_tokens is not None
        else ctx.new_dynamic_size(min=0)
    )
    torch._check(valid_tokens <= x.shape[0])
    torch._check(valid_tokens % block_size == 0)
    valid_blocks = valid_tokens // block_size
    return (
        torch.empty((valid_tokens * x.shape[1],), dtype=dtype, device=x.device),
        torch.empty(
            (x.shape[1], valid_blocks),
            dtype=torch.float32,
            device=x.device,
        ),
    )
