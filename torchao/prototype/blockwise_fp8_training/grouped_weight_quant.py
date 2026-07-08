# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.kernels import (
    EPS,
    FP8_E4M3_DTYPES,
    quant_kernel_configs,
)


@triton.autotune(configs=quant_kernel_configs, key=["M", "K"])
@triton.jit
def triton_fp8_blockwise_weight_quant_grouped_forward_rhs_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m_block = tl.program_id(axis=0)
    pid_k_block = tl.program_id(axis=1)

    offs_m = pid_m_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = pid_k_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # The Python wrapper exposes the original row-major expert weights as a
    # contiguous (E*N, K) view. N is block-aligned, so every row block stays
    # inside one expert.
    x_offsets = offs_m[:, None] * K + offs_k[None, :]
    x = tl.load(x_ptr + x_offsets).to(tl.float32)

    amax = tl.clamp(tl.max(tl.abs(x)), min=EPS, max=float("inf")).to(tl.float64)
    scale = (FP8_MAX / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=-FP8_MAX, max=FP8_MAX).to(q_ptr.dtype.element_ty)

    tl.store(q_ptr + x_offsets, y)
    tl.store(
        s_ptr + pid_m_block * (K // BLOCK_SIZE) + pid_k_block,
        tl.div_rn(1.0, scale),
    )


@triton.autotune(configs=quant_kernel_configs, key=["N", "K"])
@triton.jit
def triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_en_block = tl.program_id(axis=0)
    pid_k_block = tl.program_id(axis=1)

    n_blocks = N // BLOCK_SIZE
    k_blocks = K // BLOCK_SIZE
    expert_idx = pid_en_block // n_blocks
    pid_n_block = pid_en_block - expert_idx * n_blocks

    offs_n = pid_n_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = pid_k_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_m = pid_en_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Read the contiguous forward-weight view as (E*N, K), then transpose each
    # quantized 128x128 tile into the dgrad RHS layout (E, K, N).
    x_offsets = offs_m[:, None] * K + offs_k[None, :]
    x = tl.load(x_ptr + x_offsets).to(tl.float32)

    amax = tl.clamp(tl.max(tl.abs(x)), min=EPS, max=float("inf")).to(tl.float64)
    scale = (FP8_MAX / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=-FP8_MAX, max=FP8_MAX).to(q_ptr.dtype.element_ty)

    q_offsets = expert_idx * K * N + offs_k[:, None] * N + offs_n[None, :]
    tl.store(q_ptr + q_offsets, y.trans(1, 0))

    s_offset = expert_idx * k_blocks * n_blocks + pid_k_block * n_blocks + pid_n_block
    tl.store(s_ptr + s_offset, tl.div_rn(1.0, scale))


@triton_op(
    "torchao::triton_fp8_blockwise_weight_quant_grouped_forward_rhs",
    mutates_args={},
)
def triton_fp8_blockwise_weight_quant_grouped_forward_rhs(
    weight_t: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize expert weights for forward grouped GEMM.

    Input is TorchAO's public (E, K, N) transposed expert weight tensor.
    Output data is the direct RHS layout (E, N, K), row-major with contiguous
    K, and scales are (E, N // 128, K // 128).
    """
    assert weight_t.ndim == 3, "weight_t must be 3D"
    assert dtype in FP8_E4M3_DTYPES, f"dtype must be one of {FP8_E4M3_DTYPES}"
    E, K, N = weight_t.shape
    assert K % block_size == 0 and N % block_size == 0, (
        f"weight_t K and N must be divisible by block_size={block_size}"
    )

    q_out = torch.empty((E, N, K), dtype=dtype, device=weight_t.device)
    scale_out = torch.empty(
        (E, N // block_size, K // block_size),
        dtype=torch.float32,
        device=weight_t.device,
    )

    # `weight_t.transpose(-2, -1)` is a view back to the original row-major
    # expert weights (E, N, K). Flattening expert and N rows lets the fast
    # dense 2D cast write the forward RHS as (E, N, K) data and
    # (E, N_blocks, K_blocks) scales directly.
    weight = weight_t.transpose(-2, -1)
    assert weight.is_contiguous(), "weight_t must be per-expert column-major"
    wrap_triton(triton_fp8_blockwise_weight_quant_grouped_forward_rhs_kernel)[
        (E * (N // block_size), K // block_size)
    ](
        weight.reshape(E * N, K),
        q_out.reshape(E * N, K),
        scale_out.reshape(E * (N // block_size), K // block_size),
        M=E * N,
        K=K,
        BLOCK_SIZE=block_size,
        EPS=EPS,
        FP8_MAX=torch.finfo(dtype).max,
    )
    return q_out, scale_out


@triton_op(
    "torchao::triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs",
    mutates_args={},
)
def triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs(
    weight_t: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = e4m3_dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize expert weights for dgrad grouped GEMM.

    Input is TorchAO's public (E, K, N) transposed expert weight tensor.
    Output data is the direct RHS layout (E, K, N), row-major with contiguous
    N, and scales are (E, K // 128, N // 128).
    """
    assert weight_t.ndim == 3, "weight_t must be 3D"
    assert dtype in FP8_E4M3_DTYPES, f"dtype must be one of {FP8_E4M3_DTYPES}"
    E, K, N = weight_t.shape
    assert K % block_size == 0 and N % block_size == 0, (
        f"weight_t K and N must be divisible by block_size={block_size}"
    )

    q_out = torch.empty((E, K, N), dtype=dtype, device=weight_t.device)
    scale_out = torch.empty(
        (E, K // block_size, N // block_size),
        dtype=torch.float32,
        device=weight_t.device,
    )

    # Read the same contiguous (E, N, K) forward-weight view used by the fast
    # forward cast, but store each quantized block transposed into the dgrad
    # RHS contract: data (E, K, N), scales (E, K_blocks, N_blocks).
    weight = weight_t.transpose(-2, -1)
    assert weight.is_contiguous(), "weight_t must be per-expert column-major"
    wrap_triton(triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs_kernel)[
        (E * (N // block_size), K // block_size)
    ](
        weight.reshape(E * N, K),
        q_out,
        scale_out,
        N=N,
        K=K,
        BLOCK_SIZE=block_size,
        EPS=EPS,
        FP8_MAX=torch.finfo(dtype).max,
    )
    return q_out, scale_out
