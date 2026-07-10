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
    x_stride_e,
    x_stride_k,
    x_stride_n,
    q_ptr,
    s_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_en_block = tl.program_id(axis=0)
    pid_k_block = tl.program_id(axis=1)

    n_blocks = N // BLOCK_SIZE
    expert_idx = pid_en_block // n_blocks
    pid_n_block = pid_en_block - expert_idx * n_blocks

    offs_n = pid_n_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = pid_k_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Read logical (E, K, N) input as an (N, K) tile for one expert. Explicit
    # strides preserve per-expert column-major inputs with a non-packed E axis.
    x_offsets = (
        expert_idx * x_stride_e
        + offs_k[None, :] * x_stride_k
        + offs_n[:, None] * x_stride_n
    )
    x = tl.load(x_ptr + x_offsets).to(tl.float32)

    amax = tl.clamp(tl.max(tl.abs(x)), min=EPS, max=float("inf")).to(tl.float64)
    scale = (FP8_MAX / amax).to(tl.float32)
    y = x * scale
    y = tl.clamp(y, min=-FP8_MAX, max=FP8_MAX).to(q_ptr.dtype.element_ty)

    q_offsets = (expert_idx * N + offs_n[:, None]) * K + offs_k[None, :]
    tl.store(q_ptr + q_offsets, y)
    tl.store(
        s_ptr + pid_en_block * (K // BLOCK_SIZE) + pid_k_block,
        tl.div_rn(1.0, scale),
    )


@triton.autotune(configs=quant_kernel_configs, key=["N", "K"])
@triton.jit
def triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs_kernel(
    x_ptr,
    x_stride_e,
    x_stride_k,
    x_stride_n,
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
    # Read the same logical (E, K, N) input tile, then store it transposed into
    # the row-major dgrad RHS contract (E, K, N).
    x_offsets = (
        expert_idx * x_stride_e
        + offs_k[None, :] * x_stride_k
        + offs_n[:, None] * x_stride_n
    )
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

    assert weight_t.stride(-2) == 1, "weight_t must be per-expert column-major"
    wrap_triton(triton_fp8_blockwise_weight_quant_grouped_forward_rhs_kernel)[
        (E * (N // block_size), K // block_size)
    ](
        weight_t,
        weight_t.stride(0),
        weight_t.stride(1),
        weight_t.stride(2),
        q_out.reshape(E * N, K),
        scale_out.reshape(E * (N // block_size), K // block_size),
        M=E * N,
        N=N,
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

    assert weight_t.stride(-2) == 1, "weight_t must be per-expert column-major"
    wrap_triton(triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs_kernel)[
        (E * (N // block_size), K // block_size)
    ](
        weight_t,
        weight_t.stride(0),
        weight_t.stride(1),
        weight_t.stride(2),
        q_out,
        scale_out,
        N=N,
        K=K,
        BLOCK_SIZE=block_size,
        EPS=EPS,
        FP8_MAX=torch.finfo(dtype).max,
    )
    return q_out, scale_out
