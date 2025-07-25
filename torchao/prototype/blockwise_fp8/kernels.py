# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton import Config

# try:
#     from fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm import triton_quantize_fp8_block
# except ImportError:
#     print("Please install fbgemm-gpu to use this feature")
#     sys.exit(1)

# Original implementation at https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py

fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64, 128]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K", "M_BUCKET", "BLOCK_SIZE_K"])
@triton.jit
def blockwise_fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    M_BUCKET: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def blockwise_fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    M_BUCKET = math.ceil(math.log2(M))
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    blockwise_fp8_gemm_kernel[grid](
        a, b, c, a_s, b_s, M, N, K, M_BUCKET, BLOCK_SIZE_K=block_size
    )
    return c


@triton.jit
def fp8_blockwise_act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def fp8_blockwise_act_quant(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with block size being BLOCK_SIZEx1.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        dtype (torch.dtype, optional): The dtype to use for the quantized tensor. Default is `torch.float8_e4m3fn`.


    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `dtype`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    fp8_blockwise_act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def fp8_blockwise_weight_quant_kernel(
    x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factors in `s_ptr`.

    Args:
        x_ptr (tl.pointer): Pointer to the input tensor.
        y_ptr (tl.pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (tl.pointer): Pointer to the output tensor where scaling factors will be stored.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def fp8_blockwise_weight_quant(
    x: torch.Tensor, block_size: int = 128, dtype=torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the given weight tensor using block-wise quantization with block size being BLOCK_SIZExBLOCK_SIZE.

    Args:
        x (torch.Tensor): The weight tensor to be quantized.
        block_size (int, optional): The block size to use for quantization. Defaults to 128.
        dtype (torch.dtype, optional): The dtype to use for the quantized tensor. Defaults to `torch.float8_e4m3fn`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized weight tensor with dtype `dtype`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.dim() == 2, "Input tensor must have 2 dimensions"
    assert x.size(0) % block_size == 0 and x.size(1) % block_size == 0, (
        f"Both dimensions of x must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "dtype must be torch.float8_e4m3fn or torch.float8_e5m2"
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    s = x.new_empty(M // block_size, N // block_size, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    fp8_blockwise_weight_quant_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def fp8_blockwise_weight_dequant_kernel(
    x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def fp8_blockwise_weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    fp8_blockwise_weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# original implementation from fbgemm_gpu:
# https://github.com/pytorch/FBGEMM/blob/b19401e913fcdff536dc097fa3013a0a9d66256e/fbgemm_gpu/experimental/gemm/triton_gemm/fp8_gemm.py#L3091
def triton_quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 128,
    block_k: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor : [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block
        if k_major is True, otherwise [cdiv(K, block_k), cdiv(M, block_M)].
    """
    assert x.device != torch.device("cpu"), (
        "Blockwise quantization not support on cpu, please use row-wise quantization instead."
    )
    pt_dtype = torch.float8_e4m3fn
    tl_dtype = tl.float8e4nv
    max_fp8 = torch.finfo(pt_dtype).max
    eps = 1e-12

    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)
    if k_major:
        x_scale = torch.empty((grid_m, grid_k), device=x.device, dtype=torch.float32)
    else:
        x_scale = torch.empty((grid_k, grid_m), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=pt_dtype)

    _kernel_quantize_fp8_block[(grid_m * grid_k,)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        K_MAJOR=k_major,
    )

    return x_fp8.view(x_shape), x_scale


# original implementation from fbgemm_gpu:
# https://github.com/pytorch/FBGEMM/blob/b19401e913fcdff536dc097fa3013a0a9d66256e/fbgemm_gpu/experimental/gemm/triton_gemm/fp8_gemm.py#L3005
@triton.jit
def _kernel_quantize_fp8_block(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    M,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    stride_a_scale_m,
    stride_a_scale_k,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_MAJOR: tl.constexpr,
) -> None:
    """Quantize and scale each [BLOCK_M, BLOCK_K] block.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(A[i:i+BLOCK_M, j:j+BLOCK_K])))

    Kernel naively iterates through  matrix with [BLOCK_M, BLOCK_K] tiles.

    Todo:
        * Better tiling and ordering schemes.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [cdiv(M, BLOCK_M), cdiv(K, BLOCK_K)] reciprocal scale tensor per block.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a_scale
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
        BLOCK_M (int): Block size for M dimension of A_scale and kernel.
        BLOCK_K (int): Block size for K dimension of A_scale and kernel.
        K_MAJOR (bool): Whether output scales should be K major (True) or MN major (False).
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_offset = rm[:, None] * stride_am + rk[None, :] * stride_ak
    out_offset = rm[:, None] * stride_om + rk[None, :] * stride_ok
    a_mask = (rm < M)[:, None] & (rk < K)[None, :]
    a_block = tl.load(A + a_offset, mask=a_mask, other=0.0)

    block_max = tl.max(tl.abs(a_block))
    # Apply appropriate clamping.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        block_max = tl.clamp(block_max, EPS, ub)
    else:
        block_max = tl.maximum(block_max, EPS)
    scale = MAX_FP8 / block_max

    # Write in transposed order if specified.
    if K_MAJOR:
        scale_offset = block_m * stride_a_scale_m + block_k * stride_a_scale_k
    else:
        scale_offset = block_k * stride_a_scale_m + block_m * stride_a_scale_k
    tl.store(A_scale + scale_offset, 1.0 / scale)
    a_fp8 = a_block * scale
    # Clamp A to fp8 range to make sure there's no overflow.
    # This is required for AMD. Nvidia's default saturation
    # handles it, but it's nice to have anyway.
    a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
    a_fp8.to(TL_FP8_DTYPE)
    tl.store(A_fp8 + out_offset, a_fp8, mask=a_mask)


def torch_blockwise_scale_act_quant(x, tile_size=128):
    """
    Input: weight tensor in high precision
    Output: weight tensor in float8, and scale, tiled 1 by tile_size
    """
    assert x.is_contiguous(), "input tensor must be contiguous"
    orig_shape = x.shape

    # Reshape 2D+ input tensor into 2D tensor with shape (leading_dims, tile_size)
    x = x.reshape(-1, tile_size)

    # Compute amax along last dim (i.e., the block)
    x_amax = x.abs().max(dim=1).values.unsqueeze(1).clamp(1e-4)

    # Convert amax to scale
    fp8_dtype_max, fp8_dtype_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )
    s = fp8_dtype_max / x_amax

    # Apply scale and clamp
    x = (x * s).clamp(min=fp8_dtype_min, max=fp8_dtype_max).to(torch.float8_e4m3fn)

    # Reshape quantized output back to original shape and reshape scales accordingly
    x = x.reshape(*orig_shape)
    s = s.reshape(orig_shape[0], -1).to(torch.float)
    return x, s


def torch_blockwise_scale_weight_quant(x, tile_size=128):
    """
    Input: weight tensor in high precision
    Output: weight tensor in float8, and scale, tiled tile_size by tile_size
    """
    assert len(x.shape) == 2, "input shape must be 2D"
    assert x.is_contiguous(), "input tensor must be contiguous"
    height, width = x.shape

    # Compute block sizes
    t_h = height // tile_size
    t_w = width // tile_size

    # Reshape 2D input tensor into 4D tensor with shape (t_h, t_w, tile_size * tile_size)
    x = x.reshape(t_h, tile_size, t_w, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, tile_size * tile_size)

    # Compute amax along last dim (i.e., the block)
    m = x.abs().max(dim=1).values.unsqueeze(1).clamp(1e-4)

    # Convert amax to scale
    fp8_dtype_max, fp8_dtype_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )
    s = fp8_dtype_max / m

    # Apply scale and clamp
    x = (x * s).clamp(min=fp8_dtype_min, max=fp8_dtype_max).to(torch.float8_e4m3fn)

    # Reshape quantized output and scales back to 2D
    x = x.reshape(t_h, t_w, tile_size, tile_size)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(height, width)
    s = s.reshape(t_h, t_w).to(torch.float)
    return x, s
