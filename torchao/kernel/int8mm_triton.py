# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
import triton
import triton.language as tl

# Block sizes for tiling (must be multiples of 16 for Tensor Cores)
# TODO: Autotuning for dynamic block size selection
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


@triton.jit
def int8_scaled_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    scale_a_ptr,
    scale_b_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    INT8 Scaled Matrix Multiplication: C = (A @ B) * scale_a * scale_b

    Note: This kernel doesn't support following optimizations yet.
    TODO: Swizzling (GROUP_M) - L2 cache optimization
    TODO: SplitK - K-dimension parallelization for small M (decode)

    See torchao/kernel/intmm_triton.py for reference.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointers for A[M,K] and B[K,N]
    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # Accumulator (int32 to prevent overflow)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # K-loop: accumulate partial results
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Load scales
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale_a = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Apply scales in fp32 (prevents overflow), output as fp16
    acc_fp = acc.to(tl.float32)
    scale_a_fp32 = scale_a.to(tl.float32)[:, None]
    scale_b_fp32 = scale_b.to(tl.float32)[None, :]
    result = acc_fp * scale_a_fp32 * scale_b_fp32
    result = result.to(tl.float16)

    # Store result
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, result, boundary_check=(0, 1))


@torch.library.custom_op("torchao::int8_scaled_matmul", mutates_args=())
def int8_scaled_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    INT8 Scaled Matrix Multiplication: C = (A @ B) * scale_a * scale_b

    Args:
        A: [M, K] int8 tensor
        B: [K, N] int8 tensor
        scale_a: [M] or [M, 1] scale for A (fp16/fp32)
        scale_b: [N] or [1, N] scale for B (fp16/fp32)

    Returns:
        C: [M, N] fp16
    """
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    N = B.shape[1]

    A = A.contiguous()
    B = B.contiguous()
    scale_a = scale_a.view(-1).contiguous()
    scale_b = scale_b.view(-1).contiguous()

    # Extend scales if per-tensor
    if scale_a.numel() == 1:
        scale_a = scale_a.expand(M).contiguous()
    if scale_b.numel() == 1:
        scale_b = scale_b.expand(N).contiguous()

    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    int8_scaled_matmul_kernel[grid](
        A,
        B,
        C,
        scale_a,
        scale_b,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


@int8_scaled_matmul.register_fake
def _(A, B, scale_a, scale_b):
    """Dimension only implementation for torch.compile."""
    M, K = A.shape
    N = B.shape[1]
    return torch.empty((M, N), device=A.device, dtype=torch.float16)
