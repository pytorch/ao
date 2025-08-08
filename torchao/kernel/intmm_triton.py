# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools

import torch
import triton
import triton.language as tl

from torchao.kernel.autotuner import get_best_config_fn

# TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE to enable exhaustive option
int8_mm_kernel_configs = sum(
    [
        # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
        [
            (i, j, k, 1, 1),
            (i, j, k, 1, 2),
            (i, j, k, 2, 2),
            (i, j, k, 1, 4),
            (i, j, k, 2, 4),
            (i, j, k, 3, 4),
            (i, j, k, 4, 4),
            (i, j, k, 1, 8),
            (i, j, k, 2, 8),
            (i, j, k, 3, 8),
            (i, j, k, 4, 8),
            (i, j, k, 5, 8),
            (i, j, k, 6, 8),
            (i, j, k, 7, 8),
            (i, j, k, 8, 8),
        ]
        for (i, j, k) in itertools.product([32, 64, 128, 256], repeat=3)
    ],
    [],
)

if torch._inductor.config.max_autotune_gemm_search_space == "EXHAUSTIVE":
    int8_mm_kernel_configs = [
        (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [16, 32, 64, 128, 256], repeat=3
        )
        for num_stages in [1, 2, 3, 4, 5, 6, 7, 8]
        for num_warps in [2, 4, 8]
    ]


# Baseline configs from pytorch/pytorch
# https://github.com/pytorch/pytorch/blob/7718a1cd4f8e0b794c18a31ebd6353d6273c534e/torch/_inductor/kernel/mm_common.py#L132-L147
# int8_mm_kernel_configs = [
#     (64, 64, 32, 2, 4),
#     (64, 128, 32, 3, 4),
#     (128, 64, 32, 3, 4),
#     (64, 128, 32, 4, 8),
#     (128, 64, 32, 4, 8),
#     (64, 32, 32, 5, 8),
#     (32, 64, 32, 5, 8),
#     (128, 128, 32, 2, 8),
#     (64, 64, 64, 3, 8),
#     (128, 256, 128, 3, 8),
#     (256, 128, 128, 3, 8),
# ]

int8_mm_kernel_configs = [
    triton.Config(
        {"BLOCK_M": i, "BLOCK_N": j, "BLOCK_K": k, "GROUP_M": 8},
        num_stages=s,
        num_warps=w,
    )
    for (i, j, k, s, w) in int8_mm_kernel_configs
]


@triton.jit
def matmul_kernel_with_block_pointers(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # ----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_M, BLOCK_N]` block.
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c = accumulator  # .to(tl.float16)

    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    s1_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_s1m,
    stride_s1n,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.int32,
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)  # , allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (N * idx_m)
    tmp0 = tl.load(
        s1_ptr + (tl.broadcast_to(idx_m, mask.shape)),
        mask,
        eviction_policy="evict_last",
    )
    tl.store(c_ptr + (tl.broadcast_to(xindex, mask.shape)), acc * tmp0, mask)


def int_matmul_kernel(a, b, c, config):
    M, K = a.shape
    K, N = b.shape
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    matmul_kernel_with_block_pointers[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        **config.kwargs,
    )
    return c


def int_scaled_matmul_kernel(a, b, scales1, c, config):
    M, K = a.shape
    K, N = b.shape
    # print("a.sizes(): ", a.size(), "a.strides(): ", a.stride(), "a.dtype: ", a.dtype)
    # print("b.sizes(): ", b.size(), "b.strides(): ", b.stride(), "b.dtype: ", b.dtype)
    # print("c.sizes(): ", c.size(), "c.strides(): ", c.stride(), "c.dtype: ", c.dtype)
    # print("scales1.sizes(): ", scales1.size(), "scales1.strides(): ", scales1.stride(), "scales1.dtype", scales1.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    scaled_matmul_kernel_with_block_pointers[grid](
        a,
        b,
        c,
        scales1,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),
        scales1.stride(0),
        scales1.stride(1),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        EVEN_K=(K % 2 == 0),
        **config.kwargs,
    )
    return c


lib = torch.library.Library("torchao", "FRAGMENT")
lib.define("int_matmul(Tensor a, Tensor b) -> Tensor")
lib.define("int_scaled_matmul(Tensor a, Tensor b, Tensor scales1) -> Tensor")


@torch.library.impl(lib, "int_matmul", "Meta")
def int_matmul_meta(a, b):
    M, K = a.shape
    K, N = b.shape
    return torch.empty((M, N), device=a.device, dtype=torch.int32)


@torch.library.impl(lib, "int_matmul", "CUDA")
def int_matmul_cuda(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    # 1D launch kernel where each block gets its own program.
    best_config = get_best_config_fn(
        int_matmul_kernel, [a, b, c], int8_mm_kernel_configs
    )
    if best_config is None:
        # Fall back to decomposition
        return torch.tensor([])
    return int_matmul_kernel(a, b, c, best_config)


@torch.library.impl(lib, "int_scaled_matmul", "Meta")
def int_scaled_matmul_meta(a, b, scales1):
    M, K = a.shape
    K, N = b.shape
    return torch.empty((M, N), device=a.device, dtype=scales1.dtype)


@torch.library.impl(lib, "int_scaled_matmul", "CUDA")
def int_scaled_matmul_cuda(a, b, scales1):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # Allocates output.
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=scales1.dtype)
    # 1D launch kernel where each block gets its own program.
    best_config = get_best_config_fn(
        int_scaled_matmul_kernel, [a, b, scales1, c], int8_mm_kernel_configs
    )
    return int_scaled_matmul_kernel(a, b, scales1, c, best_config)
