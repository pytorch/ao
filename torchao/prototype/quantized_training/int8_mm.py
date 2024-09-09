# TODO: might merge this with torchao/kernel/intmm_triton.py

import torch
import triton
import triton.language as tl
from torch import Tensor

lib = torch.library.Library("torchao", "FRAGMENT")


# TODO: prune configs to speedup triton autotune
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
configs = [
    (128, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (32, 64, 32, 5, 2),
    # Good config for fp8 inputs
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
    (256, 64, 128, 4, 4),
    (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 32, 64, 4, 4),
    # https://github.com/pytorch/pytorch/blob/7868b65c4d4f34133607b0166f08e9fbf3b257c4/torch/_inductor/kernel/mm_common.py#L172
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
]

configs = [
    triton.Config(dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K), num_stages=num_stages, num_warps=num_warps)
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
]


@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _int8_mm_dequant_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scale_rowwise_ptr,
    B_scale_colwise_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
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
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    a_scale = tl.load(A_scale_rowwise_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    b_scale = tl.load(B_scale_colwise_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * a_scale * b_scale

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("int8_mm_dequant(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) -> Tensor")


def int8_mm_dequant(A: Tensor, B: Tensor, A_scale_rowwise: Tensor, B_scale_colwise: Tensor) -> Tensor:
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A_scale_rowwise.dtype is B_scale_colwise.dtype
    assert A.shape[1] == B.shape[0]
    assert A_scale_rowwise.squeeze().shape == (A.shape[0],)
    assert B_scale_colwise.squeeze().shape == (B.shape[1],)
    assert A_scale_rowwise.is_contiguous()
    assert B_scale_colwise.is_contiguous()
    return torch.ops.torchao.int8_mm_dequant(A, B, A_scale_rowwise, B_scale_colwise)


@torch.library.impl(lib, "int8_mm_dequant", "Meta")
def _(A: Tensor, B: Tensor, A_scale_rowwise: Tensor, B_scale_colwise: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A_scale_rowwise.dtype)


@torch.library.impl(lib, "int8_mm_dequant", "CUDA")
def int8_mm_dequant_cuda(A: Tensor, B: Tensor, A_scale_rowwise: Tensor, B_scale_colwise: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=A_scale_rowwise.dtype)
    grid = lambda meta: (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)
    _int8_mm_dequant_kernel[grid](
        A, B, C, A_scale_rowwise, B_scale_colwise, M, N, K, *A.stride(), *B.stride(), *C.stride(), EVEN_K=K % 2 == 0
    )
    return C
