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
    triton.Config(
        dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K),
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
]


@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
@triton.jit
def _scaled_int8_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    row_scale_ptr,
    col_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    COL_SCALE_SCALAR: tl.constexpr = False,
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

    row_scale = tl.load(row_scale_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    if COL_SCALE_SCALAR:
        # hack to support BitNet. col_scale is now a scalar
        col_scale = tl.load(col_scale_ptr).to(tl.float32)
    else:
        col_scale = tl.load(col_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * row_scale * col_scale

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define(
    "scaled_int8_mm(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) -> Tensor"
)


def scaled_int8_mm(
    A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor
) -> Tensor:
    """Compute `(A @ B) * row_scale * col_scale`, where `A` and `B` are INT8 to utilize
    INT8 tensor cores. `col_scale` can be a scalar.
    """
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert row_scale.dtype is col_scale.dtype
    assert A.shape[1] == B.shape[0]
    assert row_scale.squeeze().shape == (A.shape[0],)
    assert col_scale.squeeze().shape in ((B.shape[1],), ())
    assert row_scale.is_contiguous()
    assert col_scale.is_contiguous()
    return torch.ops.torchao.scaled_int8_mm(A, B, row_scale, col_scale)


@torch.library.impl(lib, "scaled_int8_mm", "Meta")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=row_scale.dtype)


@torch.library.impl(lib, "scaled_int8_mm", "CUDA")
def scaled_int8_mm_cuda(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=row_scale.dtype)
    grid = lambda meta: (
        triton.cdiv(meta["M"], meta["BLOCK_M"])
        * triton.cdiv(meta["N"], meta["BLOCK_N"]),
    )
    _scaled_int8_mm_kernel[grid](
        A,
        B,
        C,
        row_scale,
        col_scale,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        COL_SCALE_SCALAR=col_scale.numel() == 1,
    )
    return C
