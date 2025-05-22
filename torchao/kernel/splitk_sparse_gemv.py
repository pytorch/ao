"""
This code is adapted from https://github.com/FasterDecoding/TEAL/blob/main/kernels/sparse_gemv.py

Since we already have sparse activations from ReLU, we can get rid of the thresholding step and just use the sparse tensor directly.
"""
import sys
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

if not sys.warnoptions:
    # to suppress repeated warnings when being used in a training loop.
    warnings.simplefilter("once")

configs=[
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2), 
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),

    # # Llama 3 variants can use BLOCK_N >= 1024
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
]

@triton.autotune(
    configs=configs,
    key=["CACHE_KEY_M", "CACHE_KEY_N"],
    reset_to_zero=["Y"],  # reset the content of Y to zero before computation
)
@triton.jit
def splitk_sparse_gemv_kernel(
    Y, # Pointers to matrices
    A, X,
    # Matrix dimensions
    N, M,
    CACHE_KEY_N, CACHE_KEY_M,
    # Meta-parameters
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    # now compute the block that each program will go through
    # rn (resp. rm) denotes a range of indices for rows (resp. col) of A
    
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    A_ptr = A + (rm[:, None] * N + rn[None, :])
    X_ptr = X + rm
    Y_ptr = Y + rn
    
    # eviction policy go brrr
    x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy='evict_last') # reuse x across threadblocks
    idx = (x0 != 0.0)
    # selectively load weight rows
    a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy='evict_first') # only load weights once per threadblock
    acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], axis=0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # TODO atomic add supports bfloat16 in latest triton, we should update to that
    tl.atomic_add(Y_ptr, acc0, mask=rn < N)



# NOTE: assumes that weight is column major
@triton_op("torchao::splitk_sparse_gemv", mutates_args={})
def splitk_sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y = sparse(X) @ weight.
    :param x: input tensor [1, 1, Z]
    :param weight: weight matrix [N, Z]
    :return: result tensor y
    """
    N, Z = weight.shape
    seq_len, _ = x.shape
    assert x.shape[-1] == Z
    assert x.is_contiguous()
    
    assert weight.stride(1) > 1, "weight should be column major"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
    )

    output = torch.zeros(
        seq_len,
        N,
        device=x.device,
        dtype=torch.float16,
    )


    kernel = wrap_triton(splitk_sparse_gemv_kernel)
    kernel[grid](
        output,  # data ptrs
        weight,
        x,
        N,  # shapes
        Z,
        N // 16,  # key for triton cache (limit number of compilations)
        Z // 16,
        # can't use kwargs because auto-tuner requires args
    )

    if x.dtype is not output.dtype:
        return output.to(dtype=x.dtype)

    return output
