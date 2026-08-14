# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import math
from threading import Lock
from typing import Callable, Optional

import torch

# The blockwise fp8 GEMM is a Triton kernel. Triton is imported and the kernel is
# built lazily (and cached) on first use so this module imports fine on machines
# without Triton, and the (one-time) Triton import cost is only paid when the
# kernel is actually called.
_gemm_op: Optional[Callable] = None
_gemm_op_lock = Lock()


def _get_blockwise_fp8_gemm_op() -> Optional[Callable]:
    global _gemm_op
    op = _gemm_op
    if op is None:
        with _gemm_op_lock:
            op = _gemm_op
            if op is None:
                op = _build_blockwise_fp8_gemm_op()
                _gemm_op = op
    return op


def _build_blockwise_fp8_gemm_op() -> Optional[Callable]:
    from torch.utils._triton import has_triton

    if not has_triton():
        return None

    import triton
    import triton.language as tl
    from triton import Config

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

    @triton.autotune(
        configs=fp8_gemm_configs, key=["N", "K", "M_BUCKET", "BLOCK_SIZE_K"]
    )
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

    @torch.library.custom_op("ao::blockwise_fp8_gemm", mutates_args=())
    def _blockwise_fp8_gemm_op(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        assert a.is_contiguous()
        assert b.is_contiguous()
        assert a_s.is_contiguous()
        assert b_s.is_contiguous()
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)
        M_BUCKET = math.ceil(math.log2(M))
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        blockwise_fp8_gemm_kernel[grid](
            a, b, c, a_s, b_s, M, N, K, M_BUCKET, BLOCK_SIZE_K=block_size
        )
        return c

    @_blockwise_fp8_gemm_op.register_fake
    def _(a, a_s, b, b_s, block_size=128):
        N = b.size(0)
        c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
        return c

    return _blockwise_fp8_gemm_op


def _blockwise_fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    op = _get_blockwise_fp8_gemm_op()
    if op is None:
        raise AssertionError("unsupported without triton")
    return op(a, a_s, b, b_s, block_size)
