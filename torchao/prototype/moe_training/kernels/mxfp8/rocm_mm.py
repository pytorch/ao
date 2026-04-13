# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Dense MXFP8 matmul for ROCm (gfx950 / MI355X).

A Triton kernel for the dense 2D matmul path used by MXFP8 linears (the
shared-expert w1/w2/w3 path in MoE models). Uses tl.dot_scaled to feed
per-block E8M0 scales directly into the MFMA instruction.
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_available = is_ROCM() and _triton_kernels_available

if _available:
    import triton
    import triton.language as tl

    @triton.jit
    def _mxfp8_mm_kernel(
        A_ptr, A_stride_m, A_stride_k,
        B_ptr, B_stride_k, B_stride_n,
        A_scales_ptr, A_scales_stride_m, A_scales_stride_kb,
        B_scales_ptr, B_scales_stride_n, B_scales_stride_kb,
        C_ptr, C_stride_m, C_stride_n,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        m_base = pid_m * BLOCK_M
        n_base = pid_n * BLOCK_N

        m_offs = m_base + tl.arange(0, BLOCK_M)
        n_offs = n_base + tl.arange(0, BLOCK_N)
        m_mask = m_offs < M
        n_mask = n_offs < N

        SUB_PER_BLOCK_K: tl.constexpr = BLOCK_K // SCALE_BLOCK
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_outer = K // BLOCK_K
        for k_outer in range(0, num_outer):
            k_offs = k_outer * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            a = tl.load(
                A_ptr + m_offs[:, None] * A_stride_m + k_offs[None, :] * A_stride_k,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            b = tl.load(
                B_ptr + k_offs[:, None] * B_stride_k + n_offs[None, :] * B_stride_n,
                mask=k_mask[:, None] & n_mask[None, :], other=0.0,
            )

            kb_offs = k_outer * SUB_PER_BLOCK_K + tl.arange(0, SUB_PER_BLOCK_K)
            a_scale = tl.load(
                A_scales_ptr + m_offs[:, None] * A_scales_stride_m + kb_offs[None, :] * A_scales_stride_kb,
                mask=m_mask[:, None], other=127,
            )
            b_scale = tl.load(
                B_scales_ptr + n_offs[:, None] * B_scales_stride_n + kb_offs[None, :] * B_scales_stride_kb,
                mask=n_mask[:, None], other=127,
            )

            acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc=acc, out_dtype=tl.float32)

        c_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            C_ptr + m_offs[:, None] * C_stride_m + n_offs[None, :] * C_stride_n,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp8_mm(
        a_fp8: torch.Tensor,
        b_fp8: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """
        Dense MXFP8 matmul on ROCm. Direct kernel call for minimal overhead.
        """
        M, K = a_fp8.shape
        K2, N = b_fp8.shape

        C = torch.empty((M, N), dtype=out_dtype, device=a_fp8.device)

        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

        _mxfp8_mm_kernel[grid](
            a_fp8, a_fp8.stride(0), a_fp8.stride(1),
            b_fp8, b_fp8.stride(0), b_fp8.stride(1),
            a_scale.view(torch.uint8),
            a_scale.stride(0), a_scale.stride(1),
            b_scale.view(torch.uint8),
            b_scale.stride(0), b_scale.stride(1),
            C, C.stride(0), C.stride(1),
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SCALE_BLOCK=32,
            num_warps=num_warps, num_stages=num_stages,
        )
        return C

else:
    def triton_mxfp8_mm(*args, **kwargs):
        raise NotImplementedError(
            "triton_mxfp8_mm requires ROCm with gfx950 (MI355X) or later"
        )
