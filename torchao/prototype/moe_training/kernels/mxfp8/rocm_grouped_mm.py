# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton MXFP8 grouped GEMM kernels for ROCm (gfx950 / MI355X and later).

Uses tl.dot_scaled to feed per-block E8M0 scales directly into gfx950's
native v_mfma_scale_f32_*_f8f6f4 instructions, eliminating the FP32 scale
multiply round-trip per K-block (2-5x kernel speedup over manual scaling).
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_rocm_grouped_mm_available = is_ROCM() and _triton_kernels_available

if _rocm_grouped_mm_available:
    import triton
    import triton.language as tl

    # ==================== fwd / dgrad kernel ====================

    @triton.jit
    def _mxfp8_grouped_mm_kernel(
        A_ptr, A_stride_m, A_stride_k,
        B_ptr, B_stride_e, B_stride_n, B_stride_k,
        A_scales_ptr, A_scales_stride_m, A_scales_stride_kb,
        B_scales_ptr, B_scales_stride_e, B_scales_stride_n, B_scales_stride_kb,
        C_ptr, C_stride_m, C_stride_n,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_g = tl.program_id(2)

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)

        m_base = group_start + pid_m * BLOCK_M
        if m_base >= group_end:
            return
        n_base = pid_n * BLOCK_N
        if n_base >= N:
            return

        m_offs = m_base + tl.arange(0, BLOCK_M)
        n_offs = n_base + tl.arange(0, BLOCK_N)
        m_mask = m_offs < group_end
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
                B_ptr + pid_g * B_stride_e + n_offs[None, :] * B_stride_n + k_offs[:, None] * B_stride_k,
                mask=k_mask[:, None] & n_mask[None, :], other=0.0,
            )

            kb_offs = k_outer * SUB_PER_BLOCK_K + tl.arange(0, SUB_PER_BLOCK_K)
            a_scale = tl.load(
                A_scales_ptr + m_offs[:, None] * A_scales_stride_m + kb_offs[None, :] * A_scales_stride_kb,
                mask=m_mask[:, None], other=127,
            )
            b_scale = tl.load(
                B_scales_ptr + pid_g * B_scales_stride_e + n_offs[:, None] * B_scales_stride_n + kb_offs[None, :] * B_scales_stride_kb,
                mask=n_mask[:, None], other=127,
            )

            acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc=acc, out_dtype=tl.float32)

        c_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            C_ptr + m_offs[:, None] * C_stride_m + n_offs[None, :] * C_stride_n,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp8_grouped_mm(
        input_act: torch.Tensor,
        weight: torch.Tensor,
        input_act_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_M: int = 128,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        num_warps: int = 8,
        num_stages: int = 2,
        max_M_per_expert: int = 0,
    ) -> torch.Tensor:
        """
        Triton MXFP8 grouped GEMM for ROCm using hardware scaled MFMA.
        Direct kernel call (no wrap_triton) for minimal overhead in eager mode.
        """
        M, K = input_act.shape
        E, N, K2 = weight.shape
        SCALE_BLOCK = 32

        output = torch.empty((M, N), dtype=out_dtype, device=input_act.device)

        grid_m_bound = max_M_per_expert if max_M_per_expert > 0 else M
        grid = (
            triton.cdiv(grid_m_bound, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
            E,
        )

        _mxfp8_grouped_mm_kernel[grid](
            input_act, input_act.stride(0), input_act.stride(1),
            weight, weight.stride(0), weight.stride(1), weight.stride(2),
            input_act_scales.view(torch.uint8),
            input_act_scales.stride(0), input_act_scales.stride(1),
            weight_scales.view(torch.uint8),
            weight_scales.stride(0), weight_scales.stride(1), weight_scales.stride(2),
            output, output.stride(0), output.stride(1),
            group_end_offsets,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

    # ==================== wgrad kernel ====================

    @triton.jit
    def _mxfp8_wgrad_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

        group_start = tl.load(group_end_offsets_ptr + pid_g - 1, mask=pid_g > 0, other=0)
        group_end = tl.load(group_end_offsets_ptr + pid_g)
        M_g = group_end - group_start

        n_base = pid_n * BLOCK_N
        k_base = pid_k * BLOCK_K
        if n_base >= N or k_base >= K:
            return

        n_offs = n_base + tl.arange(0, BLOCK_N)
        k_offs = k_base + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K

        SUB_PER_BLOCK_M: tl.constexpr = BLOCK_M // SCALE_BLOCK
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        num_m_iters = (M_g + BLOCK_M - 1) // BLOCK_M
        for m_iter in range(0, num_m_iters):
            m_base = group_start + m_iter * BLOCK_M
            m_offs = m_base + tl.arange(0, BLOCK_M)
            m_mask = m_offs < group_end

            go_tile = tl.load(
                GO_ptr + n_offs[:, None] * GO_stride_n + m_offs[None, :] * GO_stride_m,
                mask=n_mask[:, None] & m_mask[None, :], other=0.0,
            )
            ia_tile = tl.load(
                IA_ptr + k_offs[None, :] * IA_stride_k + m_offs[:, None] * IA_stride_m,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )

            mb_base = m_base // SCALE_BLOCK
            mb_offs = mb_base + tl.arange(0, SUB_PER_BLOCK_M)

            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None], other=127,
            )
            ia_scale = tl.load(
                IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask[:, None], other=127,
            )

            acc = tl.dot_scaled(go_tile, go_scale, "e4m3", ia_tile, ia_scale, "e4m3", acc=acc, out_dtype=tl.float32)

        c_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    def triton_mxfp8_wgrad(
        go_t: torch.Tensor,
        go_scale: torch.Tensor,
        ia_t: torch.Tensor,
        ia_scale: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        BLOCK_N: int = 128,
        BLOCK_K: int = 128,
        BLOCK_M: int = 128,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """
        MXFP8 weight gradient grouped GEMM on ROCm.
        Direct kernel call (no wrap_triton) for minimal overhead.
        """
        N, M = go_t.shape
        K, M2 = ia_t.shape
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        grid = (
            triton.cdiv(N, BLOCK_N),
            triton.cdiv(K, BLOCK_K),
            E,
        )

        _mxfp8_wgrad_kernel[grid](
            go_t, go_t.stride(0), go_t.stride(1),
            go_scale.view(torch.uint8),
            go_scale.stride(0), go_scale.stride(1),
            ia_t, ia_t.stride(0), ia_t.stride(1),
            ia_scale.view(torch.uint8),
            ia_scale.stride(0), ia_scale.stride(1),
            output, output.stride(0), output.stride(1), output.stride(2),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

else:
    def triton_mxfp8_grouped_mm(
        input_act: torch.Tensor,
        weight: torch.Tensor,
        input_act_scales: torch.Tensor,
        weight_scales: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "triton_mxfp8_grouped_mm requires ROCm with gfx950 (MI355X) or later"
        )

    def triton_mxfp8_wgrad(*args, **kwargs):
        raise NotImplementedError(
            "triton_mxfp8_wgrad requires ROCm with gfx950 (MI355X) or later"
        )
