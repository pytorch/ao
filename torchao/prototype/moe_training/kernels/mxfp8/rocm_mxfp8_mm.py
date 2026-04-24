# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Triton MXFP8 grouped-GEMM kernels for ROCm gfx950+.

Use ``tl.dot_scaled`` to consume per-block e8m0 scales directly as a stand-in
for ``torch._scaled_grouped_mm``'s MXFP8 path until that ships on ROCm.

Contents:
  - ``triton_mxfp8_grouped_mm``: persistent grouped GEMM for MoE fwd / dgrad.
  - ``triton_mxfp8_wgrad``: weight-gradient grouped GEMM.
"""

import torch

from torchao.prototype.mx_formats.kernels import _triton_kernels_available
from torchao.utils import is_ROCM

_rocm_mxfp8_available = is_ROCM() and _triton_kernels_available

if _rocm_mxfp8_available:
    import triton
    import triton.language as tl

    # ==================== Grouped GEMM (fwd / dgrad) ====================
    #
    # Persistent kernel: grid = num_CUs * ctas_per_cu. Each CTA walks the
    # expert list with a global tile counter and picks every num_ctas-th
    # (group, m_tile, n_tile) triple. This avoids a data-dependent grid and
    # silent row-dropping when any group exceeds an M-per-expert bound, so
    # the Python dispatcher needs no sync on offsets and stays
    # torch.compile-clean.

    @triton.jit
    def _mxfp8_grouped_mm_kernel(
        A_ptr, A_stride_m, A_stride_k,
        B_ptr, B_stride_e, B_stride_n, B_stride_k,
        A_scales_ptr, A_scales_stride_m, A_scales_stride_kb,
        B_scales_ptr, B_scales_stride_e, B_scales_stride_n, B_scales_stride_kb,
        C_ptr, C_stride_m, C_stride_n,
        group_end_offsets_ptr,
        M, N, K,
        E: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_ctas = tl.num_programs(0)

        num_n = tl.cdiv(N, BLOCK_N)
        SUB_PER_BLOCK_K: tl.constexpr = BLOCK_K // SCALE_BLOCK
        K_SCALES = K // SCALE_BLOCK

        my_next = pid  # this CTA's next global tile index
        cum = 0        # total tiles across groups [0, g)

        for g in range(E):
            group_start = tl.load(
                group_end_offsets_ptr + g - 1, mask=g > 0, other=0
            )
            group_end = tl.load(group_end_offsets_ptr + g)
            group_size = group_end - group_start
            num_m = tl.cdiv(group_size, BLOCK_M)
            tiles_in_group = num_m * num_n

            while my_next < cum + tiles_in_group:
                local = my_next - cum
                mt = local // num_n
                nt = local % num_n

                m_base = group_start + mt * BLOCK_M
                n_base = nt * BLOCK_N

                m_offs = m_base + tl.arange(0, BLOCK_M)
                n_offs = n_base + tl.arange(0, BLOCK_N)
                m_mask = (m_offs < group_end) & (m_offs < M)
                n_mask = n_offs < N

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_outer in range(0, tl.cdiv(K, BLOCK_K)):
                    k_offs = k_outer * BLOCK_K + tl.arange(0, BLOCK_K)
                    k_mask = k_offs < K
                    a = tl.load(
                        A_ptr + m_offs[:, None] * A_stride_m
                              + k_offs[None, :] * A_stride_k,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0,
                    )
                    b = tl.load(
                        B_ptr + g * B_stride_e
                              + n_offs[None, :] * B_stride_n
                              + k_offs[:, None] * B_stride_k,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0,
                    )
                    kb_offs = k_outer * SUB_PER_BLOCK_K + tl.arange(0, SUB_PER_BLOCK_K)
                    kb_mask = kb_offs < K_SCALES
                    # other=127: e8m0 bias 127 = 2^0 = 1.0 (neutral); combined
                    # with data masked to 0.0 the tail contribution is 0.
                    a_scale = tl.load(
                        A_scales_ptr + m_offs[:, None] * A_scales_stride_m
                                     + kb_offs[None, :] * A_scales_stride_kb,
                        mask=m_mask[:, None] & kb_mask[None, :], other=127,
                    )
                    b_scale = tl.load(
                        B_scales_ptr + g * B_scales_stride_e
                                     + n_offs[:, None] * B_scales_stride_n
                                     + kb_offs[None, :] * B_scales_stride_kb,
                        mask=n_mask[:, None] & kb_mask[None, :], other=127,
                    )
                    acc = tl.dot_scaled(
                        a, a_scale, "e4m3",
                        b, b_scale, "e4m3",
                        acc=acc, out_dtype=tl.float32,
                    )

                c_mask = m_mask[:, None] & n_mask[None, :]
                tl.store(
                    C_ptr + m_offs[:, None] * C_stride_m
                          + n_offs[None, :] * C_stride_n,
                    acc.to(tl.bfloat16), mask=c_mask,
                )

                my_next += num_ctas

            cum += tiles_in_group

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
        ctas_per_cu: int = 2,
    ) -> torch.Tensor:
        """MXFP8 grouped GEMM: ``output[g] = input_act[group_g] @ weight[g]^T``.

        Args:
            input_act: ``(M, K)`` fp8.
            weight: ``(E, N, K)`` fp8.
            ctas_per_cu: number of persistent CTAs per compute unit; grid is
                ``num_cus * ctas_per_cu``.
        """
        M, K = input_act.shape
        E, N, _ = weight.shape
        SCALE_BLOCK = 32

        output = torch.empty((M, N), dtype=out_dtype, device=input_act.device)
        num_cus = torch.cuda.get_device_properties(input_act.device).multi_processor_count
        grid = (num_cus * ctas_per_cu,)

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
            E=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=0, kpack=1,
        )
        return output

    # ==================== Weight-gradient grouped GEMM ====================

    @triton.jit
    def _mxfp8_wgrad_direct_kernel(
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
        M_SCALES = M // SCALE_BLOCK
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        for m_iter in range(0, tl.cdiv(M_g, BLOCK_M)):
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
            mb_mask = mb_offs < M_SCALES
            # other=127: e8m0 bias 127 = 2^0 = 1.0 (neutral).
            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None] & mb_mask[None, :], other=127,
            )
            ia_scale = tl.load(
                IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask[:, None] & mb_mask[None, :], other=127,
            )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32,
            )

        c_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            C_ptr + pid_g * C_stride_e + n_offs[:, None] * C_stride_n + k_offs[None, :] * C_stride_k,
            acc.to(tl.bfloat16), mask=c_mask,
        )

    # Partial-sum wgrad: partitions the per-group M-loop across SPLIT_M CTAs
    # and writes fp32 partials to an (E, SPLIT_M, N, K) buffer; a reduce
    # kernel sums across SPLIT_M into the (E, N, K) bf16 output. Worth it
    # when a single group leaves too few CTAs to saturate the device.

    @triton.jit
    def _mxfp8_wgrad_partial_kernel(
        GO_ptr, GO_stride_n, GO_stride_m,
        GO_scales_ptr, GO_scales_stride_n, GO_scales_stride_mb,
        IA_ptr, IA_stride_k, IA_stride_m,
        IA_scales_ptr, IA_scales_stride_k, IA_scales_stride_mb,
        P_ptr, P_stride_e, P_stride_s, P_stride_n, P_stride_k,
        group_end_offsets_ptr,
        M, N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        SPLIT_M: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_eg = tl.program_id(2)
        pid_split = pid_eg % SPLIT_M
        pid_g = pid_eg // SPLIT_M

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
        M_SCALES = M // SCALE_BLOCK
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

        num_m_iters = tl.cdiv(M_g, BLOCK_M)
        for m_iter in range(pid_split, num_m_iters, SPLIT_M):
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
            mb_mask = mb_offs < M_SCALES
            go_scale = tl.load(
                GO_scales_ptr + n_offs[:, None] * GO_scales_stride_n + mb_offs[None, :] * GO_scales_stride_mb,
                mask=n_mask[:, None] & mb_mask[None, :], other=127,
            )
            ia_scale = tl.load(
                IA_scales_ptr + k_offs[:, None] * IA_scales_stride_k + mb_offs[None, :] * IA_scales_stride_mb,
                mask=k_mask[:, None] & mb_mask[None, :], other=127,
            )

            acc = tl.dot_scaled(
                go_tile, go_scale, "e4m3",
                ia_tile, ia_scale, "e4m3",
                acc=acc, out_dtype=tl.float32,
            )

        p_mask = n_mask[:, None] & k_mask[None, :]
        tl.store(
            P_ptr + pid_g * P_stride_e + pid_split * P_stride_s
                  + n_offs[:, None] * P_stride_n + k_offs[None, :] * P_stride_k,
            acc, mask=p_mask,
        )

    @triton.jit
    def _mxfp8_wgrad_reduce_kernel(
        P_ptr, P_stride_e, P_stride_s, P_stride_n, P_stride_k,
        C_ptr, C_stride_e, C_stride_n, C_stride_k,
        N, K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_g = tl.program_id(2)

        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K
        c_mask = n_mask[:, None] & k_mask[None, :]

        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for s in range(SPLIT_M):
            acc += tl.load(
                P_ptr + pid_g * P_stride_e + s * P_stride_s
                      + n_offs[:, None] * P_stride_n + k_offs[None, :] * P_stride_k,
                mask=c_mask, other=0.0,
            )

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
        BLOCK_N: int = 256,
        BLOCK_K: int = 256,
        BLOCK_M: int = 64,
        num_warps: int = 8,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """MXFP8 weight gradient: ``grad_W[g] = grad_output[group_g]^T @ input_act[group_g]``.

        Both inputs must be dim1-quantized (scales along the M / token dim).
        For a single group (``E == 1``) the (N/BN, K/BK) grid may not saturate
        the device, so we partition the per-group M-loop across SPLIT_M CTAs
        and reduce their fp32 partials in a second pass; for E >= 2 the natural
        grid is enough and we write bf16 directly.

        Args:
            go_t: ``(N, M)`` fp8.
            ia_t: ``(K, M)`` fp8.

        Returns:
            ``(E, N, K)`` bf16.
        """
        N, M = go_t.shape
        K, _ = ia_t.shape
        E = group_end_offsets.shape[0]
        SCALE_BLOCK = 32
        SPLIT_M = 2 if E == 1 else 1

        output = torch.empty((E, N, K), dtype=out_dtype, device=go_t.device)

        if SPLIT_M == 1:
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
            _mxfp8_wgrad_direct_kernel[grid](
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
                matrix_instr_nonkdim=0, kpack=1,
            )
            return output

        partials = torch.empty(
            (E, SPLIT_M, N, K), dtype=torch.float32, device=go_t.device
        )
        partial_grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E * SPLIT_M)
        _mxfp8_wgrad_partial_kernel[partial_grid](
            go_t, go_t.stride(0), go_t.stride(1),
            go_scale.view(torch.uint8),
            go_scale.stride(0), go_scale.stride(1),
            ia_t, ia_t.stride(0), ia_t.stride(1),
            ia_scale.view(torch.uint8),
            ia_scale.stride(0), ia_scale.stride(1),
            partials,
            partials.stride(0), partials.stride(1),
            partials.stride(2), partials.stride(3),
            group_end_offsets,
            M, N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_M=BLOCK_M,
            SPLIT_M=SPLIT_M, SCALE_BLOCK=SCALE_BLOCK,
            num_warps=num_warps, num_stages=num_stages,
            matrix_instr_nonkdim=0, kpack=1,
        )

        reduce_grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), E)
        _mxfp8_wgrad_reduce_kernel[reduce_grid](
            partials,
            partials.stride(0), partials.stride(1),
            partials.stride(2), partials.stride(3),
            output,
            output.stride(0), output.stride(1), output.stride(2),
            N, K,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_M=SPLIT_M,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output

else:
    _UNAVAILABLE_MSG = "ROCm MXFP8 kernels require gfx950 or later"

    def triton_mxfp8_grouped_mm(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)

    def triton_mxfp8_wgrad(*args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_MSG)
