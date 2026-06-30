"""Grouped 2D (16x16) NVFP4 E2M1 weight quantization.

Grouped analog of ``triton_weight_quantize_2d`` (quantize_2d_triton.py). It
quantizes dense BF16 expert weights shaped ``(E, M, N)``, producing rowwise and
columnwise (W.T) FP4 codes and swizzled scale factors for every expert.
"""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

BLOCK_M = 128
BLOCK_N = 128

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _store_scales_swizzle,
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
    )
    from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
        _nvfp4_2d_quantize,
    )
    from torchao.utils import is_sm_at_least_100

    @triton.jit
    def _group_weight_quantize_2d_kernel(
        a_ptr,
        global_amax_ptr,
        qa_ptr,
        sfa_ptr,
        qa_t_ptr,
        sfa_t_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Grouped 2D (16x16) NVFP4 E2M1 weight quantization -- one tile per CTA."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2)

        # Get global amax for this expert (float32).
        global_amax = tl.load(global_amax_ptr + expert)

        # Shift base pointers for packed NVFP4 tensors to this expert.
        qa_expert_ptr = qa_ptr + expert * M * (N // 2)
        qa_t_expert_ptr = qa_t_ptr + expert * N * (M // 2)

        # Shift base pointers for FP8 scale factors to this expert.
        sfa_expert_stride = (M // 128) * (N // 64) * 32 * 16
        sfa_t_expert_stride = (N // 128) * (M // 64) * 32 * 16
        sfa_expert_ptr = sfa_ptr + expert * sfa_expert_stride
        sfa_t_expert_ptr = sfa_t_ptr + expert * sfa_t_expert_stride

        # Load a 2D (BLOCK_M, BLOCK_N) tile for this expert.
        a_expert_ptr = a_ptr + expert * M * N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(a_expert_ptr + offs_m[:, None] * N + offs_n[None, :])

        # Compute per-16x16-block scales and scaled values.
        sfa, qa = _nvfp4_2d_quantize(a, global_amax, BLOCK_M, BLOCK_N)

        # Pack FP4 values into uint8 -- non-transposed: (BLOCK_M, BLOCK_N//2, 2).
        qa_pairs = qa.reshape(BLOCK_M, BLOCK_N // 2, 2).split()
        qa_fp4x2 = convert_8xfp32_to_4xfp4_packed(qa_pairs)

        # Store packed FP4 values in this expert's rowwise output.
        outer = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        packed_offsets = outer[:, None] * (N // 2) + packed_inner[None, :]
        tl.store(qa_expert_ptr + packed_offsets, qa_fp4x2)

        # Expand and swizzle rowwise scales:
        # (BLOCK_M//16, BLOCK_N//16) -> (M//128, N//64, 32, 16).
        expand_sfa = (
            tl.expand_dims(sfa, axis=1)
            .broadcast_to([BLOCK_M // 16, 16, BLOCK_N // 16])
            .reshape(BLOCK_M, BLOCK_N // 16)
        )
        swizzle_expand_sfa = _swizzle_scales(expand_sfa, BLOCK_M, BLOCK_N)
        _store_scales_swizzle(
            swizzle_expand_sfa,
            sfa_expert_ptr,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )

        # Colwise path: quantize the transposed tile (rowwise W.T).
        a_t = tl.trans(a)  # (BLOCK_N, BLOCK_M)
        sfa_t, qa_t = _nvfp4_2d_quantize(a_t, global_amax, BLOCK_N, BLOCK_M)

        # Pack and store transposed FP4 values in this expert's W.T output.
        qa_t_pairs = qa_t.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        qa_t_fp4x2 = convert_8xfp32_to_4xfp4_packed(qa_t_pairs)
        outer_t = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_inner_t = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
        packed_offsets_t = outer_t[:, None] * (M // 2) + packed_inner_t[None, :]
        tl.store(qa_t_expert_ptr + packed_offsets_t, qa_t_fp4x2)

        # Expand and swizzle colwise scales:
        # (BLOCK_N//16, BLOCK_M//16) -> (N//128, M//64, 32, 16).
        expand_sfa_t = (
            tl.expand_dims(sfa_t, axis=1)
            .broadcast_to([BLOCK_N // 16, 16, BLOCK_M // 16])
            .reshape(BLOCK_N, BLOCK_M // 16)
        )
        swizzle_expand_sfa_t = _swizzle_scales(expand_sfa_t, BLOCK_N, BLOCK_M)
        _store_scales_swizzle(
            swizzle_expand_sfa_t,
            sfa_t_expert_ptr,
            pid_n,
            pid_m,
            N,
            M,
            BLOCK_N,
            BLOCK_M,
        )

    @torch.library.custom_op(
        "torchao::triton_group_weight_quantize_2d", mutates_args=()
    )
    def triton_group_weight_quantize_2d(
        A: torch.Tensor,
        global_amax: torch.Tensor,
        num_tensors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-expert 2D (16x16) NVFP4 E2M1 weight quantization without RHT.

        Args:
            A: Dense ``(E, M, N)`` BF16 weights. Each expert is a contiguous
                2D matrix; M and N must satisfy the swizzle and tile constraints.
            global_amax: ``(E,)`` float32 per-expert absolute maxima. The caller
                computes ``A[e].float().abs().max()`` (and optionally all-reduces
                for tensor parallelism) before passing it in.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            A 4-tuple containing:
              - ``(E, M, N//2)`` uint8 rowwise FP4 codes.
              - ``(E, M//128, N//64, 32, 16)`` FP8 rowwise swizzled scales.
              - ``(E, N, M//2)`` uint8 colwise FP4 codes (rowwise W.T).
              - ``(E, N//128, M//64, 32, 16)`` FP8 colwise swizzled scales.
        """
        if not is_sm_at_least_100():
            raise NotImplementedError("triton_group_weight_quantize_2d requires SM100+")
        if A.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16, got {A.dtype}")
        if A.ndim != 3:
            raise ValueError("Tensor A must be 3-D")
        if not A.is_contiguous():
            raise ValueError("A must be contiguous")

        E, M, N = A.shape
        if E != num_tensors:
            raise ValueError(f"Expected {num_tensors} experts, got {E}")
        if global_amax.shape != (E,):
            raise ValueError(f"global_amax must have shape ({E},)")
        if global_amax.dtype != torch.float32:
            raise ValueError(f"Expected float32 global_amax, got {global_amax.dtype}")
        if not global_amax.is_cuda or global_amax.device != A.device:
            raise ValueError("global_amax must be on the same device as A")
        if not global_amax.is_contiguous():
            raise ValueError("global_amax must be contiguous")
        if M % BLOCK_M != 0 or N % BLOCK_N != 0:
            raise ValueError(
                f"Expected M divisible by {BLOCK_M} and N divisible by {BLOCK_N}, "
                f"got M={M}, N={N}"
            )

        qa = torch.empty((E, M, N // 2), dtype=torch.uint8, device=A.device)
        sfa = torch.empty(
            (E, M // 128, N // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )
        qa_t = torch.empty((E, N, M // 2), dtype=torch.uint8, device=A.device)
        sfa_t = torch.empty(
            (E, N // 128, M // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _group_weight_quantize_2d_kernel[grid](
            A,
            global_amax,
            qa,
            sfa,
            qa_t,
            sfa_t,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=3,
        )
        return qa, sfa, qa_t, sfa_t

    @triton_group_weight_quantize_2d.register_fake
    def _(A, global_amax, num_tensors):
        E, M, N = A.shape
        qa = A.new_empty((E, M, N // 2), dtype=torch.uint8)
        sfa = A.new_empty((E, M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
        qa_t = A.new_empty((E, N, M // 2), dtype=torch.uint8)
        sfa_t = A.new_empty((E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
        return qa, sfa, qa_t, sfa_t

else:

    def triton_group_weight_quantize_2d(
        A: torch.Tensor,
        global_amax: torch.Tensor,
        num_tensors: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_weight_quantize_2d requires torch 2.10.0+ and Triton"
        )
