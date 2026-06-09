# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional

import torch

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.deepgemm_grouped_kernels import (
    can_use_deepgemm_m_grouped,
    deepgemm_blockwise_scaled_grouped_mm,
    deepgemm_blockwise_scaled_grouped_mm_wgrad,
    prepare_deepgemm_wgrad_plan,
    triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm,
)
from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    emulated_blockwise_scaled_grouped_mm,
    triton_fp8_blockwise_weight_quant_grouped_rhs,
    triton_fp8_blockwise_weight_quant_grouped_transposed_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _is_column_major,
    _is_row_major,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)
from torchao.prototype.moe_training.utils import (
    conditional_nostrict_trace,
    pad_token_groups,
    unpad_token_groups,
)
from torchao.quantization.quantize_.common import KernelPreference


class _GroupedMMBackend(str, Enum):
    DEEPGEMM = "deepgemm"
    EMULATED = "emulated"


class _GroupedMMBackendPlan:
    kind: _GroupedMMBackend

    def quantize_forward_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def quantize_dgrad_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def grouped_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_s: torch.Tensor,
        scale_recipe_a: int,
        b_s: torch.Tensor,
        scale_recipe_b: int,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        *,
        original_group_end_offsets: Optional[torch.Tensor] = None,
        padded_group_start_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def prepare_wgrad_plan(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ):
        return None

    def wgrad(
        self,
        wgrad_plan,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class _EmulatedGroupedMMBackendPlan(_GroupedMMBackendPlan):
    kind = _GroupedMMBackend.EMULATED

    def quantize_forward_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The emulated backend consumes TorchAO's grouped RHS layout:
        # (E, K, N) data with (E, K_blocks, N_blocks) scales.
        return triton_fp8_blockwise_weight_quant_grouped_transposed_rhs(
            B_t,
            block_size=block_size,
            dtype=dtype,
        )

    def quantize_dgrad_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The emulated backend consumes TorchAO's grouped RHS layout for
        # grad_output @ weight: (E, N, K) data with
        # (E, N_blocks, K_blocks) scales.
        return triton_fp8_blockwise_weight_quant_grouped_rhs(
            B_t,
            block_size=block_size,
            dtype=dtype,
        )

    def grouped_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_s: torch.Tensor,
        scale_recipe_a: int,
        b_s: torch.Tensor,
        scale_recipe_b: int,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        *,
        original_group_end_offsets: Optional[torch.Tensor] = None,
        padded_group_start_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return emulated_blockwise_scaled_grouped_mm(
            a,
            b,
            a_s,
            scale_recipe_a,
            b_s,
            scale_recipe_b,
            offs,
            out_dtype,
            block_size,
        )


class _DeepGemmGroupedMMBackendPlan(_GroupedMMBackendPlan):
    kind = _GroupedMMBackend.DEEPGEMM

    def quantize_forward_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeepGEMM forward consumes RHS as (E, N, K), with K contiguous and
        # scales as (E, N_blocks, K_blocks). This quantizer writes that
        # layout directly, avoiding a dispatch-time transpose/copy.
        return triton_fp8_blockwise_weight_quant_grouped_transposed_rhs_deepgemm(
            B_t,
            block_size=block_size,
            dtype=dtype,
        )

    def quantize_dgrad_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeepGEMM dgrad consumes RHS as (E, K, N), with N contiguous and
        # scales as (E, K_blocks, N_blocks). This quantizer writes that
        # layout directly, avoiding a dispatch-time transpose/copy.
        return triton_fp8_blockwise_weight_quant_grouped_rhs_deepgemm(
            B_t,
            block_size=block_size,
            dtype=dtype,
        )

    def grouped_mm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_s: torch.Tensor,
        scale_recipe_a: int,
        b_s: torch.Tensor,
        scale_recipe_b: int,
        offs: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        *,
        original_group_end_offsets: Optional[torch.Tensor] = None,
        padded_group_start_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return deepgemm_blockwise_scaled_grouped_mm(
            a,
            b,
            a_s,
            scale_recipe_a,
            b_s,
            scale_recipe_b,
            offs,
            out_dtype,
            block_size,
            original_group_end_offsets=original_group_end_offsets,
            padded_group_start_offsets=padded_group_start_offsets,
        )

    def prepare_wgrad_plan(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ):
        return prepare_deepgemm_wgrad_plan(
            padded_grad_output,
            padded_a,
            group_end_offsets,
            block_size,
            dtype,
        )

    def wgrad(
        self,
        wgrad_plan,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        return deepgemm_blockwise_scaled_grouped_mm_wgrad(
            wgrad_plan.lhs,
            wgrad_plan.rhs,
            group_end_offsets,
            out_dtype,
            block_size,
        )


_EMULATED_GROUPED_MM_BACKEND_PLAN = _EmulatedGroupedMMBackendPlan()
_DEEPGEMM_GROUPED_MM_BACKEND_PLAN = _DeepGemmGroupedMMBackendPlan()


@conditional_nostrict_trace
def _to_fp8_blockwise_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    float8_dtype: torch.dtype = e4m3_dtype,
    block_size: int = 128,
    pad_token_groups_for_grouped_mm: bool = True,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
) -> torch.Tensor:
    """
    Differentiable FP8 blockwise grouped matrix multiplication.

    A has shape (M, K). B_t has shape (E, K, N), transposed and in
    per-expert column-major layout.
    """
    assert block_size == 128, "Only block_size=128 is supported"
    assert kernel_preference in (
        KernelPreference.AUTO,
        KernelPreference.EMULATED,
    ), "kernel_preference must be AUTO or EMULATED"
    return _Float8BlockwiseGroupedMM.apply(
        A,
        B_t,
        offs,
        out_dtype,
        float8_dtype,
        block_size,
        pad_token_groups_for_grouped_mm,
        kernel_preference,
    )


@conditional_nostrict_trace
def _to_fp8_blockwise_then_emulated_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    float8_dtype: torch.dtype = e4m3_dtype,
    block_size: int = 128,
    pad_token_groups_for_grouped_mm: bool = True,
) -> torch.Tensor:
    """
    Differentiable FP8 blockwise grouped matrix multiplication using an emulated GEMM.
    """
    return _to_fp8_blockwise_then_scaled_grouped_mm(
        A,
        B_t,
        offs,
        out_dtype,
        float8_dtype,
        block_size,
        pad_token_groups_for_grouped_mm,
        KernelPreference.EMULATED,
    )


def _select_fp8_blockwise_grouped_mm_backend(
    kernel_preference: KernelPreference,
    A: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
) -> _GroupedMMBackendPlan:
    if kernel_preference == KernelPreference.EMULATED:
        return _EMULATED_GROUPED_MM_BACKEND_PLAN

    assert kernel_preference == KernelPreference.AUTO, (
        "kernel_preference must be AUTO or EMULATED"
    )
    if can_use_deepgemm_m_grouped(A, out_dtype, block_size):
        return _DEEPGEMM_GROUPED_MM_BACKEND_PLAN
    return _EMULATED_GROUPED_MM_BACKEND_PLAN


def _emulated_wgrad(
    padded_grad_output: torch.Tensor,
    padded_a: torch.Tensor,
    group_end_offsets: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    grad_output_t_fp8, grad_output_t_scale = (
        triton_fp8_blockwise_act_quant_transposed_lhs(
            padded_grad_output.contiguous(),
            block_size=block_size,
            dtype=dtype,
        )
    )
    A_rhs_fp8, A_rhs_scale = triton_fp8_blockwise_act_quant_rhs(
        padded_a.contiguous(),
        block_size=block_size,
        dtype=dtype,
    )
    return emulated_blockwise_scaled_grouped_mm(
        grad_output_t_fp8,
        A_rhs_fp8,
        grad_output_t_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        A_rhs_scale,
        _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
        group_end_offsets,
        out_dtype,
        block_size,
    )


class _Float8BlockwiseGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        float8_dtype: torch.dtype = e4m3_dtype,
        block_size: int = 128,
        pad_token_groups_for_grouped_mm: bool = True,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
    ) -> torch.Tensor:
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B_t must be 3D"
        assert offs is not None and offs.dtype == torch.int32, "offs must be int32"
        assert A.dtype in (torch.bfloat16, torch.float32), "A must be bf16 or fp32"
        assert B_t.dtype in (torch.bfloat16, torch.float32), "B_t must be bf16 or fp32"
        assert A.shape[-1] == B_t.shape[-2], (
            f"shape {A.shape} and {B_t.shape} are not compatible"
        )
        assert A.shape[-1] % block_size == 0 and B_t.shape[-1] % block_size == 0, (
            f"K and N must be divisible by block_size={block_size}"
        )
        assert _is_row_major(A), "A must be row-major"
        assert _is_column_major(B_t), "B_t must be per-expert column-major"

        backend_plan = _select_fp8_blockwise_grouped_mm_backend(
            kernel_preference,
            A,
            out_dtype,
            block_size,
        )
        num_tokens = A.shape[0]
        padded_group_start_offsets = None
        padded_group_end_offsets = offs
        if pad_token_groups_for_grouped_mm:
            padded_A, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(A, offs, alignment_size=block_size)
            )
        else:
            padded_A = A

        A_fp8, A_scale = triton_fp8_blockwise_act_quant_lhs(
            padded_A.contiguous(),
            block_size=block_size,
            dtype=float8_dtype,
        )
        B_t_fp8, B_t_scale = backend_plan.quantize_forward_rhs(
            B_t,
            block_size,
            float8_dtype,
        )
        out = backend_plan.grouped_mm(
            A_fp8,
            B_t_fp8,
            A_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            B_t_scale,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            padded_group_end_offsets,
            out_dtype,
            block_size,
            original_group_end_offsets=offs
            if pad_token_groups_for_grouped_mm
            else None,
            padded_group_start_offsets=padded_group_start_offsets,
        )
        if pad_token_groups_for_grouped_mm:
            out = unpad_token_groups(
                out,
                offs,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=block_size,
            )

        ctx.save_for_backward(
            padded_A,
            B_t,
            offs,
            padded_group_start_offsets,
            padded_group_end_offsets,
        )
        ctx.out_dtype = out_dtype
        ctx.float8_dtype = float8_dtype
        ctx.block_size = block_size
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens
        ctx.backend_plan = backend_plan
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            padded_A,
            B_t,
            original_group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
        ) = ctx.saved_tensors
        out_dtype = ctx.out_dtype
        float8_dtype = ctx.float8_dtype
        block_size = ctx.block_size
        pad_token_groups_for_grouped_mm = ctx.pad_token_groups_for_grouped_mm
        num_tokens = ctx.num_tokens
        backend_plan = ctx.backend_plan

        if pad_token_groups_for_grouped_mm:
            padded_grad_output, _, _ = pad_token_groups(
                grad_output,
                original_group_end_offsets,
                alignment_size=block_size,
            )
        else:
            padded_grad_output = grad_output

        grad_output_fp8, grad_output_scale = triton_fp8_blockwise_act_quant_lhs(
            padded_grad_output.contiguous(),
            block_size=block_size,
            dtype=float8_dtype,
        )
        B_fp8, B_scale = backend_plan.quantize_dgrad_rhs(
            B_t,
            block_size,
            float8_dtype,
        )
        grad_A = backend_plan.grouped_mm(
            grad_output_fp8,
            B_fp8,
            grad_output_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            B_scale,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            padded_group_end_offsets,
            out_dtype,
            block_size,
            original_group_end_offsets=original_group_end_offsets
            if pad_token_groups_for_grouped_mm
            else None,
            padded_group_start_offsets=padded_group_start_offsets,
        )
        if pad_token_groups_for_grouped_mm:
            grad_A = unpad_token_groups(
                grad_A,
                original_group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=block_size,
            )

        wgrad_plan = backend_plan.prepare_wgrad_plan(
            padded_grad_output,
            padded_A,
            padded_group_end_offsets,
            block_size,
            float8_dtype,
        )

        if wgrad_plan is None:
            grad_B = _emulated_wgrad(
                padded_grad_output,
                padded_A,
                padded_group_end_offsets,
                out_dtype,
                block_size,
                float8_dtype,
            )
        else:
            grad_B = backend_plan.wgrad(
                wgrad_plan,
                padded_group_end_offsets,
                out_dtype,
                block_size,
            )
        return (
            grad_A,
            grad_B.transpose(-2, -1),
            None,
            None,
            None,
            None,
            None,
            None,
        )
