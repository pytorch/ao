# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from torchao.prototype.blockwise_fp8_training.deepgemm_grouped_kernels import (
    can_use_deepgemm_grouped_training,
    deepgemm_blockwise_scaled_grouped_mm,
    deepgemm_blockwise_scaled_grouped_mm_wgrad,
    prepare_deepgemm_wgrad_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    DeepGemmGroupedOffsetPlan,
    build_deepgemm_grouped_offset_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_quant import (
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
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)
from torchao.quantization.quantize_.common import KernelPreference


class _GroupedMMBackend(str, Enum):
    DEEPGEMM = "deepgemm"
    EMULATED = "emulated"


@dataclass(frozen=True)
class _GroupedMMBackendSelection:
    plan: "_GroupedMMBackendPlan"
    deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None


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
        deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
        *,
        deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None,
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
        deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None,
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

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
        *,
        deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None,
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
        deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None,
    ) -> torch.Tensor:
        assert deepgemm_offset_plan is not None, (
            "DeepGEMM backend requires grouped offset metadata"
        )
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
            offset_plan=deepgemm_offset_plan,
        )

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
        *,
        deepgemm_offset_plan: DeepGemmGroupedOffsetPlan | None,
    ) -> torch.Tensor:
        assert deepgemm_offset_plan is not None, (
            "DeepGEMM backend requires grouped offset metadata"
        )
        wgrad_plan = prepare_deepgemm_wgrad_plan(
            padded_grad_output,
            padded_a,
            deepgemm_offset_plan,
            block_size,
            dtype,
        )
        assert wgrad_plan is not None, (
            "DeepGEMM backend requires block-aligned group sizes for wgrad"
        )
        return deepgemm_blockwise_scaled_grouped_mm_wgrad(
            wgrad_plan.lhs,
            wgrad_plan.rhs,
            deepgemm_offset_plan,
            out_dtype,
            block_size,
        )


_EMULATED_GROUPED_MM_BACKEND_PLAN = _EmulatedGroupedMMBackendPlan()
_DEEPGEMM_GROUPED_MM_BACKEND_PLAN = _DeepGemmGroupedMMBackendPlan()


def _select_fp8_blockwise_grouped_mm_backend(
    kernel_preference: KernelPreference,
    A: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    group_end_offsets: torch.Tensor,
    *,
    original_group_end_offsets: Optional[torch.Tensor] = None,
    padded_group_start_offsets: Optional[torch.Tensor] = None,
    num_rows: Optional[int] = None,
) -> _GroupedMMBackendSelection:
    if kernel_preference == KernelPreference.EMULATED:
        return _GroupedMMBackendSelection(
            plan=_EMULATED_GROUPED_MM_BACKEND_PLAN,
            deepgemm_offset_plan=None,
        )

    assert kernel_preference == KernelPreference.AUTO, (
        "kernel_preference must be AUTO or EMULATED"
    )
    if not can_use_deepgemm_grouped_training(A, out_dtype, block_size):
        return _GroupedMMBackendSelection(
            plan=_EMULATED_GROUPED_MM_BACKEND_PLAN,
            deepgemm_offset_plan=None,
        )

    deepgemm_offset_plan = build_deepgemm_grouped_offset_plan(
        group_end_offsets,
        original_group_end_offsets=original_group_end_offsets,
        padded_group_start_offsets=padded_group_start_offsets,
        num_rows=num_rows,
    )
    if not deepgemm_offset_plan.groups_are_block_aligned(block_size):
        return _GroupedMMBackendSelection(
            plan=_EMULATED_GROUPED_MM_BACKEND_PLAN,
            deepgemm_offset_plan=None,
        )

    return _GroupedMMBackendSelection(
        plan=_DEEPGEMM_GROUPED_MM_BACKEND_PLAN,
        deepgemm_offset_plan=deepgemm_offset_plan,
    )
