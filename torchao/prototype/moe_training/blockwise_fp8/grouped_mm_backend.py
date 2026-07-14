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
from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    emulated_blockwise_scaled_grouped_mm,
)
from torchao.prototype.blockwise_fp8_training.grouped_weight_quant import (
    triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs,
    triton_fp8_blockwise_weight_quant_grouped_forward_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
)
from torchao.quantization.quantize_.common import KernelPreference


class _GroupedMMBackendKind(str, Enum):
    """Grouped GEMM backend selected for the FP8 MoE training op."""

    DEEPGEMM = "deepgemm"
    EMULATED = "emulated"


class _GroupedMMBackend:
    """Backend-specific quantization and grouped GEMM implementation."""

    kind: _GroupedMMBackendKind

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
    ) -> torch.Tensor:
        raise NotImplementedError


class _EmulatedGroupedMMBackend(_GroupedMMBackend):
    """TorchAO emulated backend using PyTorch grouped-mm operand layouts."""

    kind = _GroupedMMBackendKind.EMULATED

    def quantize_forward_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The shared direct quantizer writes forward RHS as row-major
        # (E, N, K) data with (E, N_blocks, K_blocks) scales. Transpose views
        # convert that to the RHS contract required by torch._grouped_mm and
        # torch._scaled_grouped_mm for output = A @ B_t: (E, K, N) data with
        # (E, K_blocks, N_blocks) scales.
        q, scale = triton_fp8_blockwise_weight_quant_grouped_forward_rhs(
            B_t,
            block_size=block_size,
            dtype=dtype,
        )
        return q.transpose(-2, -1), scale.transpose(-2, -1)

    def quantize_dgrad_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The shared direct quantizer writes dgrad RHS as row-major
        # (E, K, N) data with (E, K_blocks, N_blocks) scales. Transpose views
        # convert that to the RHS contract required by torch._grouped_mm and
        # torch._scaled_grouped_mm for grad_output @ weight: (E, N, K) data with
        # (E, N_blocks, K_blocks) scales.
        q, scale = triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs(
            B_t,
            block_size=block_size,
            dtype=dtype,
        )
        return q.transpose(-2, -1), scale.transpose(-2, -1)

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
    ) -> torch.Tensor:
        # For expert e, wgrad is [N, M_e] @ [M_e, K] -> [N, K]. PyTorch's
        # 2D x 2D grouped_mm contract represents all experts as two 2D tensors:
        #   grad_output_t: [N, M], row-major
        #   A:             [M, K], column-major
        # `group_end_offsets` partitions their shared M dimension into M_e.
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


@dataclass(frozen=True)
class _DeepGemmGroupedMMBackend(_GroupedMMBackend):
    """DeepGEMM backend plus the offset metadata shared by its kernels."""

    kind = _GroupedMMBackendKind.DEEPGEMM
    offset_plan: DeepGemmGroupedOffsetPlan

    def quantize_forward_rhs(
        self,
        B_t: torch.Tensor,
        block_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # DeepGEMM forward consumes RHS as (E, N, K), with K contiguous and
        # scales as (E, N_blocks, K_blocks). This quantizer writes that
        # layout directly, avoiding a dispatch-time transpose/copy.
        return triton_fp8_blockwise_weight_quant_grouped_forward_rhs(
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
        return triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs(
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
            offset_plan=self.offset_plan,
        )

    def wgrad(
        self,
        padded_grad_output: torch.Tensor,
        padded_a: torch.Tensor,
        group_end_offsets: torch.Tensor,
        out_dtype: torch.dtype,
        block_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # DeepGEMM computes the same [N, M_e] @ [M_e, K] wgrad. Its API calls
        # this K-grouped because the expert-dependent M_e is the GEMM
        # reduction/K extent. `k_grouped_fp8_gemm_nt_contiguous` takes two flat
        # expert-major buffers. The quantizer
        # concatenates row-major [N, M_e] blocks in expert order for the LHS
        # and row-major [K, M_e] blocks in expert order for the RHS. The flat
        # segment lengths are therefore [N * M_0, ..., N * M_{E-1}] and
        # [K * M_0, ..., K * M_{E-1}], respectively.
        wgrad_plan = prepare_deepgemm_wgrad_plan(
            padded_grad_output,
            padded_a,
            self.offset_plan,
            block_size,
            dtype,
        )
        assert wgrad_plan is not None, (
            "DeepGEMM backend requires block-aligned group sizes for wgrad"
        )
        return deepgemm_blockwise_scaled_grouped_mm_wgrad(
            wgrad_plan.lhs,
            wgrad_plan.rhs,
            self.offset_plan,
            out_dtype,
            block_size,
        )


_EMULATED_GROUPED_MM_BACKEND = _EmulatedGroupedMMBackend()


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
) -> _GroupedMMBackend:
    """Select the grouped GEMM backend for one forward/backward pass.

    ``KernelPreference.EMULATED`` always selects the TorchAO emulated backend.
    ``KernelPreference.AUTO`` selects DeepGEMM only when the optional dependency
    exposes both M-grouped and K-grouped training kernels, the input is on
    CUDA SM90+, ``out_dtype`` is bf16, ``block_size`` is 128, and every expert
    group is block-aligned. Any unsupported AUTO case falls back to emulated.
    When DeepGEMM is selected, the returned backend owns the offset/layout plan
    reused by forward, dgrad, and wgrad. The autograd function saves this
    backend, so wgrad cannot independently choose a different layout or kernel.
    """

    if kernel_preference == KernelPreference.EMULATED:
        return _EMULATED_GROUPED_MM_BACKEND

    assert kernel_preference == KernelPreference.AUTO, (
        "kernel_preference must be AUTO or EMULATED"
    )
    if not can_use_deepgemm_grouped_training(A, out_dtype, block_size):
        return _EMULATED_GROUPED_MM_BACKEND

    groups_block_aligned_by_construction = original_group_end_offsets is not None
    offset_plan = build_deepgemm_grouped_offset_plan(
        group_end_offsets,
        original_group_end_offsets=original_group_end_offsets,
        padded_group_start_offsets=padded_group_start_offsets,
        num_rows=num_rows,
        groups_block_aligned_by_construction=groups_block_aligned_by_construction,
    )
    if not offset_plan.groups_are_block_aligned(block_size):
        return _EMULATED_GROUPED_MM_BACKEND

    return _DeepGemmGroupedMMBackend(offset_plan=offset_plan)
