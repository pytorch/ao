# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchao.prototype.moe_training.kernels import (
    triton_fp8_colwise_3d_scale_and_cast,
    triton_fp8_per_group_colwise_scales_dual,
    triton_fp8_rowwise_2d_scale_and_cast,
    triton_fp8_rowwise_3d_transpose_rhs,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
    pad_token_groups,
    unpad_token_groups,
)


def _to_fp8_rowwise_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
    pad_token_groups_for_grouped_mm: bool = True,
) -> torch.Tensor:
    """
    Differentiable FP8 grouped matrix multiplication with dynamic FP8 rowwise quantization.

    This function quantizes inputs A and B_t to FP8 format using rowwise scaling,
    then performs a scaled grouped matrix multiplication. It's differentiable and
    supports backpropagation through the quantization and GEMM operations.

    Args:
        A: Left operand tensor of shape (M, K). Must be row-major,
            with dtype float32 or bfloat16, and K divisible by 16.
        B_t: Right operand tensor of shape (E, K, N), transposed and in per-group column-major
            format, meaning strides of (K*N, 1, K). Must have dtype float32 or bfloat16, with K and N divisible by 16.
        offs: Offset tensor of shape (num_groups + 1,) with dtype int32, defining
            group boundaries for the grouped GEMM operation. Group sizes must be divisible by 16.
        out_dtype: Output dtype for the result. Defaults to torch.bfloat16.
        float8_dtype: Float8 dtype for quantization. Defaults to torch.float8_e4m3fn.
        pad_token_groups_for_grouped_mm: Whether to pad token groups to the next multiple of 16
            (requirement for FP8 grouped GEMM). If your tokens are already padded, set to False.

    Returns:
        torch.Tensor: Result of grouped matrix multiplication with shape (M, N).

    Note:
        - A must be row-major and B_t must be column-major due to hardware requirements
        - Both A and B_t are quantized to float8_e4m3fn with rowwise scaling
        - Scales are computed per-row and rounded to powers of 2 for efficiency
        - This function is fully differentiable via custom autograd implementation
    """
    return _Float8GroupedMM.apply(
        A, B_t, offs, out_dtype, float8_dtype, pad_token_groups_for_grouped_mm
    )


class _Float8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic float8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
        pad_token_groups_for_grouped_mm: bool = True,
    ) -> torch.Tensor:
        assert not pad_token_groups_for_grouped_mm, (
            "pad_token_groups_for_grouped_mm=True is not yet supported"
        )
        # torchao _quantize_then_scaled_grouped_mm only supports A=2D|3D and B=3D.
        assert A.ndim == 2 or A.ndim == 3, "A must be 2D or 3D"
        assert B_t.ndim == 3, "B must be 3D"

        assert A.size(-1) % 16 == 0, (
            f"A must have a last dim divisible by 16, but got shape: {A.shape}"
        )
        assert B_t.size(-2) % 16 == 0 and B_t.size(-1) % 16 == 0, (
            f"B must have last 2 dims divisible by 16, but got shape: {B_t.shape}"
        )

        # Assert input tensors are in high-precision dtypes.
        assert A.dtype == torch.float32 or A.dtype == torch.bfloat16, (
            "A must be float32 or bfloat16"
        )
        assert B_t.dtype == torch.float32 or B_t.dtype == torch.bfloat16, (
            "B must be float32 or bfloat16"
        )
        assert offs is None or offs.dtype == torch.int32, (
            "offs must be int32 tensor or None"
        )

        # Assert A and B dims are compatible for a scaled grouped GEMM.
        assert A.size(-1) == B_t.size(-2), (
            f"shape {A.shape} and {B_t.shape} are not compatible for _quantize_then_scaled_grouped_mm"
        )

        # The left operand in the scaled grouped GEMM must be row-major due to hardware requirements.
        assert not _is_column_major(A), "A must be row-major"

        # Due to hardware requirements, the right operand in a scaled grouped GEMM must be column-major.
        assert _is_column_major(B_t), "B must be column-major"

        # Save original group_end_offsets and num_tokens before padding
        num_tokens = A.shape[0]
        padded_group_start_offsets = None
        padded_group_end_offsets = None

        # Conditionally pad token groups if not aligned to 16
        if pad_token_groups_for_grouped_mm:
            padded_A, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(
                    A, offs, alignment_size=16
                )  # TODO: support emulated mode
            )
        else:
            padded_A = A
            padded_group_end_offsets = offs

        # Convert high precision input tensor to float8, row-major for left operand of grouped GEMM.
        # Uses fused Triton kernel that computes per-row absmax + scale + FP8 cast
        # in a single kernel launch (replaces 3 separate ops: tensor_to_scale,
        # multiply by scale, and to_fp8_saturated).
        # padded_A shape: (M, K) or (padded_M, K) if padding was used
        # A_scales shape: (M, 1) or (padded_M, 1) if padding was used
        A_data_row_major, A_scales = triton_fp8_rowwise_2d_scale_and_cast(
            padded_A,
            output_dtype=float8_dtype,
            round_scales_to_power_of_2=True,
        )

        # Convert B to float8, column-major for right operand of grouped GEMM.
        # Fuses the 3-op B_t chain (tensor_to_scale + B_t.to(float32) * scales +
        # to_fp8_saturated) into one Triton kernel using a two-pass approach
        # (absmax along K, then scale + cast with L2 cache reuse).
        # B_t shape: (E, K, N) column-major
        # B_t_scales shape: (E, 1, N)
        B_t_data_col_major, B_t_scales = triton_fp8_colwise_3d_scale_and_cast(
            B_t,
            output_dtype=float8_dtype,
            round_scales_to_power_of_2=True,
        )

        # Store what we need for backward.
        ctx.save_for_backward(
            padded_A, B_t, offs, padded_group_start_offsets, padded_group_end_offsets
        )
        ctx.out_dtype = out_dtype
        ctx.float8_dtype = float8_dtype
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens

        # Perform scaled grouped GEMM and return result.
        # output shape: scaled grouped mm of (M,K) @ (B,K,N) = (M,N)
        assert not _is_column_major(A_data_row_major), (
            "A must be row-major for output = A @ B"
        )
        assert _is_column_major(B_t_data_col_major), (
            "B must be column-major for output = A @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # A_scales shape: (M,1) or (padded_M, 1)
        # B_t_scales shape: (E, 1, N)
        A_scales = A_scales.squeeze(-1)
        B_t_scales = B_t_scales.squeeze(1)
        output = torch._scaled_grouped_mm(
            A_data_row_major,
            B_t_data_col_major,
            A_scales.reciprocal(),  # Reciprocals are needed for rescaling the output.
            B_t_scales.reciprocal(),
            padded_group_end_offsets,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # Unpad output if padding was used
        if pad_token_groups_for_grouped_mm:
            output = unpad_token_groups(
                output,
                offs,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=16,
            )

        assert output.shape[0] == num_tokens

        return output

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
        pad_token_groups_for_grouped_mm = ctx.pad_token_groups_for_grouped_mm
        num_tokens = ctx.num_tokens

        # Pad grad_output if padding was used in forward (needed for both dgrad and wgrad)
        if pad_token_groups_for_grouped_mm:
            padded_grad_output, _, _ = pad_token_groups(
                grad_output,
                original_group_end_offsets,
                alignment_size=16,
            )
        else:
            padded_grad_output = grad_output

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        # Uses fused Triton kernel (same as forward A quantization) to replace 3
        # separate ops (tensor_to_scale + multiply + to_fp8_saturated) in one launch.
        # padded_grad_output shape: (Mg, N) or (padded_Mg, N) if padding was used
        # grad_output_scales shape: (Mg, 1) or (padded_Mg, 1) if padding was used
        grad_output_data_row_major, grad_output_scales = (
            triton_fp8_rowwise_2d_scale_and_cast(
                padded_grad_output,
                output_dtype=float8_dtype,
                round_scales_to_power_of_2=True,
            )
        )

        # Compute B fp8 column-major for right operand of grouped GEMM:
        # grad_A = grad_output @ B.
        B_data_col_major, B_scales = triton_fp8_rowwise_3d_transpose_rhs(
            B_t,
            output_dtype=float8_dtype,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_A.
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        assert not _is_column_major(grad_output_data_row_major), (
            "grad_output must be row-major for grad_A = grad_output @ B"
        )
        assert _is_column_major(B_data_col_major), (
            "B must be column-major for grad_A = grad_output @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # grad_output_scales shape: (M,1) or (padded_M, 1)
        # B_scales shape: (E, 1, N)
        grad_output_scales = grad_output_scales.squeeze(-1)
        B_scales = B_scales.squeeze(1)
        grad_A = torch._scaled_grouped_mm(
            grad_output_data_row_major,
            B_data_col_major,
            grad_output_scales.reciprocal(),
            B_scales.reciprocal(),
            padded_group_end_offsets,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # Unpad grad_A if padding was used
        if pad_token_groups_for_grouped_mm:
            grad_A = unpad_token_groups(
                grad_A,
                original_group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=16,
            )

        # grad_B is a special case. both operands of the grouped gemm will be 2D with offsets determing the "groups."
        # Compute colwise scales for grad_output_t and A in a single fused kernel launch.
        # The dual kernel merges the row-iteration loops for both tensors, halving launches
        # and reducing per-row overhead vs two sequential triton_fp8_per_group_colwise_scales calls.
        grad_out_data_colwise, grad_out_scales, A_data_col_major, A_scales = (
            triton_fp8_per_group_colwise_scales_dual(
                padded_grad_output,
                padded_A,
                padded_group_end_offsets,
                float8_dtype,
                round_scales_to_power_of_2=True,
            )
        )
        grad_output_t_data_row_major = grad_out_data_colwise.t()
        grad_output_t_scales = grad_out_scales.t()

        # Compute grad_B = grad_output_t @ A.
        # grad_B = grad_output_t @ A
        assert not _is_column_major(grad_output_t_data_row_major), (
            "grad_output_t must be row-major for grad_B = grad_output_t @ A"
        )
        assert _is_column_major(A_data_col_major), (
            "A must be column-major for grad_B = grad_output_t @ A"
        )

        # Per-token group scales computed via triton kernels above do not have
        # the empty dim like the scales computed via tensor_to_scale, so we need
        # don't need to squeeze here.
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_data_row_major,
            A_data_col_major,
            grad_output_t_scales.reciprocal(),
            A_scales.reciprocal(),
            padded_group_end_offsets,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return grad_A, grad_B.transpose(-2, -1), None, None, None, None
