# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.scaled_grouped_mm.utils import _is_column_major


def _scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """
    This function performs dynamic float8 quantization with row-wise scaling
    on the input tensors A and B, then performs a scaled grouped GEMM and returns the results.

    Args:
        A (bf16/float32 torch.Tensor): The first high-precision input tensor, which must be a 2D tensor of shape (M * num_groups, K)
            and in row-major memory layout.
        B_t (bf16/float32 torch.Tensor): The second high-precision input tensor which must be 3D, which must be shape (B, K, N)
            and in column-major memory layout.
        offs (int32 torch.Tensor): The offsets to use to mark the starting index of each group along dim0 of the A tensor.
        out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Currently only torch.bfloat16 is supported.
    """
    return _Float8GroupedMM.apply(
        A,
        B_t,
        offs,
        out_dtype,
    )


class _Float8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic float8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: torch.Tensor,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> torch.Tensor:
        # torchao _scaled_grouped_mm only supports A=2D, B=3D.
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B must be 3D"

        assert (
            A.size(-1) % 16 == 0
        ), f"A must have a last dim divisible by 16, but got shape: {A.shape}"
        assert (
            B_t.size(-2) % 16 == 0 and B_t.size(-1) % 16 == 0
        ), f"B must have last 2 dims divisible by 16, but got shape: {B_t.shape}"

        # Assert input tensors are in high-precision dtypes.
        assert (
            A.dtype == torch.float32 or A.dtype == torch.bfloat16
        ), "A must be float32 or bfloat16"
        assert (
            B_t.dtype == torch.float32 or B_t.dtype == torch.bfloat16
        ), "B must be float32 or bfloat16"
        assert offs.dtype == torch.int32, "offs must be int32"

        # Assert A and B dims are compatible for a scaled grouped GEMM.
        assert A.size(-1) == B_t.size(
            -2
        ), f"shape {A.shape} and {B_t.shape} are not compatible for _scaled_grouped_mm"

        # The left operand in the scaled grouped GEMM must be row-major due to hardware requirements.
        assert not _is_column_major(A), "A must be row-major"

        # Due to hardware requirements, the right operand in a scaled grouped GEMM must be column-major.
        assert _is_column_major(B_t), "B must be column-major"

        # Convert high precision input tensor to float8, row-major for left operand of grouped GEMM.
        # A shape: (M, K)
        # A_scales shape: (M,1)
        A_scales = tensor_to_scale(
            A,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        A_scaled = A.to(torch.float32) * A_scales
        A_fp8_row_major = to_fp8_saturated(A_scaled, torch.float8_e4m3fn)

        # Convert B to float8, column-major for right operand of grouped GEMM.
        # B shape: (B, K, N)
        # B scales must be computed rowwise keeping the outer/final dim, so:
        # B_scales shape: (B, 1, N)
        B_t_scales = tensor_to_scale(
            B_t,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=True,
        )
        B_t_scaled = B_t.to(torch.float32) * B_t_scales
        B_t_fp8_col_major = to_fp8_saturated(B_t_scaled, torch.float8_e4m3fn)

        # Precompute non-transposed B column-major for backward, to save memory by storing the
        # low precision B tensor instead of the high precision B tensor.
        # In the backward this is needed for grad_A: grad_output @ B.
        B = B_t.contiguous().transpose(-2, -1)

        # - B shape: (B, K, N)
        # - B scales must be computed rowwise keeping the outer/final dim, so:
        # - B_scale shape: (B, 1, N)
        B_scales = tensor_to_scale(
            B,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=True,
        )
        B_scaled = B.to(torch.float32) * B_scales
        B_fp8_col_major = to_fp8_saturated(B_scaled, torch.float8_e4m3fn)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_fp8_col_major, B_scales, offs)
        ctx.out_dtype = out_dtype

        # Perform scaled grouped GEMM and return result.
        # output shape: scaled grouped mm of (M,K) @ (B,K,N) = (M,N)
        return torch._scaled_grouped_mm(
            A_fp8_row_major,
            B_t_fp8_col_major,
            A_scales.squeeze().reciprocal(),
            B_t_scales.squeeze().reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, B_fp8_col_major, B_scales, offs = ctx.saved_tensors
        out_dtype = ctx.out_dtype

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        #
        # grad_output shape: (M, N)
        # grad_output_scale shape: (M, 1)
        grad_output_scales = tensor_to_scale(
            grad_output,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        grad_output_scaled = grad_output.to(torch.float32) * grad_output_scales
        grad_output_fp8_row_major = to_fp8_saturated(
            grad_output_scaled, torch.float8_e4m3fn
        )

        # Compute grad_A.
        #
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8_row_major,
            B_fp8_col_major,
            grad_output_scales.squeeze().reciprocal(),
            B_scales.squeeze().reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # Convert tranpose of grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_B: grad_output_t @ A
        grad_output_t_row_major = grad_output.transpose(-2, -1).contiguous()

        # Convert A to float8, column-major for right operand of grouped GEMM:
        # needed for grad_B: grad_output @ A
        A_col_major = A.transpose(-2, -1).contiguous().transpose(-2, -1)

        # grad_B is a special case. both operands of the grouped gemm will be 2D with offsets determing the "groups."
        # Compute scales for grad_output_t and A, which are both 2D tensors with offsets which define the "jagged" groups.
        grad_output_t_fp8_row_major, grad_output_t_scales = (
            _to_2d_jagged_float8_tensor_rowwise(
                grad_output_t_row_major,
                offs,
                target_dtype=torch.float8_e4m3fn,
                round_scales_to_power_of_2=True,
            )
        )
        A_fp8_col_major, A_scales = _to_2d_jagged_float8_tensor_colwise(
            A_col_major,
            offs,
            target_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_B = grad_output_t @ A.
        # grad_B = grad_output_t @ A
        # grad_B = (N,M) @ (M,K) = (N,K)
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_fp8_row_major,
            A_fp8_col_major,
            grad_output_t_scales.reciprocal(),
            A_scales.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return grad_A, grad_B.transpose(-2, -1), None, None, None, None


def _to_2d_jagged_float8_tensor_colwise(
    A_col_major: torch.Tensor,
    offs: torch.Tensor,
    target_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function converts the 2D input tensor A to a jagged float8 tensor,
    with scales computed along *logical columns* for each group individually,
    where groups are determined based on the offsets.

    For the right operand of a normal scaled GEMM, the rowwise scales are computed over logical columns.
    (i.e., a tensor of (K,N) will have scales of shape (1,N).

    However, for a 2D right operand of a grouped GEMM, these logical columns go through multiple distinct
    groups/subtensors, for which we want to compute scales individually. So we cannot take one set of scales
    along the logical columns and apply it to the entire tensor.

    Instead, we need to compute scales for each subtensor individually. For a tensor of shape (K,N) this results
    in scales of shape (1,N * num_groups).

    Args:
        A (torch.Tensor): The input tensor to be converted to a jagged float8 tensor.

    Returns:
        A tuple containing the jagged float8 tensor and the scales used for the conversion.
    """
    assert A_col_major.ndim == 2, "A must be 2D"

    num_groups = offs.numel()
    A_fp8_col_major = torch.empty_like(A_col_major, dtype=target_dtype)
    A_scales = torch.empty(
        A_fp8_col_major.size(1) * num_groups,
        dtype=torch.float32,
        device=A_fp8_col_major.device,
    )

    start_idx = 0
    next_scale_idx = 0
    for end_idx in offs.tolist():
        # Get the subtensor of A for this group, fetching the next group of rows, with all columns for each.
        subtensor = A_col_major[start_idx:end_idx, :]  # (local_group_size, K)

        # Compute local rowwise scales for this subtensor, which are along logical columns for the right operand.
        subtensor_scales = tensor_to_scale(
            subtensor,
            target_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=0,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )

        # Apply scales to subtensor and convert to float8.
        tensor_scaled = subtensor.to(torch.float32) * subtensor_scales
        float8_subtensor = to_fp8_saturated(tensor_scaled, target_dtype)

        # Store this portion of the resulting float8 tensor and scales.
        A_fp8_col_major[start_idx:end_idx, :] = float8_subtensor
        A_scales[next_scale_idx : next_scale_idx + subtensor_scales.numel()] = (
            subtensor_scales.squeeze()
        )

        # Update start index for next group.
        start_idx = end_idx
        next_scale_idx += subtensor_scales.numel()

    return A_fp8_col_major, A_scales


def _to_2d_jagged_float8_tensor_rowwise(
    x: torch.Tensor,
    offs: torch.Tensor,
    target_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function converts the 2D input tensor to a jagged float8 tensor,
    with scales computed along *logical rows* for each group individually,
    where groups are determined based on the offsets.

    For a 2D *left* operand of a normal scaled GEMM, the rowwise scales are computed over logical rows.
    (i.e., a tensor of (M,K) will have scales of shape (M,1).

    However, for a 2D left operand of a grouped GEMM, these logical rows go through multiple distinct
    groups/subtensors, for which we want to compute scales individually. So we cannot take one set of scales
    along the logical rows and apply it to the entire tensor.

    Instead, we need to compute scales for each subtensor individually. For a tensor of shape (M,K) this results
    in scales of shape (M * num_groups, 1).

    Args:
        A (torch.Tensor): The input tensor to be converted to a jagged float8 tensor.

    Returns:
        A tuple containing the jagged float8 tensor and the scales used for the conversion.
    """
    assert x.ndim == 2, "input tensor must be 2D"

    num_groups = offs.numel()
    x_fp8 = torch.empty_like(x, dtype=target_dtype)
    x_scales = torch.empty(
        x_fp8.size(0) * num_groups, dtype=torch.float32, device=x_fp8.device
    )

    start_idx = 0
    next_scale_idx = 0
    for end_idx in offs.tolist():
        # Get the subtensor of A for this group, fetching all rows with the next group of rows.
        subtensor = x[:, start_idx:end_idx]  # (M, local_group_size)

        # Compute local rowwise scales for this subtensor, which are along logical rows for the left operand.
        subtensor_scales = tensor_to_scale(
            subtensor,
            target_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )

        # Apply scales to subtensor and convert to float8.
        tensor_scaled = subtensor.to(torch.float32) * subtensor_scales
        float8_subtensor = to_fp8_saturated(tensor_scaled, target_dtype)

        # Store this portion of the resulting float8 tensor and scales.
        x_fp8[:, start_idx:end_idx] = float8_subtensor
        x_scales[next_scale_idx : next_scale_idx + subtensor_scales.numel()] = (
            subtensor_scales.squeeze()
        )

        # Update start index for next group.
        start_idx = end_idx
        next_scale_idx += subtensor_scales.numel()

    return x_fp8, x_scales
