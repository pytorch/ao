from typing import Optional, Tuple

import torch

from torchao.float8.config import (
    Float8LinearConfig,
    Float8LinearRecipeName,
    ScalingGranularity,
)
from torchao.float8.float8_scaling_utils import (
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated


def _grouped_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    float8_recipe: Float8LinearRecipeName,
    offs: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    This function performs dynamic float8 quantization on the input tensors A and B using the given recipe,
    then performs a scaled grouped GEMM and returns the results.

    Args:
        A (torch.Tensor): The first input tensor, which must be a 2D tensor of shape (M * num_groups, K).
        B (torch.Tensor): The second input tensor which must be 3D, which must be shape (B, K, N).
        float8_recipe (Float8LinearRecipeName): The recipe to use for dynamic float8 quantization.
        offs (Optional[torch.Tensor]): The offsets to use to mark the starting index of each group in the input tensor of shape
        out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Currently only torch.bfloat16 is supported.
        use_fast_accum (bool): Whether to use fast accumulation or not. Default is False.
    """
    return _Float8GroupedMM.apply(
        A,
        B,
        float8_recipe,
        offs,
        out_dtype,
        use_fast_accum,
    )


class _Float8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic float8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        float8_recipe_name: Float8LinearRecipeName,
        offs: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        use_fast_accum: bool = False,
    ) -> torch.Tensor:
        # torch._scaled_grouped_mm only supports rowwise scaling currently.
        assert float8_recipe_name == Float8LinearRecipeName.ROWWISE, (
            "Only rowwise scaling is supported by torch._scaled_grouped_mm."
        )

        assert A.ndim == 2, "A must be 2D"
        assert B.ndim == 3, "B must be 3D"

        # Dim 1 of B must match the final dim of A.
        assert A.size(-1) == B.size(-2), (
            f"shape {A.shape} and {B.shape} are not compatible for _scaled_grouped_mm"
        )

        # Fetch float8 config from specified recipe name.
        float8_config = Float8LinearConfig.from_recipe_name(float8_recipe_name)

        # Store what we need for backward.
        ctx.save_for_backward(A, B)
        ctx.float8_config = float8_config
        ctx.offs = offs
        ctx.use_fast_accum = use_fast_accum
        ctx.out_dtype = out_dtype

        # Convert high precision input tensor to float8, row-major for left operand of grouped GEMM.
        # A shape: (M, K)
        # A_scale shape: (M,1)
        # torch._scaled_grouped_mm requires scales without any empty dims, so squeeze A_scale.
        # A_scale shape: (M,)
        A_fp8_row_major = hp_tensor_to_float8_dynamic(
            A,
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=float8_config.cast_config_input.scaling_granularity,
            axiswise_dim=-1,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        A_scale = A_fp8_row_major._scale.squeeze()

        # Convert B to float8, column-major for right operand of grouped GEMM.
        # B shape: (B, K, N)
        # B scales must be computed rowwise keeping the outer/final dim, so:
        # B_scale shape: (B, 1, N)
        # torch._scaled_grouped_mm requires scales without any empty dims, so squeeze B_scale.
        # B scale shape: (B, N)
        B_fp8_col_major = hp_tensor_to_float8_dynamic(
            B,
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
            axiswise_dim=-2,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        B_scale = B_fp8_col_major._scale.squeeze()

        # Perform scaled grouped GEMM and return result.
        # output shape: scaled grouped mm of (M,K) @ (B,K,N) = (M,N)
        return torch._scaled_grouped_mm(
            A_fp8_row_major._data,
            B_fp8_col_major._data,
            A_scale,
            B_scale,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, B = ctx.saved_tensors
        offs = ctx.offs
        float8_config = ctx.float8_config
        use_fast_accum = ctx.use_fast_accum
        out_dtype = ctx.out_dtype

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        #
        # grad_output shape: (M, N)
        # grad_output_scale shape: (M, 1)
        # squeeze grad_output_scale to remove empty dim, as required by torch._scaled_grouped_mm.
        # grad_output_scale shape: (M,)
        grad_output_fp8_row_major = hp_tensor_to_float8_dynamic(
            grad_output,
            float8_config.cast_config_grad_output.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.GRAD_OUTPUT,
            scaling_granularity=float8_config.cast_config_grad_output.scaling_granularity,
            axiswise_dim=-1,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        grad_output_scale = grad_output_fp8_row_major._scale.squeeze()

        # Convert B to non-transposed, float8, column-major for right operand of grouped GEMM
        # needed for grad_A: grad_output @ B.
        # Since B was transposed before entry to forward, we need to transpose it back here for this.
        B_non_transposed_col_major = B.contiguous().transpose(-2, -1)

        # - B shape: (B, K, N)
        # - B scales must be computed rowwise keeping the outer/final dim, so:
        # - B_scale shape: (B, 1, N)
        # - torch._scaled_grouped_mm requires scales without any empty dims, so squeeze A_scale.
        # - B scale shape: (B, N)
        B_non_transposed_fp8_col_major = hp_tensor_to_float8_dynamic(
            B_non_transposed_col_major,
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
            axiswise_dim=-2,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        B_scale = B_non_transposed_fp8_col_major._scale.squeeze()

        # Compute grad_A.
        #
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8_row_major._data,
            B_non_transposed_fp8_col_major._data,
            grad_output_scale,
            B_scale,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
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
                target_dtype=float8_config.cast_config_grad_output.target_dtype,
                round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
            )
        )
        A_fp8_col_major, A_scales = _to_2d_jagged_float8_tensor_colwise(
            A_col_major,
            offs,
            target_dtype=float8_config.cast_config_input.target_dtype,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )

        # Compute grad_B = grad_output_t @ A.
        # grad_B = grad_output_t @ A
        # grad_B = (N,M) @ (M,K) = (N,K)
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_fp8_row_major,
            A_fp8_col_major,
            grad_output_t_scales,
            A_scales,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
        # Since B was transposed before entry to forward, we need to transpose the grad_B to get the gradient for transposed B input.
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

    # Special case: for the 2D-2D grouped GEMM of grad_B = grad_output_t @ A, the rowwise A scales need be computed
    # for each subtensor in A separately (as defined by the offsets).
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
