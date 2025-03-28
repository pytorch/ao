from typing import Optional

import torch

from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig


def _grouped_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    float8_recipe: Float8LinearRecipeName,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    This function performs dynamic float8 quantization on the input tensors A and B using the given recipe,
    then performs a scaled grouped GEMM and returns the results.

    Args:
        A (torch.Tensor): The first input tensor, which can be 2D or 3D.
        B (torch.Tensor): The second input tensor which must be 3D. Dim 1 of B must match the final dim of A.
        float8_recipe (Float8LinearRecipeName): The recipe to use for dynamic float8 quantization.
        offs (Optional[torch.Tensor]): The offsets to use to mark the starting index of each group. This
            is required when 2D A tensor is used, otherwise it should be None.
        out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Currently only torch.bfloat16 is supported.
        use_fast_accum (bool): Whether to use fast accumulation or not. Default is False.
    """
    # perform dynamic float8 quantization using the given recipe, if specified
    return _Float8GroupedMM.apply(
        A,
        B,
        float8_recipe,
        offs,
        out_dtype,
        use_fast_accum,
    )


class _Float8GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        float8_recipe_name: Float8LinearRecipeName,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        use_fast_accum: bool = False,
    ) -> torch.Tensor:
        # torch._scaled_grouped_mm only supports rowwise scaling currently.
        assert (
            float8_recipe_name == Float8LinearRecipeName.ROWWISE
        ), "Only rowwise scaling is supported by torch._scaled_grouped_mm."

        assert 2 <= A.ndim <= 3, "A must be 2D or 3D"
        assert 2 <= B.ndim <= 3, "B must be 2D or 3D"

        # Dim 1 of B must match the final dim of A.
        assert A.size(-1) == B.size(
            -2
        ), f"shape {A.shape} and {B.shape} are not compatible for _scaled_grouped_mm"

        # offsets are required for 2D A tensor, otherwise it should be None.
        if A.ndim == 2 or B.ndim == 2:
            assert offs is not None, "offs must be specified for 2D tensor"

        # TODO: pad dims to be multiples of 16, as required by torch._scaled_grouped_mm.

        # Fetch float8 config from specified recipe name.
        float8_config = Float8LinearConfig.from_recipe_name(float8_recipe_name)

        # Store what we need for backward.
        ctx.save_for_backward(A, B)
        ctx.float8_config = float8_config
        ctx.offs = offs

        # Convert high precision input tensor to float8, row-major for left operand of grouped GEMM.
        # A shape: (M, K) or (B, M, K)
        # A_scale shape: (M,1) or (B, M, 1)
        # torch._scaled_grouped_mm requires scales without any empty dims, so squeeze A_scale.
        # A_scale shape: (M,) or (B, M)
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
        # B shape: (K,N) or (B, K, N)
        # B scales must be computed rowwise keeping the outer/final dim, so:
        # B_scale shape: (1,N) or (B, 1, N)
        # torch._scaled_grouped_mm requires scales without any empty dims, so squeeze A_scale.
        # B scale shape: (N,) or (B, N)
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

        # Special case: 2D-2D grouped GEMM, the scales must be multiplied by the number of groups,
        # which is the size of the `offs` tensor.
        if A.ndim == 2 and B.ndim == 2:
            A_scale = A_scale.repeat(offs.numel())
            B_scale = B_scale.repeat(offs.numel())

        # Perform scaled grouped GEMM and return result.
        # output shape: (M, N) or (B, M, N)
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

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        #
        # grad_output shape: (M, N) or (B, M, N)
        # grad_output_scale shape: (M, 1) or (B, M, 1)
        # squeeze grad_output_scale to remove empty dim, as required by torch._scaled_grouped_mm.
        # grad_output_scale shape: (M,) or (B, M)
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
        B_non_transposed = B.transpose(-2, -1)

        # - B shape: (K,N) or (B, K, N)
        # - B scales must be computed rowwise keeping the outer/final dim, so:
        # - B_scale shape: (1,N) or (B, 1, N)
        # - torch._scaled_grouped_mm requires scales without any empty dims, so squeeze A_scale.
        # - B scale shape: (N,) or (B, N)
        B_fp8_col_major = hp_tensor_to_float8_dynamic(
            B_non_transposed,
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
            axiswise_dim=-2,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        B_scale = B_fp8_col_major._scale.squeeze()

        # Compute grad_A.
        #
        # Case 1: A=2D, B=3D with A=(M,K), B^T=(B,K,N), output=(B,M,N)
        # grad_A = grad_output @ B
        # grad_A = (B,M,N) @ (B,N,K) = (B,M,K)
        #
        # Case 2: A=3D, B=2D with A=(B,M,K), B^T=(K,N) case, output=(B,M,N)
        # grad_A = grad_output @ B
        # grad_A = (B,M,N) @ (N,K) = (B,M,K)
        #
        # Case 3: A=3D, B=3D with A=(B,M,K), B^T=(B,K,N) case, output=(B,M,N)
        # grad_A = grad_output @ B
        # grad_A = (B,M,N) @ (B,N,K) = (B,M,K)
        #
        # Case 4: A=2D, B=2D with A=(M,K), B^T=(K,N) case, output=(M,N)
        # grad_A = grad_output @ B
        # grad_A = (M,N) @ (N,K) = (M,K)
        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8_row_major._data,
            B_fp8_col_major._data,
            grad_output_scale,
            B_scale,
            offs,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # Convert tranpose of grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_B: grad_output_t @ A
        grad_output_t = grad_output.transpose(-2, -1)

        # - grad_output_t shape: (N, M) or (B, N, M)
        # - grad_output_t_scale shape: (N, 1) or (B, N, 1)
        # - squeeze grad_output_t_scale to remove empty dim, as required by torch._scaled_grouped_mm.
        # - grad_output_t_scale shape: (N,) or (B, N)
        grad_output_t_fp8_row_major = hp_tensor_to_float8_dynamic(
            grad_output_t,
            float8_config.cast_config_grad_output.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.GRAD_OUTPUT,
            scaling_granularity=float8_config.cast_config_grad_output.scaling_granularity,
            axiswise_dim=-1,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        grad_output_t_scale = grad_output_t_fp8_row_major._scale.squeeze()

        # Convert A to float8, column-major for right operand of grouped GEMM:
        # needed for grad_B: grad_output_t @ A
        #
        # - A shape: (M, K) or (B, M, K)
        # - A scales must be computed rowwise keeping the outer/final dim, for right operand in grouped GEMM, so:
        # - A_scale shape: (1,K) or (B, 1, K)
        # - torch._scaled_grouped_mm requires scales without any empty dims, so squeeze A_scale.
        # - A scale shape: (K,) or (B, K)
        A_fp8_col_major = hp_tensor_to_float8_dynamic(
            A.transpose(-2, -1)
            .contiguous()
            .tranpose(-2, -1),  # Convert to column-major
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=float8_config.cast_config_input.scaling_granularity,
            axiswise_dim=-2,
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        A_scale = A_fp8_col_major._scale.squeeze()

        # Compute grad_B = grad_output_t @ A.
        #
        # Case 1: A=2D, B=3D with A=(M,K), B^T=(B,K,N) case, output=(B,M,N)
        # grad_B = grad_output_t @ A
        # grad_B = (B,N,M) @ (B,M,K) = (B,N,K)
        #
        # Case 2: A=3D, B=2D with A=(B,M,K), B^T=(K,N) case, output=(B,M,N)
        # grad_B = grad_output_t @ A
        # grad_B = (B,N,M) @ (B,M,K) = (B,N,K)  ----> do we need to reduce along dim0 so it's (N,K)?
        #
        # Case 3: A=3D, B=3D with A=(B,M,K), B^T=(B,K,N) case, output=(B,M,N)
        # grad_B = grad_output_t @ A
        # grad_B = (B,N,M) @ (B,M,K) = (B,N,K)
        #
        # Case 4: A=2D, B=2D with A=(M,K), B^T=(K,N) case, output=(M,N)
        # grad_B = grad_output_t @ A
        # grad_B = (N,M) @ (M,K) = (N,K)
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_fp8_row_major._data,
            A_fp8_col_major._data,
            grad_output_t_scale,
            A_scale,
            offs,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_A, grad_B, None, None, None, None
