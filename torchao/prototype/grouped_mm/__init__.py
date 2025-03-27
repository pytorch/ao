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

        # perform dynamic float8 quantization using the given recipe, if specified
        assert 2 <= A.ndim <= 3, "A must be 2D or 3D"
        assert B.ndim == 3, "B must be 3D"

        # Dim 1 of B must match the final dim of A.
        assert B.size(1) == A.size(-1), "Dim 1 of B must match the final dim of A"

        # offsets are required for 2D A tensor, otherwise it should be None.
        if A.ndim == 2:
            assert offs is not None, "offs must be specified for 2D A tensor"
        else:
            assert offs is None, "offs must not be specified for 3D A tensor"

        # TODO: pad dims to be multiples of 16, as required by torch._scaled_grouped_mm.

        # Fetch float8 config from specified recipe name.
        float8_config = Float8LinearConfig.from_recipe_name(float8_recipe_name)

        # Convert high precision input tensor to float8.
        A_fp8 = hp_tensor_to_float8_dynamic(
            A,
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=float8_config.cast_config_input.scaling_granularity,
            axiswise_dim=get_maybe_axiswise_dim(
                -1, float8_config.cast_config_input.scaling_granularity
            ),
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )

        # Convert high precision weight tensor to float8.
        B_fp8 = hp_tensor_to_float8_dynamic(
            B,
            float8_config.cast_config_input.target_dtype,
            linear_mm_config=LinearMMConfig(),
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=float8_config.cast_config_weight.scaling_granularity,
            axiswise_dim=get_maybe_axiswise_dim(
                1, float8_config.cast_config_input.scaling_granularity
            ),
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )

        # Store what we need for backward.
        ctx.save_for_backward(A, B)
        ctx.float_config = float8_config
        ctx.offs = offs

        # For rowwise scaling, torch._scaled_grouped_mm requires scales without any empty dims.
        A_fp8._scale = A_fp8._scale.squeeze()
        B_fp8._scale = B_fp8._scale.squeeze()

        # Perform scaled grouped GEMM and return result.
        return torch._scaled_grouped_mm(
            A_fp8._data,
            B_fp8._data,
            A_fp8._scale,
            B_fp8._scale,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, None, None, None, None, None
