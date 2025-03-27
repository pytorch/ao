from typing import Optional

import torch

from torchao import float8
from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig


def grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    float8_recipe: Float8LinearRecipeName,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
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
        assert 2 <= B.ndim <= 3, "B must be 2D or 3D"

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
                -1, float8_config.cast_config_input.scaling_granularity
            ),
            round_scales_to_power_of_2=float8_config.round_scales_to_power_of_2,
        )
        B_fp8_t = B_fp8.transpose(-2, -1)

        # Store what we need for backward.
        ctx.save_for_backward(A, B)
        ctx.float_config = float8_config
        ctx.offs = offs

        # For rowwise scaling, torch._scaled_grouped_mm requires scales without any empty dims.
        A_fp8._scale = A_fp8._scale.squeeze()
        B_fp8_t._scale = B_fp8_t._scale.squeeze()

        # Perform scaled grouped GEMM and return result.
        return torch._scaled_grouped_mm(
            A_fp8._data,
            B_fp8_t._data,
            A_fp8._scale,
            B_fp8_t._scale,
            offs,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, None, None, None, None, None
