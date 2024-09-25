import torch
from torchao.float8.config import (
    ScalingGranularity, 
    ScalingType, 
    CastConfig, 
    Float8LinearConfig,
    Float8GemmConfig,
)

scaling_granularities_by_gemm_lcw_recipe = [
    # @lcw's recipe
    # output = input @ weight_t
    #   input: axiswise
    #   weight_t: axiswise
    (ScalingGranularity.AXISWISE, ScalingGranularity.AXISWISE, False, False),
    # grad_input = grad_output @ weight
    #   grad_output: axiswise
    #   weight: tensorwise (but that can be computed from axiswise done in the forward)
    (ScalingGranularity.AXISWISE, ScalingGranularity.TENSORWISE, False, False),
    # grad_weight = input_t @ grad_output, in high precision (bfloat16)
    #   input_t: high precision
    #   grad_output: high precision
    (ScalingGranularity.TENSORWISE, ScalingGranularity.TENSORWISE, True, True),
]

scaling_granularities_by_gemm_all_tensorwise = [
    (ScalingGranularity.TENSORWISE, ScalingGranularity.TENSORWISE, False, False),
    (ScalingGranularity.TENSORWISE, ScalingGranularity.TENSORWISE, False, False),
    (ScalingGranularity.TENSORWISE, ScalingGranularity.TENSORWISE, False, False),
]

scaling_granularities_by_gemm_all_axiswise = [
    (ScalingGranularity.AXISWISE, ScalingGranularity.AXISWISE, False, False),
    (ScalingGranularity.AXISWISE, ScalingGranularity.AXISWISE, False, False),
    (ScalingGranularity.AXISWISE, ScalingGranularity.AXISWISE, False, False),
]

# scaling granularity and keep_in_original_precision to test by gemm arguments in this 
# order: output, grad_input, grad_weight
scaling_granularities_by_gemm = [
    # TODO(before land): move this last
    scaling_granularities_by_gemm_lcw_recipe,
    # scaling_granularities_by_gemm_all_tensorwise,
    # scaling_granularities_by_gemm_all_axiswise,
]

def get_test_float8_linear_config(
    scaling_type_input,
    scaling_type_weight,
    scaling_type_grad_output,
    scaling_granularities_by_gemm,
    emulate: bool,
):
    (
        (scaling_granularity_input, scaling_granularity_weight, original_prec_input, original_prec_weight),
        (scaling_granularity_grad_output, scaling_granularity_weight_for_grad_input, original_prec_grad_output, original_prec_weight_for_grad_input),
        (scaling_granularity_input_for_grad_weight, scaling_granularity_grad_output_for_grad_weight, original_prec_input_for_grad_weight, original_prec_grad_output_for_grad_weight),
    ) = scaling_granularities_by_gemm

    static_scale_one = torch.tensor([1.0], device="cuda")

    if scaling_type_input is ScalingType.STATIC:
        static_scale_input = static_scale_one
    else:
        static_scale_input = None
    if scaling_type_weight is ScalingType.STATIC:
        static_scale_weight = static_scale_one
    else:
        static_scale_weight = None
    if scaling_type_grad_output is ScalingType.STATIC:
        static_scale_grad_output = static_scale_one
    else:
        static_scale_grad_output = None

    cast_config_input = CastConfig(
        scaling_type=scaling_type_input,
        scaling_granularity=scaling_granularity_input,
        static_scale=static_scale_input,
        keep_in_original_precision=original_prec_input,
    )
    cast_config_input_for_grad_weight = CastConfig(
        scaling_type=scaling_type_input,
        scaling_granularity=scaling_granularity_input_for_grad_weight,
        static_scale=static_scale_input,
        keep_in_original_precision=original_prec_input_for_grad_weight,
    )

    cast_config_weight = CastConfig(
        scaling_type=scaling_type_weight,
        scaling_granularity=scaling_granularity_weight,
        static_scale=static_scale_weight,
        keep_in_original_precision=original_prec_weight,
    )
    cast_config_weight_for_grad_input = CastConfig(
        scaling_type=scaling_type_weight,
        scaling_granularity=scaling_granularity_weight_for_grad_input,
        static_scale=static_scale_weight,
        keep_in_original_precision=original_prec_weight_for_grad_input,
    )

    cast_config_grad_output = CastConfig(
        scaling_type=scaling_type_grad_output,
        scaling_granularity=scaling_granularity_grad_output,
        static_scale=static_scale_grad_output,
        keep_in_original_precision=original_prec_grad_output,
    )
    cast_config_grad_output_for_grad_weight = CastConfig(
        scaling_type=scaling_type_grad_output,
        scaling_granularity=scaling_granularity_grad_output_for_grad_weight,
        static_scale=static_scale_grad_output,
        keep_in_original_precision=original_prec_grad_output_for_grad_weight,
    )

    gemm_config_output = Float8GemmConfig(use_fast_accum=True)
    # TODO(this PR): toggle fast accum by axiswise scaling presence
    gemm_config_grad_input = Float8GemmConfig(use_fast_accum=True)
    gemm_config_grad_weight = Float8GemmConfig(use_fast_accum=True)

    config = Float8LinearConfig(
        cast_config_input=cast_config_input,
        cast_config_weight=cast_config_weight,
        cast_config_grad_output=cast_config_grad_output,
        cast_config_input_for_grad_weight=cast_config_input_for_grad_weight,
        cast_config_weight_for_grad_input=cast_config_weight_for_grad_input,
        cast_config_grad_output_for_grad_weight=cast_config_grad_output_for_grad_weight,
        gemm_config_output=gemm_config_output,
        gemm_config_grad_input=gemm_config_grad_input,
        gemm_config_grad_weight=gemm_config_grad_weight,
        emulate=emulate,
    )
    return config
