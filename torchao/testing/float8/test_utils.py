import torch
from torchao.float8.config import (
    ScalingGranularity, 
    ScalingType, 
    CastConfig, 
    Float8LinearConfig,
)


def get_test_float8_linear_config(
    scaling_type_input,
    scaling_type_weight,
    scaling_type_grad_output,
    emulate: bool,
):
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
        static_scale=static_scale_input,
    )
    cast_config_weight = CastConfig(
        scaling_type=scaling_type_weight,
        static_scale=static_scale_weight,
    )
    cast_config_grad_output = CastConfig(
        scaling_type=scaling_type_grad_output,
        static_scale=static_scale_grad_output,
    )

    config = Float8LinearConfig(
        cast_config_input=cast_config_input,
        cast_config_weight=cast_config_weight,
        cast_config_grad_output=cast_config_grad_output,
        emulate=emulate,
    )
    return config
