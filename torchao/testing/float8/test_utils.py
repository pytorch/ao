from torchao.float8.config import (
    CastConfig,
    Float8LinearConfig,
)


def get_test_float8_linear_config(
    scaling_type_input,
    scaling_type_weight,
    scaling_type_grad_output,
    emulate: bool,
):
    cast_config_input = CastConfig(
        scaling_type=scaling_type_input,
    )
    cast_config_weight = CastConfig(
        scaling_type=scaling_type_weight,
    )
    cast_config_grad_output = CastConfig(
        scaling_type=scaling_type_grad_output,
    )

    config = Float8LinearConfig(
        cast_config_input=cast_config_input,
        cast_config_weight=cast_config_weight,
        cast_config_grad_output=cast_config_grad_output,
        emulate=emulate,
    )
    return config
