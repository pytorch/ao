from dataclasses import dataclass

import torch

import torchao
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_primitives import (
    MappingType,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


@dataclass
class IntNWeightOnlyConfig(AOBaseConfig):
    """
    Configuration for applying int N-bit weight only quantization to a linear layer.
    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller size is more fine grained, choices are [512, 256, 128, 64, 32]
        `n`: number of bits to quantize to, choices are [8, 6, 5, 4, 3, 2]
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    Usage:
        from torchao.quantization import quantize_
        quantize_(model, intN_weight_only(n=your_bit_choice, group_size=group_size), optional_filter_func_for_desired_layers_to_quantize)
    """

    group_size: int = 32
    n: int = 8
    symmetric: bool = False
    set_inductor_config: bool = True


# for bc
intN_weight_only = IntNWeightOnlyConfig


@register_quantize_module_handler(IntNWeightOnlyConfig)
def _intN_weight_only_transform(
    module: torch.nn.Module,
    config: IntNWeightOnlyConfig,
) -> torch.nn.Module:
    group_size = config.group_size
    n = config.n
    symmetric = config.symmetric
    weight = module.weight
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    # for asymmetric quantization
    def apply_intN_weight_only_quant_asym(weight):
        # avoid circular dependency
        from torchao.dtypes import to_affine_quantized_intx

        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.uint8
        quant_min = 0
        quant_max = 2**n - 1
        eps = 1e-6
        zero_point_dtype = torch.int64
        return to_affine_quantized_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps,
            zero_point_dtype=zero_point_dtype,
        )  # , preserve_zero=preserve_zero,zero_point_domain=zero_point_domain)

    # for symmetric quantization
    def apply_intN_weight_only_quant_sym(weight):
        # avoid circular dependency
        from torchao.dtypes import to_affine_quantized_intx

        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        quant_min = -(2 ** (n - 1))
        quant_max = 2 ** (n - 1) - 1
        eps = 1e-6
        zero_point_dtype = torch.int64
        return to_affine_quantized_intx(
            weight,
            mapping_type,
            block_size,
            target_dtype,
            quant_min,
            quant_max,
            eps=eps,
            zero_point_dtype=zero_point_dtype,
        )

    assert n in [8, 6, 5, 4, 3, 2], "n must be one of [8, 6, 5, 4, 3, 2]"
    if n == 8:
        raise AssertionError(
            "Someone needs to refactor this code to handle int8_weight_only again"
        )
    elif n == 4:
        raise AssertionError(
            "Someone needs to refactor this code to handle int4_weight_only again"
        )
    else:
        if symmetric:
            new_weight = apply_intN_weight_only_quant_sym(weight)
        else:
            new_weight = apply_intN_weight_only_quant_asym(weight)
        module.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    return module
