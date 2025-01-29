import torch

from torchao.quantization import int4_weight_only, int8_weight_only
from torchao.quantization.quant_api import _get_linear_subclass_inserter
from torchao.quantization.quant_primitives import (
    MappingType,
)


def intN_weight_only(group_size=32, n=8, symmetric=False):
    """
    Apply int N-bit weight only quantization to a linear layer.
    Args:
        `group_size`: parameter for quantization, controls the granularity of quantization, smaller size is more fine grained, choices are [512, 256, 128, 64, 32]
        `n`: number of bits to quantize to, choices are [8, 6, 5, 4, 3, 2]
    Usage:
        from torchao.quantization import quantize_
        quantize_(model, intN_weight_only(n=your_bit_choice, group_size=group_size), optional_filter_func_for_desired_layers_to_quantize)
    """

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

    try:
        assert n in [8, 6, 5, 4, 3, 2], "n must be one of [8, 6, 5, 4, 3, 2]"
        if n == 8:
            return int8_weight_only()
        elif n == 4:
            return int4_weight_only(group_size=group_size)
        else:
            if symmetric:
                return _get_linear_subclass_inserter(apply_intN_weight_only_quant_sym)
            else:
                return _get_linear_subclass_inserter(apply_intN_weight_only_quant_asym)
    except Exception:
        raise
