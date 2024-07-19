import torch

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)

def intN_weight_only_asym(group_size=32, n=8):
    def apply_intN_weight_only_quant_asym(weight):
        # avoid circular dep
        from torchao.dtypes import to_affine_quantized
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        quant_min = 0
        quant_max = 2**n-1
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, zero_point_dtype=zero_point_dtype, preserve_zero=preserve_zero,zero_point_domain=zero_point_domain)

    return apply_intN_weight_only_quant_asym

def intN_weight_only_sym(group_size=32, n=8):
    def apply_intN_weight_only_quant_sym(weight):
        # avoid circular dep
        from torchao.dtypes import to_affine_quantized
        mapping_type = MappingType.SYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int8
        quant_min = -2**(n-1)
        quant_max = 2**(n-1)-1
        eps = 1e-6
        preserve_zero = True
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.INT
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, quant_min, quant_max, eps, zero_point_dtype=zero_point_dtype, preserve_zero=preserve_zero,zero_point_domain=zero_point_domain)

    return apply_intN_weight_only_quant_sym
