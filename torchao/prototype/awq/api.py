import torch
import torch.nn.functional as F
from torchao.dtypes.affine_quantized_tensor import AWQLayoutType
from torchao.prototype.awq.core import AWQObserver, ObservedLinear
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized
from torchao.dtypes.uintx.Uintx import to_uintx
from typing import Optional, Tuple



    
def insert_awq_observer(model: torch.nn.Module, quant_dtype: torch.dtype, group_size: int, input_dtype: torch.dtype, device: torch.device):
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    if quant_dtype == torch.uint4:
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.uint4
        quant_min = 0
        quant_max = 15
        eps = torch.finfo(torch.float32).eps
        preserve_zero = True
        zero_point_dtype = torch.int64
        zero_point_domain = ZeroPointDomain.INT
        print("##########################\ninsert-uint4\n##########################\n")

    elif quant_dtype == torch.int8:
        mapping_type = MappingType.SYMMETRIC
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64
        zero_point_domain = ZeroPointDomain.INT
        preserve_zero = True
        block_size = (1, -1)
        quant_min = None
        quant_max = None
        print("##########################\ninsert-int8\n##########################\n")
    else:
        raise NotImplementedError(f"{quant_dtype} not supported. Use either torch.uint4 or torch.int8")

    def replace_with_observer(layer):
        observer = AWQObserver(
            layer.weight,
            layer.bias, 
            block_size, 
            input_dtype, 
            mapping_type,
            target_dtype, 
            device,
            preserve_zero = preserve_zero,
            zero_point_domain = zero_point_domain,
            zero_point_dtype = zero_point_dtype,
            quant_min=quant_min,
            quant_max = quant_max,
            eps = eps)
        return ObservedLinear.from_float(layer, observer)
    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)

def _observed_linear_subclass_inserter(constructor):
    def insert_subclass(observed_linear):
        linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, observed_linear.bias!=None, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
        linear.weight = torch.nn.Parameter(constructor(observed_linear), requires_grad=False)
        linear.bias = observed_linear.bias
        return linear

    return insert_subclass

def awq_quant(quant_dtype = torch.uint4, group_size = 128):
    
    def weight_quant_func(observed_linear):
        # weight quantization
        equalization_scale = observed_linear.act_obs.calculate_qparams()
        if quant_dtype == torch.uint4:
            mapping_type = MappingType.ASYMMETRIC
            block_size = (1, group_size)
            target_dtype = torch.uint8
            quant_min = 0
            quant_max = 15
            eps = torch.finfo(torch.float32).eps
            preserve_zero = True
            zero_point_dtype = torch.int64
            zero_point_domain = ZeroPointDomain.INT
            layout_type = AWQLayoutType(equalization_scale, quant_dtype)

        elif quant_dtype == torch.int8:
            mapping_type = MappingType.SYMMETRIC
            target_dtype = torch.int8
            eps = torch.finfo(torch.float32).eps
            zero_point_dtype = torch.int64
            zero_point_domain = ZeroPointDomain.INT
            preserve_zero = True
            block_size = (1, -1)
            quant_min = None
            quant_max = None
            layout_type = AWQLayoutType(equalization_scale, quant_dtype)

        else:
            print(quant_dtype)
            raise("AWQ supports only uint4 and int8 quantization for now")
        
        return to_affine_quantized(
            observed_linear.weight,
            mapping_type, block_size, 
            target_dtype, quant_min, 
            quant_max, eps, 
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type)
    
    return _observed_linear_subclass_inserter(weight_quant_func)


