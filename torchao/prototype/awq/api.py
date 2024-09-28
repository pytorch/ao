import torch
import torch.nn.functional as F
import json
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
     _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes import to_affine_quantized_intx
from torchao.dtypes.uintx.uintx import _DTYPE_TO_BIT_WIDTH
from torchao.prototype.awq.core import(
    AWQObserver, 
    ObservedLinear, 
    AwqLayoutType
) 
from typing import Dict, Optional


assert len(_DTYPE_TO_BIT_WIDTH) > 0, "Error importing low bit torch.uint dtypes. Please upgrade to torch 2.3+"

def insert_awq_observer_(model: torch.nn.Module, n_validation_examples: int, validation_sequence_len: int,  quant_dtype: torch.dtype = torch.uint4,   scale_search_space_size: int = 20, group_size: int = 128):
    """
    Inserts AWQObserver into Linear layers of a given model.

    Args:
        model: The model to be modified (in place). Ensure model is on the desired device for calibration
        n_validation_examples: Number of examples used to validate scale options
        validation_sequence_len: Number of tokens in each validation example
        quant_dtype: The data type of the quantized weights. Currently only torch.uint4 is intended to be used but can be used with torch.uint1 -> torch.uint8
        scale search space size: how many different scale options to try. Original AWQ implementation uses 20. A larger size can lead to better results but takes longer to calibrate
        group_size: Quantization granularity. Use -1 for channel wise quantization
    """
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)

    # AQT config
    mapping_type = MappingType.ASYMMETRIC
    block_size = (1, group_size)
    quant_min = 0
    quant_max = 255 if quant_dtype == torch.uint8 else 2 ** _DTYPE_TO_BIT_WIDTH[quant_dtype] - 1 
    eps = torch.finfo(torch.float32).eps
    preserve_zero = True
    zero_point_dtype = torch.int64
    zero_point_domain = ZeroPointDomain.INT
    assert quant_dtype in _DTYPE_TO_BIT_WIDTH or quant_dtype == torch.uint8, "Invalid quant_dtype. Please use torch.uint1 .. torch.uint8"

    def replace_with_observer(layer):
        # creates observer and replaces linear layers with observed linear layers
        observer = AWQObserver(
            layer.weight,
            layer.bias, 
            block_size, 
            mapping_type,
            quant_dtype, 
            n_validation_examples,
            validation_sequence_len,
            scale_search_space_size,
            preserve_zero = preserve_zero,
            zero_point_domain = zero_point_domain,
            zero_point_dtype = zero_point_dtype,
            quant_min=quant_min,
            quant_max = quant_max,
            eps = eps)
        return ObservedLinear.from_float(layer, observer)
    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)

def _observed_linear_subclass_inserter(constructor):
    """
    Replaces unquantized observed linear instances with quantized linear instances.

    Args:
        constructor: the function which applies quantization to the observed linear layer
    """
    def insert_subclass(observed_linear):
        # creates the new linear layer using constructor
        linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, observed_linear.bias!=None, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
        linear.weight = torch.nn.Parameter(constructor(observed_linear), requires_grad=False)
        linear.bias = observed_linear.bias
        return linear

    return insert_subclass
    
def save_equalization_scales(model: torch.nn.Module, save_path: str) -> Dict[str, torch.Tensor]:
    result = {}

    def recurse(module: torch.nn.Module, name: str = ''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Apply the analysis function to this layer
            if isinstance(child, ObservedLinear):
                result[full_name] = child.act_obs.calculate_qparams()
            
            # Recurse into child modules
            recurse(child, full_name)

    recurse(model)
    
    torch.save(result, save_path)

def load_equalization_scales_and_quantize_(model: torch.nn.Module, equalization_scale_path: str, quant_dtype: torch.dtype = torch.uint4, group_size: int = 128, device=None) -> torch.nn.Module:
    equalization_scales = torch.load(equalization_scale_path)
    
    def recurse(module: torch.nn.Module, name: str = ''):
        if isinstance(module, ObservedLinear):
            module.equalization_scale = equalization_scales[name]
            if device is not None:
                module.to(device=device)  # move to device before quantization
            module = awq_uintx(quant_dtype, group_size)(module)
        else:
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                new_child = recurse(child, full_name)
                if new_child is not child:
                    setattr(model, full_name, new_child)
            if device is not None:
                module.to(device=device)
        return module
    
    recurse(model)
    

def awq_uintx(quant_dtype: torch.dtype = torch.uint4, group_size: int = 128):
    """
    Quantizes linear layers when passed into quantize_()

    Args:
        quant_dtype: The data type of the quantized weights. Currently only torch.uint4 is intended to be used but can be used with torch.uint1 -> torch.uint8
        group_size: Quantization granularity. Use -1 for channel wise quantization
    """
    
    assert quant_dtype in _DTYPE_TO_BIT_WIDTH or quant_dtype == torch.uint8, "Invalid quant_dtype. Please use torch.uint1 .. torch.uint8"
    def weight_quant_func(observed_linear):
        # weight quantization
        # AQT config
        if observed_linear.equalization_scale is None:
            equalization_scale = observed_linear.act_obs.calculate_qparams()
        else:
            equalization_scale = observed_linear.equalization_scale
        target_dtype = torch.uint8
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        quant_min = _DTYPE_TO_QVALUE_BOUNDS[quant_dtype][0]
        quant_max = _DTYPE_TO_QVALUE_BOUNDS[quant_dtype][1]
        eps = torch.finfo(torch.float32).eps
        preserve_zero = True
        zero_point_dtype = torch.int64
        zero_point_domain = ZeroPointDomain.INT
        layout_type = AwqLayoutType(quant_dtype, equalization_scale)
        
        return to_affine_quantized_intx(
            observed_linear.weight * equalization_scale,
            mapping_type, block_size, 
            target_dtype, quant_min, 
            quant_max, eps, 
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            layout_type=layout_type)
    
    return _observed_linear_subclass_inserter(weight_quant_func)


