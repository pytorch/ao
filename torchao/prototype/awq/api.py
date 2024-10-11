import torch
import torch.nn.functional as F

from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
     _DTYPE_TO_QVALUE_BOUNDS,
)
from torchao.quantization import to_weight_tensor_with_linear_activation_scale_metadata
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes.uintx import _DTYPE_TO_BIT_WIDTH, UintxLayout
from torchao.dtypes import( 
    to_affine_quantized_intx,
    TensorCoreTiledLayout,
)
from .core import(
    AWQObserver, 
    AWQObservedLinear, 
) 


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
    assert quant_dtype in _DTYPE_TO_BIT_WIDTH or quant_dtype == torch.uint8, "Invalid quant_dtype. Please use torch.uint1 .. torch.uint8"
    # AQT config
    mapping_type = MappingType.ASYMMETRIC
    quantization_granularity = PerGroup(group_size)
    quant_min = 0
    quant_max = 255 if quant_dtype == torch.uint8 else 2 ** _DTYPE_TO_BIT_WIDTH[quant_dtype] - 1 
    eps = torch.finfo(torch.float32).eps
    preserve_zero = True
    zero_point_dtype = torch.int64
    zero_point_domain = ZeroPointDomain.INT
    

    def replace_with_observer(layer):
        # creates observer and replaces linear layers with AWQObservedLinear layers
        observer = AWQObserver(
            layer.weight,
            layer.bias, 
            quantization_granularity, 
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
        return AWQObservedLinear.from_float(layer, observer)
    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)

def _observed_linear_subclass_inserter(constructor):
    """
    Replaces unquantized AWQObservedLinear instances with quantized linear instances.

    Args:
        constructor: the function which applies quantization to the AWQObservedLinear layer
    """
    def insert_subclass(observed_linear):
        # creates the new linear layer using constructor
        linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, observed_linear.bias!=None, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
        linear.weight = torch.nn.Parameter(constructor(observed_linear), requires_grad=False)
        linear.bias = observed_linear.bias
        return linear

    return insert_subclass
    

def awq_uintx(quant_dtype: torch.dtype = torch.uint4,
              group_size: int = 64, 
              use_hqq: bool = False,):
    """
    Quantizes linear layers when passed into quantize_()

    Args:
        quant_dtype: The data type of the quantized weights. Currently only torch.uint4 is intended to be used but can be used with torch.uint1 -> torch.uint8
        group_size: Quantization granularity. Use -1 for channel wise quantization
        weight_quant_fn: The quantization function to be used, which takes in the weight and returns the quantized weight. If None, then affine uint4 quantization is used
    """
    assert quant_dtype in _DTYPE_TO_BIT_WIDTH or quant_dtype == torch.uint8, "Invalid quant_dtype. Please use torch.uint1 .. torch.uint8"
    
    def weight_quant_func(observed_linear):
        equalization_scale = observed_linear.act_obs.calculate_qparams()
        # AQT config
        if quant_dtype == torch.uint4:
            target_dtype = torch.int32
            eps = 1e-6
            preserve_zero = False
            zero_point_dtype = torch.bfloat16
            zero_point_domain = ZeroPointDomain.FLOAT
            _layout = TensorCoreTiledLayout(inner_k_tiles=8)
        else:
            target_dtype = torch.uint8
            eps = torch.finfo(torch.float32).eps
            preserve_zero = True
            zero_point_dtype = torch.int64
            zero_point_domain = ZeroPointDomain.INT
            _layout = UintxLayout(quant_dtype)
            
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        quant_min = _DTYPE_TO_QVALUE_BOUNDS[quant_dtype][0]
        quant_max = _DTYPE_TO_QVALUE_BOUNDS[quant_dtype][1]
        qw = to_affine_quantized_intx(
            observed_linear.weight * equalization_scale,
            mapping_type,
            block_size, 
            target_dtype, quant_min, 
            quant_max, eps, 
            zero_point_dtype=zero_point_dtype,
            preserve_zero=preserve_zero,
            zero_point_domain=zero_point_domain,
            _layout=_layout,
            use_hqq=use_hqq
        )
        
        return to_weight_tensor_with_linear_activation_scale_metadata(qw, equalization_scale)
    
    return _observed_linear_subclass_inserter(weight_quant_func)
