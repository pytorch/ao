import torch
import torch.nn.functional as F
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.dtypes.affine_quantized_tensor import AWQ_INT4_LayoutType, AWQLayoutType
from torchao.prototype.awq.core import AWQObserver
from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
)
from torchao.dtypes import to_affine_quantized

class ObservedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, act_obs: torch.nn.Module, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        self.act_obs(input)
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_float(cls, float_linear, act_obs):
        observed_linear = cls(float_linear.in_features, float_linear.out_features, act_obs, False, device=float_linear.weight.device, dtype=float_linear.weight.dtype)
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear


def insert_awq_observer(model, quant_dtype, group_size, input_dtype, device):
    assert quant_dtype in ["int4", "int8"]
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    if quant_dtype == "int4":
        mapping_type = MappingType.ASYMMETRIC
        block_size = (1, group_size)
        target_dtype = torch.int32
        quant_min = 0
        quant_max = 15
        eps = 1e-6
        preserve_zero = False
        zero_point_dtype = torch.bfloat16
        zero_point_domain = ZeroPointDomain.FLOAT

    elif quant_dtype == "int8":
        mapping_type = MappingType.SYMMETRIC
        target_dtype = torch.int8
        eps = torch.finfo(torch.float32).eps
        zero_point_dtype = torch.int64
        zero_point_domain = ZeroPointDomain.INT
        preserve_zero = True
        block_size = (1, -1)
        quant_min = None
        quant_max = None

    def replace_with_observer(layer):
        observer = AWQObserver(
            layer.weight, 
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

# variant of _get_linear_subclass_inserter that works with observed linear class
def _observed_linear_subclass_inserter(constructor):
    def insert_subclass(observed_linear):
        linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, False, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
        linear.weight = torch.nn.Parameter(constructor(observed_linear), requires_grad=False)
        return linear

    return insert_subclass

def awq_quant(quant_dtype = "int4", group_size = 128, scale_list =[]):
    
    def weight_quant_func(observed_linear):
        assert observed_linear.act_obs.counter > 0, "Calibrate the observer first" 
        assert quant_dtype in ["int4", "int8"]
        mapping_type = MappingType.ASYMMETRIC
        # weight quantization
        equalization_scale = observed_linear.act_obs.calculate_qparams()
        if quant_dtype == "int4":
            mapping_type = MappingType.ASYMMETRIC
            block_size = (1, group_size)
            target_dtype = torch.int32
            quant_min = 0
            quant_max = 15
            eps = 1e-6
            preserve_zero = False
            zero_point_dtype = torch.bfloat16
            zero_point_domain = ZeroPointDomain.FLOAT
            layout_type = AWQ_INT4_LayoutType(equalization_scale)

        elif quant_dtype == "int8":
            mapping_type = MappingType.SYMMETRIC
            target_dtype = torch.int8
            eps = torch.finfo(torch.float32).eps
            zero_point_dtype = torch.int64
            zero_point_domain = ZeroPointDomain.INT
            preserve_zero = True
            block_size = (1, observed_linear.weight.shape[1])
            layout_type = AWQLayoutType(equalization_scale)
            quant_min = None
            quant_max = None

        scale_list.append(equalization_scale)
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


