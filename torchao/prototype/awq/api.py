import torch
import torch.nn.functional as F
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.prototype.awq.core import AWQ_AQTLayout, AWQLayoutType, AWQObserver
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
    
def insert_awq_observer(model, input_dtype, device):
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    def replace_with_observer(layer):
        observer = AWQObserver(layer.weight, input_dtype, MappingType.ASYMMETRIC, torch.int8, device)
        return ObservedLinear.from_float(layer, observer)
    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)

# converting observed linear module to linear module with quantzied weights
# with tensor subclasses
def awq_quant(observed_linear, target_dtype=torch.int8):
    assert observed_linear.act_obs.counter > 0, "Calibrate the observer first" 
    block_size = (1, -1)
    mapping_type = MappingType.ASYMMETRIC
    # weight quantization
    equalization_scale = observed_linear.act_obs.calculate_qparams()
    layout_type = AWQLayoutType(equalization_scale)
    def weight_quant_func(weight):
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, layout_type = layout_type, zero_point_domain = ZeroPointDomain.INT)
    
    linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, False, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
    linear.weight = observed_linear.weight
    linear.bias = observed_linear.bias
    linear.weight = torch.nn.Parameter(weight_quant_func(linear.weight), requires_grad=False)
    return linear


