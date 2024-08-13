"""
Demo for static quantization flow
"""
import torch
import copy

# TODO: use the generalized observer for affine qunatization in the future
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
import torch.nn.functional as F
from torch import Tensor
from torchao.dtypes import to_affine_quantized_static
from torchao.quantization.utils import compute_error
from torchao.quantization import quantize_
from torchao.quantization import to_linear_activation_quantized
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter



class ObservedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, act_obs: torch.nn.Module, weight_obs: torch.nn.Module, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs
        self.weight_obs = weight_obs

    def forward(self, input: Tensor):
        observed_input = self.act_obs(input)
        observed_weight = self.weight_obs(self.weight)
        return F.linear(observed_input, observed_weight, self.bias)

    @classmethod
    def from_float(cls, float_linear, act_obs, weight_obs):
        observed_linear = cls(float_linear.in_features, float_linear.out_features, act_obs, weight_obs, False, device=float_linear.weight.device, dtype=float_linear.weight.dtype)
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear

def insert_observers_(model, act_obs, weight_obs):
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    replacement_fn = lambda m: ObservedLinear.from_float(m, act_obs, weight_obs)
    act_obs = copy.deepcopy(act_obs)
    weight_obs = copy.deepcopy(weight_obs)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn, _is_linear)

# converting observed linear module to linear module with quantzied weights (and quantized activations)
# with tensor subclasses
def apply_static_quant(observed_linear):
    target_dtype = torch.uint8

    # weight quantization
    weight_scale, weight_zero_point = observed_linear.weight_obs.calculate_qparams()
    def weight_quant_func(weight):
        block_size = (1, weight.shape[1])
        return to_affine_quantized_static(weight, weight_scale, weight_zero_point, block_size, target_dtype)
    linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, False, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
    linear.weight = observed_linear.weight
    linear.bias = observed_linear.bias

    linear.weight = torch.nn.Parameter(weight_quant_func(linear.weight), requires_grad=False)

    # activation quantization
    act_scale, act_zero_point = observed_linear.act_obs.calculate_qparams()
    input_quant_func = lambda x: to_affine_quantized_static(x, act_scale, act_zero_point, x.shape, target_dtype)
    linear.weight = torch.nn.Parameter(to_linear_activation_quantized(linear.weight, input_quant_func), requires_grad=False)

    return linear


# alternative for converting observed linear module to quantized linear module
class QuantizedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, act_obs: torch.nn.Module, weight_obs: torch.nn.Module, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.act_scale, self.act_zero_point = act_obs.calculate_qparams()
        weight_scale, weight_zero_point = weight_obs.calculate_qparams()
        assert weight.dim() == 2
        block_size = (1, weight.shape[1])
        target_dtype = torch.uint8
        self.qweight = to_affine_quantized_static(weight, weight_scale, weight_zero_point, block_size, target_dtype)
        self.bias = bias

    def forward(self, input: Tensor):
        block_size = input.shape
        target_dtype = torch.uint8
        qinput = to_affine_quantized_static(input, self.act_scale, self.act_zero_point, block_size, target_dtype)
        return F.linear(qinput, self.qweight, self.bias)

    @classmethod
    def from_observed(cls, observed_linear):
        quantized_linear = cls(observed_linear.in_features, observed_linear.out_features, observed_linear.act_obs, observed_linear.weight_obs, observed_linear.weight, observed_linear.bias)
        return quantized_linear

def apply_static_quant2(observed_linear):
    return QuantizedLinear.from_observed(observed_linear)

class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

dtype = torch.bfloat16
m = ToyLinearModel(1024, 1024, 1024).eval().to(dtype).to("cuda")
m_bf16 = copy.deepcopy(m)
example_inputs = m.example_inputs(dtype=dtype, device="cuda")

m_bf16 = torch.compile(m_bf16, mode='max-autotune')

# TODO: use the generalized observer for affine qunatization in the future
act_obs = MinMaxObserver(dtype=torch.uint8, qscheme=torch.per_tensor_affine).to("cuda")
weight_obs = PerChannelMinMaxObserver(dtype=torch.uint8, qscheme=torch.per_channel_affine).to("cuda")

before_quant = m(*example_inputs)

insert_observers_(m, act_obs, weight_obs)
# calibrating / training
for _ in range(10):
    m(*example_inputs)

after_obs = m(*example_inputs)

m2 = copy.deepcopy(m)

is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)

# quantized linear represented as an nn.Linear with modified tensor subclass weights
# for both activation and weight quantization
quantize_(m, apply_static_quant, is_observed_linear)
print("quantized model (applying tensor subclass to weight):", m)
after_quant = m(*example_inputs)
assert compute_error(before_quant, after_quant) > 30
print("test passed")

# quantized linear as a standalone module
quantize_(m2, apply_static_quant2, is_observed_linear)
print("quantized model (quantized module):", m2)
after_quant = m2(*example_inputs)
assert compute_error(before_quant, after_quant) > 30
print("test passed")
