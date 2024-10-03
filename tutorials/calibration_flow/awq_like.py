"""
Demo for awq like flow that applies equalization scale to input activation
* insert_observers_: insert observer for activation and weight
* quantize_: convert the observed linear module to quantized linear module
   * we first quantize the weight with to_affine_quantized_intx/floatx
   * then we apply equalization scale to linear activation with to_weight_tensor_with_linear_activation_scale_metadata (input activation will be divided by equalization_scale), and then call F.linear with
     scaled input activation and quantized weight (so we can reuse the efficient quantized linear kernels used by quantized weight)
"""
import torch
import copy

import torch.nn.functional as F
from torch import Tensor
from torchao.dtypes import (
    to_affine_quantized_intx_static,
    to_affine_quantized_floatx_static,
    Float8LayoutType,
)
from torchao.quantization.utils import compute_error
from torchao.quantization import quantize_
from torchao.quantization import to_weight_tensor_with_linear_activation_scale_metadata
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.observer import (
    AffineQuantizedMinMaxObserver,
    PerTensor,
    PerAxis,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    FP8_TYPES,
)


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

    def replacement_fn(m):
        copied_act_obs = copy.deepcopy(act_obs)
        copied_weight_obs = copy.deepcopy(weight_obs)
        return ObservedLinear.from_float(m, copied_act_obs, copied_weight_obs)

    _replace_with_custom_fn_if_matches_filter(model, replacement_fn, _is_linear)

# converting observed linear module to linear module with quantzied weights (and quantized activations)
# with tensor subclasses
def apply_awq(target_dtype: torch.dtype):
    # target_dtype = torch.uint8
    def _apply_awq_to_linear(observed_linear):
        # weight quantization
        weight_scale, weight_zero_point = observed_linear.weight_obs.calculate_qparams()
        def weight_quant_func(weight):
            block_size = (1, weight.shape[1])
            if target_dtype == torch.uint8:
                return to_affine_quantized_intx_static(weight, weight_scale, weight_zero_point, block_size, target_dtype)
            elif target_dtype == torch.float8_e4m3fn:
                return to_affine_quantized_floatx_static(weight, weight_scale, block_size, target_dtype, Float8LayoutType(mm_config=None))
            else:
                raise ValueError(f"Unsupported target dtype {target_dtype}")
        linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, False, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
        linear.weight = observed_linear.weight
        linear.bias = observed_linear.bias

        # activation quantization
        # pretend this to be the equalization scale, in reality the `act_obs` should
        # be an observer that can caluclate equalization scale
        equalization_scale, _ = observed_linear.act_obs.calculate_qparams()
        equalization_scale = torch.ones_like(equalization_scale)

        linear.weight = torch.nn.Parameter(weight_quant_func(linear.weight * equalization_scale), requires_grad=False)

        linear.weight = torch.nn.Parameter(to_weight_tensor_with_linear_activation_scale_metadata(linear.weight, equalization_scale), requires_grad=False)

        return linear

    return _apply_awq_to_linear



######## Test ##########
class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, k, bias=False)
        self.linear2 = torch.nn.Linear(k, n, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def test_awq(target_dtype: torch.dtype, mapping_type: MappingType):
    print(f"Testing {target_dtype} static quantization:")
    torch.manual_seed(0)

    dtype = torch.bfloat16
    m = ToyLinearModel().eval().to(dtype).to("cuda")

    m_for_test = copy.deepcopy(m)

    m_bf16 = copy.deepcopy(m)
    example_inputs = m.example_inputs(dtype=dtype, device="cuda")
    print("example inputs shape:", example_inputs[0].shape)

    m_bf16 = torch.compile(m_bf16, mode='max-autotune')

    act_obs = AffineQuantizedMinMaxObserver(mapping_type, target_dtype, granularity_type=PerTensor(), eps=torch.finfo(torch.float32).eps)
    weight_obs = AffineQuantizedMinMaxObserver(mapping_type, target_dtype, granularity_type=PerAxis(axis=0), eps=torch.finfo(torch.float32).eps)

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
    quantize_(m, apply_awq(target_dtype), is_observed_linear)
    print("quantized model (applying tensor subclass to weight):", m)
    after_quant = m(*example_inputs)
    assert compute_error(before_quant, after_quant) > 25
    print("test passed")


if __name__ == "__main__":
    test_awq(torch.uint8, MappingType.ASYMMETRIC)
    test_awq(torch.float8_e4m3fn, MappingType.SYMMETRIC)
