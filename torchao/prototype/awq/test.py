from copy import deepcopy
import torch
import torch.nn.functional as F
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.prototype.awq.core import AWQ_AQTLayout, AWQLayoutType, AWQObserver
from torchao.quantization import quantize_, int8_weight_only
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
    
def insert_awq_observer(model, device):
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    def replace_with_observer(layer):
        observer = AWQObserver((layer.weight), MappingType.ASYMMETRIC, torch.int8, device)
        return ObservedLinear.from_float(layer, observer)
    _replace_with_custom_fn_if_matches_filter(model, replace_with_observer, _is_linear)

# converting observed linear module to linear module with quantzied weights
# with tensor subclasses
def awq_quant(observed_linear):
    assert observed_linear.act_obs.counter > 0, "Calibrate the observer first" 
    target_dtype = torch.int8
    block_size = (1, -1)
    mapping_type = MappingType.ASYMMETRIC
    # weight quantization
    equalization_scale = observed_linear.act_obs.calculate_qparams()
    layout_type = AWQLayoutType(equalization_scale)
    def weight_quant_func(weight):
        return to_affine_quantized(weight, mapping_type, block_size, target_dtype, layout_type = layout_type)
    
    linear = torch.nn.Linear(observed_linear.in_features, observed_linear.out_features, False, device=observed_linear.weight.device, dtype=observed_linear.weight.dtype)
    linear.weight = observed_linear.weight
    linear.bias = observed_linear.bias
    linear.weight = torch.nn.Parameter(weight_quant_func(linear.weight), requires_grad=False)
    return linear


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 1, bias=False)

    def example_inputs(self, batch_size, sequence_length=10, dtype=torch.half, device="cpu"):
        return [torch.randn(1, sequence_length, self.linear1.in_features, dtype=dtype, device=device) for j in range(batch_size)]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
if __name__ == "__main__":
    for i in range(10):
        device = ("cpu")
        torch.manual_seed(i)
        dataset_size = 1000
        dtype = torch.bfloat16
        l1,l2,l3 = 512,256,128
        m = ToyLinearModel(l1,l2,l3).eval().to(dtype).to(device)
        m_bf16 = deepcopy(m)

        dataset = m.example_inputs(dataset_size,  dtype=dtype, device=device)
        calibration_data = dataset[:100]
        bf16_out = torch.cat([m_bf16(i.squeeze(0)) for i in dataset], dim=0)
        

        m_int8wo = deepcopy(m)
        quantize_(m_int8wo, int8_weight_only())
        int8wo_out = torch.cat([m_int8wo(i.squeeze(0)) for i in dataset])

        # calibrate
        insert_awq_observer(m, device)
        for example in calibration_data:
            m(example.to(device))
        # print('calibrated')

        # quantize
        is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
        quantize_(m, awq_quant, is_observed_linear)
        awq_out = torch.cat([m(i.squeeze(0)) for i in dataset])

        # compare accuracy
        awq_err = torch.sum(torch.abs(awq_out - bf16_out)).sum().item() / dataset_size
        int8wo_err = torch.sum(torch.abs(int8wo_out - bf16_out)).sum().item() / dataset_size
        print(f"AWQ error: {awq_err}")
        print(f"Int8WO error: {int8wo_err}")