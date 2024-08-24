from copy import deepcopy
import torch
import torch.nn.functional as F
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization import quantize_, int8_weight_only
from torchao.prototype.awq.api import ObservedLinear, insert_awq_observer, awq_quant

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
    

device = ("cpu")
torch.manual_seed(34)
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
insert_awq_observer(m, dtype, device)
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