from copy import deepcopy
import pytest
import torch
from torchao.quantization import quantize_
from torchao.prototype.awq.api import ObservedLinear, insert_awq_observer_, awq_uintx
from torchao.utils import TORCH_VERSION_AT_LEAST_2_3



class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 1, bias=False)

    def example_inputs(self, batch_size, sequence_length=10, dtype=torch.bfloat16, device="cuda"):
        return [torch.randn(1, sequence_length, self.linear1.in_features, dtype=dtype, device=device) for j in range(batch_size)]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    

devices = ["cuda"]
# torch.uintx dtypes are introduced in 2.3
if TORCH_VERSION_AT_LEAST_2_3:
    qdtypes = (torch.uint1, torch.uint2, torch.uint3, torch.uint4, torch.uint5, torch.uint6, torch.uint7, torch.uint8)
else:
    qdtypes = ()

idtypes = (torch.bfloat16,)#, torch.half, torch.float32)
@pytest.mark.parametrize("device", devices)   
@pytest.mark.parametrize("qdtype", qdtypes)
@pytest.mark.parametrize("idtype", idtypes)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_3,reason="torch.uint(2-7) requires torch2.3+")
def test(device, qdtype, idtype):
    dataset_size = 100
    l1,l2,l3 = 512,256,128
    original_dtype = idtype
    quant_dtype = qdtype
    group_size = 128

    m = ToyLinearModel(l1,l2,l3).eval().to(original_dtype).to(device)
    m_bf16 = deepcopy(m)

    dataset = m.example_inputs(dataset_size,  dtype=original_dtype, device=device)
    calibration_data = dataset[:50]
    bf16_out = torch.cat([m_bf16(i.squeeze(0)) for i in dataset], dim=0)

    # calibrate
    insert_awq_observer_(m, quant_dtype=quant_dtype, group_size=group_size)
    for example in calibration_data:
        m(example.to(device))
    # print('calibrated')

    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, ObservedLinear)
    quantize_(m, awq_uintx(quant_dtype = quant_dtype, group_size = group_size), is_observed_linear)
    awq_out = torch.cat([m(i.squeeze(0)) for i in dataset])
    
    assert awq_out is not None
