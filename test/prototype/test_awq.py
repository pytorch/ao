from copy import deepcopy
import os
import pytest
import torch
from torchao.quantization import quantize_

from torchao.utils import TORCH_VERSION_AT_LEAST_2_3
if TORCH_VERSION_AT_LEAST_2_3:
    from torchao.prototype.awq import insert_awq_observer_, awq_uintx, AWQObservedLinear

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
    qdtypes = (torch.uint3, torch.uint4,  torch.uint8)
else:
    qdtypes = ()
    
@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset() # reset cache between tests
    
idtypes = (torch.half, torch.bfloat16)#, torch.half, torch.float32)
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
    n_calibration_examples = 10
    n_validation_examples = 10
    sequence_length = 5

    m = ToyLinearModel(l1,l2,l3).eval().to(original_dtype).to(device)
    m_bf16 = deepcopy(m)

    dataset = m.example_inputs(dataset_size, sequence_length=sequence_length,  dtype=original_dtype, device=device)
    calibration_data = dataset[:n_calibration_examples]
    bf16_out = torch.cat([m_bf16(i.squeeze(0)) for i in dataset], dim=0)

    # calibrate
    insert_awq_observer_(m, n_validation_examples, sequence_length, quant_dtype=quant_dtype, group_size=group_size)

    for example in calibration_data:
        m(example.to(device))

    
    # quantize
    is_observed_linear = lambda m, fqn: isinstance(m, AWQObservedLinear)
    quantize_(m, awq_uintx(quant_dtype = quant_dtype, group_size = group_size), is_observed_linear)
    
    model_save_path = "awq_model.pt"
    torch.save(m, model_save_path)
    loaded_model = torch.load(model_save_path)
    os.remove(model_save_path)
    
    m = torch.compile(m, fullgraph=True)
    loaded_model = torch.compile(loaded_model, fullgraph=True)
    awq_out = torch.cat([m(i.squeeze(0)) for i in dataset])
    awq_save_load_out = torch.cat([loaded_model(i.squeeze(0)) for i in dataset])
    
    assert awq_out is not None
    assert awq_save_load_out is not None
    assert torch.allclose(awq_out, awq_save_load_out, atol = 1e-2)
    # remove equalization scale file
    
