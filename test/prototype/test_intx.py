from math import log
from copy import deepcopy
import pytest

import torch

from torchao.prototype.intx import intx_affine_weight_only
from torchao.quantization.quant_api import quantize_
from torchao.utils import TORCH_VERSION_AFTER_2_5

bit_sizes = (1,2,3,4,5,6,7)
group_sizes = [32,64,128]
devices = ["cpu", "cuda"]
@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset() # reset cache between tests



class Linear16(torch.nn.Module):
    def __init__(self, scale, device):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(scale * 2, scale, bias=False, dtype=torch.float16, device=device),
            torch.nn.Linear(scale, scale, bias=False, dtype=torch.float16, device=device),
            torch.nn.Linear(scale, scale//2, bias=False, dtype=torch.float16, device=device),
        )

    def forward(self, x):
        return self.net(x)
    
@pytest.mark.parametrize("bit_size", bit_sizes)
@pytest.mark.parametrize("group_size", group_sizes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  
@pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="only works with fix in the nightly build")
def test_intx_affine_weight_only_model_quant(bit_size, group_size, device):
    scale = 512
    fp16 = Linear16(scale, device)
    intx = quantize_(fp16, intx_weight_only(bit_size, group_size=group_size))
    intx = torch.compile(intx, fullgraph=True)
    test_input = torch.randn(scale*2, dtype=torch.float16, device=device)
    output = intx.forward(test_input)
    assert output, "model quantization failed"
    
@pytest.mark.parametrize("bit_size", bit_sizes)
@pytest.mark.parametrize("group_size", group_sizes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  
@pytest.mark.skipif(TORCH_VERSION_AFTER_2_5, reason="only works with fix in the nightly build")
def test_intx_affine_weight_only_quant(bit_size): 
    input_float = torch.randn((1,8), dtype=torch.float16)
    print('input_float', input_float)
    mapping_type = MappingType.SYMMETRIC
    quant_min = 0
    quant_max = 2**bit_size - 1
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int32
    zero_point_domain = ZeroPointDomain.INT
    target_dtype = torch.uint8
    block_size = (1, input_float.shape[1])
    
    scale, zero_point = choose_qparams_affine(
            input_float, mapping_type, block_size,
            target_dtype, quant_min, quant_max, eps, torch.float32, 
            zero_point_dtype, True, zero_point_domain
    )
    
    aqt = quantize_affine(
        input_float, block_size, scale,
        zero_point, target_dtype,
        quant_min = quant_min,
        quant_max = quant_max,
        zero_point_domain = zero_point_domain
        )
        
    q =  to_intx(aqt, bit_size, -1)
    assert q, "quantization failed"
    deqaunt = dequantize_affine(
        q, block_size, scale,
        zero_point, target_dtype,
        quant_min = quant_min,
        quant_max = quant_max,
        zero_point_domain = zero_point_domain
    )
    assert deqaunt, "deqauntization failed"
