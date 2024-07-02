from math import log
from copy import deepcopy
import pytest

import torch

from torchao.quantization.quant_api import quantize, intx_weight_only


bit_sizes = (1,2,3,4,5,6,7)
layouts = ("plain", "packed")
group_sizes = [32,64,128]
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
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("group_size", group_sizes)
def test_intx_cpu(bit_size, layout,group_size):
    scale = 512
    fp16 = Linear16(scale, "cpu")
    fp16c = torch.compile(fp16, fullgraph=True)
    intx = deepcopy(fp16)
    intx = quantize(intx, intx_weight_only(bit_size, group_size=group_size, layout=layout))
    intx = torch.compile(intx, fullgraph=True)
    test_input = torch.randn(scale*2, dtype=torch.float16, device="cpu")
    intx.forward(test_input)

@pytest.mark.parametrize("bit_size", bit_sizes)
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("group_size", group_sizes)
def test_intx_gpu(bit_size, layout, group_size):
    scale = 512
    fp16 = Linear16(scale, "cuda")
    fp16c = torch.compile(fp16, fullgraph=True)
    intx = deepcopy(fp16)
    intx = quantize(intx, intx_weight_only(bit_size, group_size=group_size, layout=layout))
    intx = torch.compile(intx, fullgraph=True)
    test_input = torch.randn(scale*2, dtype=torch.float16).cuda()
    intx.forward(test_input)