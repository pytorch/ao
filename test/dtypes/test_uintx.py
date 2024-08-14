from math import log
from copy import deepcopy
import pytest

import torch

from torchao.dtypes.uintx.Uintx import to_uintx
from torchao.quantization.quant_api import quantize_, uintx_weight_only
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

from torchao.quantization.quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
)

bit_widths = (1, 2, 3, 4, 5, 6, 7)
group_sizes = [32, 64, 128]
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

@pytest.mark.parametrize("bit_width", bit_widths)
@pytest.mark.parametrize("group_size", group_sizes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_5, reason="only works with fix in the nightly build")
def test_uintx_weight_only_model_quant(bit_width, group_size, device):
    scale = 512
    fp16 = Linear16(scale, device)
    quantize_(fp16, uintx_weight_only(bit_width, group_size=group_size))
    uintx = torch.compile(fp16, fullgraph=True)
    test_input = torch.randn(scale*2, dtype=torch.float16, device=device)
    output = uintx.forward(test_input)
    assert output != None, "model quantization failed"

@pytest.mark.parametrize("bit_width", bit_widths)
@pytest.mark.parametrize("group_size", group_sizes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_5, reason="only works with fix in the nightly build")
def test_uintx_weight_only_quant(bit_width, group_size, device):
    input_float = torch.randn((1, 256), dtype=torch.float16, device = device)
    mapping_type = MappingType.SYMMETRIC
    quant_min = 0
    quant_max = 2 ** bit_width - 1
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int32
    zero_point_domain = ZeroPointDomain.INT
    target_dtype = torch.uint8
    block_size = (1, group_size)

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

    q =  to_uintx(aqt, bit_width, -1)
    assert q != None, "quantization failed"
    deqaunt = dequantize_affine(
        q, block_size, scale,
        zero_point, target_dtype,
        quant_min = quant_min,
        quant_max = quant_max,
        zero_point_domain = zero_point_domain
    )
    assert deqaunt != None, "deqauntization failed"
