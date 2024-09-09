# FIXME: move this test to the appropriate test file!!!

import copy

from torchao.quantization import quantize_
from torchao.quantization.quant_api import int8_dynamic_activation_int4_weight_cutlass

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

import pytest


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 256)
        self.linear2 = torch.nn.Linear(256, 128, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


class TestS8S4LinearCUTLASS(TestCase):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_s8s4_linear_cutlass_(self):
        # FIXME: remove this!
        torch.manual_seed(0)

        dtype = torch.float16  # torch.bfloat16

        input = torch.rand((64, 128)).to(dtype).cuda()
        model = ToyModel().to(dtype).cuda()

        output_ref = model(input)

        modelq = copy.deepcopy(model)
        quantize_(modelq, int8_dynamic_activation_int4_weight_cutlass())
        output = modelq(input)

        assert torch.allclose(output, output_ref, rtol=1e-1, atol=0)


if __name__ == "__main__":
    run_tests()
