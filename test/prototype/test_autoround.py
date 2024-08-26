import pytest
from torchao.prototype.autoround.utils import is_auto_round_available

if not is_auto_round_available():
    pytest.skip("AutoRound is not available", allow_module_level=True)

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torchao import quantize_

from torchao.dtypes import AffineQuantizedTensor
from torchao.prototype.autoround.core import (
    apply_auto_round,
    prepare_model_for_applying_auto_round_,
)
from torchao.prototype.autoround.multi_tensor import MultiTensor
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

_AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# Copied from https://github.com/pytorch/ao/pull/721
class TwoLinear(torch.nn.Module):
    def __init__(self, in_features=64, out_features=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.linear2 = torch.nn.Linear(in_features, out_features)

    def forward(self, x, y):
        x = self.linear1(x)
        y = self.linear2(y)
        return x + y


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.two_linear1 = TwoLinear()
        self.two_linear2 = TwoLinear(128, 256)

    def forward(self, x, y):
        x1 = self.two_linear1(x, y)
        x2 = self.two_linear2(x1, x1)
        return x2


def _is_two_linear(mod, fqn):
    return isinstance(mod, TwoLinear)


class TestAutoRound(TestCase):

    @pytest.mark.skip(not TORCH_VERSION_AT_LEAST_2_5, "Requires torch 2.5 or later")
    @parametrize("device", _AVAILABLE_DEVICES)
    @torch.no_grad()
    def test_auto_round(self, device: str):
        example_inputs = (
            torch.randn(32, 64).to(device),
            torch.randn(32, 64).to(device),
        )
        m = M().eval().to(device)
        before_quant = m(*example_inputs)
        prepare_model_for_applying_auto_round_(
            m,
            is_target_module=_is_two_linear,
            bits=7,
            group_size=32,
            iters=20,
            device=device,
        )
        input1 = []
        input2 = []
        for _ in range(10):
            input1.append(torch.randn(32, 64).to(device))
            input2.append(torch.randn(32, 64).to(device))

        mt_input1 = MultiTensor(input1)
        mt_input2 = MultiTensor(input2)
        out = m(mt_input1, mt_input2)
        quantize_(m, apply_auto_round(), _is_two_linear, device=device)
        for l in m.modules():
            if isinstance(l, torch.nn.Linear):
                assert isinstance(l.weight, AffineQuantizedTensor)
        after_quant = m(*example_inputs)
        assert after_quant is not None, "Quantized model forward pass failed"


instantiate_parametrized_tests(TestAutoRound)

if __name__ == "__main__":
    run_tests()
