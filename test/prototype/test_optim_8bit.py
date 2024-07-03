import copy

import pytest
import torch
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.prototype import low_bit_optim
from torchao.prototype.low_bit_optim.subclass_8bit import quantize_8bit_with_qmap, QMAP_SIGNED
from torchao.utils import TORCH_VERSION_AFTER_2_3

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestDTQ8bit(TestCase):
    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_correctness(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.tensor(QMAP_SIGNED, device=device)

        actual_codes, actual_scale = quantize_8bit_with_qmap(x, qmap, 256, implementation=1)
        expected_codes, expected_scale = quantize_8bit_with_qmap(x, qmap, 256, implementation=0)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_compile(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.tensor(QMAP_SIGNED, device=device)

        actual_codes, actual_scale = torch.compile(quantize_8bit_with_qmap, fullgraph=True)(x, qmap, 256)
        expected_codes, expected_scale = quantize_8bit_with_qmap(x, qmap, 256)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)


@pytest.mark.skipif(bnb is None, reason="bitsandbytes is not availablle")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 8-bit Adam only works for CUDA")
@pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
class TestOptim8bit(TestCase):
    @parametrize("optim_name", ["Adam8bit", "AdamW8bit"])
    def test_adam_8bit_correctness(self, optim_name):
        device = "cuda"
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
        model2 = copy.deepcopy(model1)

        optim1 = getattr(bnb.optim, optim_name)(model1.parameters())
        optim2 = getattr(low_bit_optim, optim_name)(model2.parameters())

        for _ in range(2):
            x = torch.randn(4, 32, device=device)

            loss1 = model1(x).sum()
            loss1.backward()
            optim1.step()
            optim1.zero_grad()

            loss2 = model2(x).sum()
            loss2.backward()
            optim2.step()
            optim2.zero_grad()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(p2, p1, rtol=1e-5, atol=1e-5)


instantiate_parametrized_tests(TestDTQ8bit)
instantiate_parametrized_tests(TestOptim8bit)


if __name__ == "__main__":
    run_tests()
