import copy
from functools import partial

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
from torchao.prototype.low_bit_optim import subclass_8bit, subclass_4bit
from torchao.utils import TORCH_VERSION_AFTER_2_3, TORCH_VERSION_AFTER_2_4

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    import lpmm
except ImportError:
    lpmm = None


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestQuantize(TestCase):
    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_correctness(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.tensor(subclass_8bit.QMAP_SIGNED, device=device)

        actual_codes, actual_scale = subclass_8bit.quantize_8bit_with_qmap(x, qmap, 256, implementation=1)
        expected_codes, expected_scale = subclass_8bit.quantize_8bit_with_qmap(x, qmap, 256, implementation=0)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_8bit_with_qmap_compile(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.tensor(subclass_8bit.QMAP_SIGNED, device=device)

        compiled_f = torch.compile(subclass_8bit.quantize_8bit_with_qmap, fullgraph=True)
        actual_codes, actual_scale = compiled_f(x, qmap, 256)
        expected_codes, expected_scale = subclass_8bit.quantize_8bit_with_qmap(x, qmap, 256)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_4bit_with_qmap_correctness(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.tensor(subclass_4bit.QMAP_SIGNED, device=device)

        actual_codes, actual_scale = subclass_4bit.quantize_4bit_with_qmap(x, qmap, 256, implementation=1)
        expected_codes, expected_scale = subclass_4bit.quantize_4bit_with_qmap(x, qmap, 256, implementation=0)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)

    @parametrize("device", _DEVICES)
    def test_quantize_4bit_with_qmap_compile(self, device):
        x = torch.randn(32, 1024, device=device)
        qmap = torch.tensor(subclass_4bit.QMAP_SIGNED, device=device)

        compiled_f = torch.compile(subclass_4bit.quantize_4bit_with_qmap, fullgraph=True)
        actual_codes, actual_scale = compiled_f(x, qmap, 256)
        expected_codes, expected_scale = subclass_4bit.quantize_4bit_with_qmap(x, qmap, 256)

        torch.testing.assert_close(actual_codes, expected_codes)
        torch.testing.assert_close(actual_scale, expected_scale)


class TestOptim(TestCase):
    @pytest.mark.skipif(bnb is None, reason="bitsandbytes is not availablle")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 8-bit Adam only works for CUDA")
    @pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
    @parametrize("optim_name", ["Adam8bit", "AdamW8bit"])
    def test_optim_8bit_correctness(self, optim_name):
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

    @pytest.mark.skipif(lpmm is None, reason="lpmm is not availablle")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="lpmm 4-bit Adam only works for CUDA")
    @pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
    @parametrize("optim_name", ["Adam4bit", "AdamW4bit"])
    def test_optim_4bit_correctness(self, optim_name):
        device = "cuda"
        model1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
        model2 = copy.deepcopy(model1)

        # lpmm doesn't have Adam. use AdamW with no weight decay instead.
        if optim_name == "Adam4bit":
            optim1 = lpmm.optim.AdamW(model1.parameters(), weight_decay=0)
        elif optim_name == "AdamW4bit":
            optim1 = lpmm.optim.AdamW(model1.parameters())
        else:
            raise ValueError(f"Unsupported {optim_name} optimizer for lpmm")
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

    @pytest.mark.xfail(not TORCH_VERSION_AFTER_2_3, reason="torch.compile() fails for PyTorch < 2.3")
    @parametrize("optim_name", ["AdamFp8", "AdamWFp8"])
    @parametrize("device", _DEVICES)
    def test_optim_fp8_smoke(self, optim_name, device):
        if device == "cuda" and torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("FP8 requires compute capability >= 8.9")
        if device == "cpu" and not TORCH_VERSION_AFTER_2_4:
            pytest.skip("fill_cpu not implemented for Float8_e4m3fn")

        model = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Linear(1024, 128)).to(device)
        optim = getattr(low_bit_optim, optim_name)(model.parameters())

        x = torch.randn(4, 32, device=device)
        loss = model(x).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()


instantiate_parametrized_tests(TestQuantize)
instantiate_parametrized_tests(TestOptim)


if __name__ == "__main__":
    run_tests()
