# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchao.prototype.autoround.utils import is_auto_round_available

if not is_auto_round_available():
    pytest.skip("AutoRound is not available", allow_module_level=True)

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao import quantize_
from torchao.dtypes import AffineQuantizedTensor
from torchao.prototype.autoround.core import (
    apply_auto_round,
    prepare_model_for_applying_auto_round_,
)
from torchao.prototype.autoround.multi_tensor import MultiTensor

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


class ModelWithInplaceOp(torch.nn.Module):
    def __init__(self, DIM=128):
        super().__init__()
        self.lin = torch.nn.Linear(DIM, DIM)
        self.register_buffer("other", torch.zeros(DIM, DIM))

    def forward(self, x, idx):
        x = x + self.lin(x)
        # update buffer
        self.other[idx] = x
        return x


class M2(torch.nn.Module):
    def __init__(self, DIM=128):
        super().__init__()
        self.m1 = ModelWithInplaceOp(DIM)
        self.m2 = ModelWithInplaceOp(DIM)

    def forward(self, x, idx):
        x = self.m1(x, idx)
        x = self.m2(x, idx)
        return x


def _check_params_and_buffers_type(module, check_fun):
    return [check_fun(p) for p in module.parameters()] + [
        check_fun(b) for b in module.buffers()
    ]


class TestAutoRound(TestCase):
    @pytest.mark.skip("these tests are broken on main branch")
    @parametrize("device", _AVAILABLE_DEVICES)
    @torch.no_grad()
    def test_auto_round(self, device: str):
        example_inputs = (
            torch.randn(32, 64).to(device),
            torch.randn(32, 64).to(device),
        )
        m = M().eval().to(device)
        m(*example_inputs)
        prepare_model_for_applying_auto_round_(
            m,
            is_target_module=_is_two_linear,
            bits=7,
            group_size=32,
            iters=20,
            device=device,
        )
        assert all(
            _check_params_and_buffers_type(m, lambda x: isinstance(x, MultiTensor))
        ), "Expected all parameters and buffers to be `MultiTensor`."
        input1 = []
        input2 = []
        for _ in range(10):
            input1.append(torch.randn(32, 64).to(device))
            input2.append(torch.randn(32, 64).to(device))

        mt_input1 = MultiTensor(input1)
        mt_input2 = MultiTensor(input2)
        out = m(mt_input1, mt_input2)
        assert isinstance(out, MultiTensor), f"Expected MultiTensor, got {type(out)}"
        assert all(
            _check_params_and_buffers_type(m, lambda x: not isinstance(x, MultiTensor))
        ), "Expected all parameters and buffers have been converted back to tensor."
        quantize_(m, apply_auto_round(), _is_two_linear, device=device)
        for l in m.modules():
            if isinstance(l, torch.nn.Linear):
                assert isinstance(l.weight, AffineQuantizedTensor)
        after_quant = m(*example_inputs)
        assert after_quant is not None, "Quantized model forward pass failed"

    @pytest.mark.skip("these tests are broken on main branch")
    @parametrize("device", _AVAILABLE_DEVICES)
    @torch.no_grad()
    def test_wrap_model_with_multi_tensor(self, device: str):
        _is_model_with_inplace_op = lambda mod, fqn: isinstance(mod, ModelWithInplaceOp)

        DIM = 128
        m = M2(DIM).eval().to(device)
        prepare_model_for_applying_auto_round_(
            m,
            is_target_module=_is_model_with_inplace_op,
            bits=7,
            group_size=32,
            iters=20,
            device=device,
        )
        assert all(
            _check_params_and_buffers_type(m, lambda x: isinstance(x, MultiTensor))
        ), "Expected all parameters and buffers to be `MultiTensor`."
        input1 = []
        input2 = []
        for _ in range(2):
            input1.append(torch.randn(DIM, DIM).to(device))
            input2.append(torch.randint(0, DIM, (DIM,), dtype=torch.long).to(device))

        mt_input1 = MultiTensor(input1)
        mt_input2 = MultiTensor(input2)
        out = m(mt_input1, mt_input2)
        assert isinstance(out, MultiTensor), f"Expected MultiTensor, got {type(out)}"
        assert all(
            _check_params_and_buffers_type(m, lambda x: not isinstance(x, MultiTensor))
        ), "Expected all parameters and buffers have been converted back to tensor."
        quantize_(m, apply_auto_round(), _is_model_with_inplace_op, device=device)
        for l in m.modules():
            if isinstance(l, torch.nn.Linear):
                assert isinstance(l.weight, AffineQuantizedTensor)
        after_quant = m(input1[0], input2[0])
        assert after_quant is not None, "Quantized model forward pass failed"


instantiate_parametrized_tests(TestAutoRound)

if __name__ == "__main__":
    run_tests()
