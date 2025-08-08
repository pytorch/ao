# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
import torch.nn as nn

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.prototype.float8nocompile.float8nocompile_linear_utils import (
    convert_to_float8_nocompile_training,
)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.Linear(64, 32, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@pytest.fixture
def model1():
    torch.manual_seed(0)
    return TestModel()


@pytest.fixture
def model2():
    torch.manual_seed(0)
    return TestModel()


@pytest.mark.parametrize(
    "input_shape", [(16, 32), (1, 16, 32), (2, 16, 32), (128, 8192, 32)]
)
@pytest.mark.parametrize("no_precompute_for_backward", [True, False])
def test_model_weights_and_gradients(
    model1, model2, input_shape: tuple[int, int], no_precompute_for_backward: bool
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model1 = model1.to(torch.bfloat16).to(device)
    model2 = model2.to(torch.bfloat16).to(device)

    # compare production float8 linear conversion with no-compile version
    convert_to_float8_training(model2)
    convert_to_float8_nocompile_training(
        model1, no_precompute_for_backward=no_precompute_for_backward
    )

    input_tensor = torch.randn(
        *input_shape, requires_grad=True, dtype=torch.bfloat16, device=device
    )
    input_copy1 = input_tensor.clone().detach().requires_grad_(True)
    input_copy2 = input_tensor.clone().detach().requires_grad_(True)

    loss_fn = nn.MSELoss()

    output1 = model1(input_copy1)
    output2 = model2(input_copy2)

    loss1 = loss_fn(output1, torch.zeros_like(output1))
    loss2 = loss_fn(output2, torch.zeros_like(output2))

    loss1.backward()
    loss2.backward()

    # compare the outputs, weight gradients, and input gradients
    assert torch.allclose(output1, output2, atol=0, rtol=0)
    assert torch.allclose(input_copy1.grad, input_copy2.grad, atol=0, rtol=0)
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(param1.grad, param2.grad, atol=0, rtol=0)
