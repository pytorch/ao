import pytest
import torch
import torch.nn as nn

from torchao.prototype.float8nocompile.float8nocompile_linear_utils import (
    convert_to_float8_nocompile_training,
)

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

if not TORCH_VERSION_AT_LEAST_2_5:
    raise AssertionError("torchao.float8 requires PyTorch version 2.5 or greater")


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


def test_model_weights_and_gradients(model1, model2):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model1 = model1.to(torch.bfloat16).to(device)
    model2 = model2.to(torch.bfloat16).to(device)

    # swap nn.Linear layers with Float8LinearNoCompile layers in model1 only
    convert_to_float8_nocompile_training(model1)

    input_data = torch.randn(16, 32, dtype=torch.bfloat16).to(device)
    loss_fn = nn.MSELoss()

    output1 = model1(input_data)
    output2 = model2(input_data)

    loss1 = loss_fn(output1, torch.zeros_like(output1))
    loss2 = loss_fn(output2, torch.zeros_like(output2))

    loss1.backward()
    loss2.backward()

    # compare the weights and gradients of both models
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(param1.data, param2.data, atol=0, rtol=0)
        assert torch.allclose(param1.grad, param2.grad, atol=1e-3, rtol=1e-3)
