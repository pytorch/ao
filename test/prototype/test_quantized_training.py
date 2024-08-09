import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.prototype.quantized_training import Int8QTLinearWeight, int8_weight_only_quantized_training
from torchao.quantization.quant_api import quantize_


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestQuantizedTraining(TestCase):
    @parametrize("device", _DEVICES)
    def test_int8_stochastic_rounding(self, device):
        x = torch.randn(32, device=device)
        x_samples = x.view(1, -1).repeat(100_000, 1)

        x_int8, x_scale = Int8QTLinearWeight.quantize(x_samples, stochastic_rounding=True)
        x_dequant_samples = x_int8 * x_scale.view(-1, 1)
        x_dequant_mean = x_dequant_samples.mean(0)

        # a more rigorous test would be to do a hypothesis testing.
        # due to the statistical nature, this assertion may still fail, though very rarely.
        torch.testing.assert_close(x_dequant_mean, x, atol=1e-4, rtol=1e-4)

    @parametrize("device", _DEVICES)
    @parametrize("leading_dims", [(), (2,), (2, 4)])
    @parametrize("bias", [False, True])
    def test_int8_linear_forward(self, leading_dims, bias, device):
        embed_dim = 32

        linear_fp32 = nn.Linear(embed_dim, embed_dim * 2, bias=bias, device=device)
        linear_int8 = copy.deepcopy(linear_fp32)
        quantize_(linear_int8, int8_weight_only_quantized_training())
        assert isinstance(linear_int8.weight, Int8QTLinearWeight)

        inputs = torch.randn(leading_dims + (embed_dim,), device=device)
        out_fp32 = linear_fp32(inputs)
        out_int8 = linear_int8(inputs)
        torch.testing.assert_close(out_fp32, out_int8, atol=1e-2, rtol=1e-2)

    @parametrize("device", _DEVICES)
    def test_int8_linear_backward(self, device):
        bsize = 4
        embed_dim = 32
        n_classes = 10

        model_fp32 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 2, n_classes),
        ).to(device)
        model_int8 = copy.deepcopy(model_fp32)
        quantize_(model_int8, int8_weight_only_quantized_training())

        inputs = torch.randn(bsize, embed_dim, device=device)
        labels = torch.randint(n_classes, size=(bsize,), device=device)
        F.cross_entropy(model_fp32(inputs), labels).backward()
        F.cross_entropy(model_int8(inputs), labels).backward()

        for p_fp32, p_int8 in zip(model_fp32.parameters(), model_int8.parameters()):
            torch.testing.assert_close(p_fp32.grad, p_int8.grad, atol=1e-3, rtol=1e-2)


instantiate_parametrized_tests(TestQuantizedTraining)


if __name__ == "__main__":
    run_tests()
