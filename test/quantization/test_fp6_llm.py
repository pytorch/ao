import pytest
import torch
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.dtypes.float6_e3m2 import to_float6_e3m2, from_float6_e3m2
from torchao.quantization.fp6_llm import to_tc_float6_e3m2, from_tc_float6_e3m2, Fp6LlmLinear, convert_fp6_llm
from torchao.ops import prepack_fp6_weight


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestFp6LlmLinear(TestCase):
    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)

        expected = prepack_fp6_weight(to_float6_e3m2(x.cpu()).view(torch.int32)).view(torch.uint8)
        actual = to_tc_float6_e3m2(x)
        torch.testing.assert_close(actual.view(-1).cpu(), expected.view(-1))

    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_compile(self, device):
        x = torch.randn(256, 64, device=device)

        expected = to_tc_float6_e3m2(x)
        actual = torch.compile(to_tc_float6_e3m2)(x)
        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_from_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)
        x = from_float6_e3m2(to_float6_e3m2(x))  # quantize and dequantize so that the values are exactly representable in FP6

        actual = from_tc_float6_e3m2(to_tc_float6_e3m2(x), *x.shape)
        torch.testing.assert_close(actual, x)

    @parametrize("device", _DEVICES)
    def test_from_tc_float6_e3m2_compile(self, device):
        M, N = 256, 64
        x = torch.randint(256, size=(M * N * 3 // 4,), dtype=torch.uint8, device=device)

        expected = from_tc_float6_e3m2(x, M, N)
        actual = torch.compile(from_tc_float6_e3m2)(x, M, N)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("bias", [False, True])
    def test_fp6_llm_linear_forward(self, bias):
        N, OC, IC = 4, 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, bias=bias, device=device)
        fp6_linear = Fp6LlmLinear.from_float(linear)
        assert (fp6_linear.bias is not None) == bias

        x = torch.randn(N, IC, device=device)
        fp6_linear(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("bias", [False, True])
    def test_fp6_llm_linear_compile(self, bias):
        N, OC, IC = 4, 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, bias=bias, device=device)
        fp6_linear = Fp6LlmLinear.from_float(linear)

        x = torch.randn(N, IC, device=device)
        expected = fp6_linear(x)
        actual = torch.compile(fp6_linear)(x)
        torch.testing.assert_close(actual, expected)

    def test_convert_fp6_llm(self):
        device = "cuda"
        model = nn.Sequential(nn.Linear(64, 256, bias=False), nn.Linear(256, 256)).to(device)
        convert_fp6_llm(model)

        assert isinstance(model[0], Fp6LlmLinear)
        assert model[0].bias is None
        assert isinstance(model[1], Fp6LlmLinear)
        assert model[1].bias is not None

        x = torch.randn(4, 64, device=device)
        model(x)


instantiate_parametrized_tests(TestFp6LlmLinear)


if __name__ == "__main__":
    run_tests()
