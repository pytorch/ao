import pytest
import torch
from torch import nn
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.prototype.fp6_llm.fp6_llm import (
    to_tc_float6_e3m2,
    from_tc_float6_e3m2,
    _to_tc_float6_e3m2_ref,
    Fp6LlmLinear,
    convert_fp6_llm,
)
from torchao.prototype.mx_formats.custom_cast import f6_e3m2_unpacked_to_f32, f32_to_f6_e3m2_unpacked


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class TestFp6LlmLinear(TestCase):
    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)

        expected = _to_tc_float6_e3m2_ref(x)
        actual = to_tc_float6_e3m2(x)
        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_compile(self, device):
        x = torch.randn(256, 64, device=device)

        expected = to_tc_float6_e3m2(x)
        actual = torch.compile(to_tc_float6_e3m2, fullgraph=True)(x)
        torch.testing.assert_close(actual, expected)

    @parametrize("device", _DEVICES)
    def test_from_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)

        # quantize and dequantize so that the values are exactly representable in FP6
        x = f6_e3m2_unpacked_to_f32(f32_to_f6_e3m2_unpacked(x))

        actual = from_tc_float6_e3m2(to_tc_float6_e3m2(x))
        torch.testing.assert_close(actual, x)

    @parametrize("device", _DEVICES)
    def test_from_tc_float6_e3m2_compile(self, device):
        M, N = 256, 64
        x = torch.randint(256, size=(M, N * 3 // 4), dtype=torch.uint8, device=device)

        expected = from_tc_float6_e3m2(x)
        actual = torch.compile(from_tc_float6_e3m2, fullgraph=True)(x)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("leading_dims", [(4,), (2, 4)])
    @parametrize("bias", [False, True])
    def test_fp6_llm_linear_forward(self, bias, leading_dims):
        OC, IC = 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, bias=bias, device=device)
        fp6_linear = Fp6LlmLinear.from_float(linear)
        assert (fp6_linear.bias is not None) == bias

        x = torch.randn(*leading_dims, IC, device=device, dtype=torch.half)
        fp6_linear(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("bias", [False, True])
    def test_fp6_llm_linear_compile(self, bias):
        N, OC, IC = 4, 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, bias=bias, device=device)
        fp6_linear = Fp6LlmLinear.from_float(linear)

        x = torch.randn(N, IC, device=device, dtype=torch.half)
        expected = fp6_linear(x)
        actual = torch.compile(fp6_linear, fullgraph=True)(x)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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
