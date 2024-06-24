import copy

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.prototype.fp6_llm.fp6_llm import (
    QuantLlmLinearWeight,
    quant_llm_fpx_weight_only,
    to_tc_float6_e3m2,
    from_tc_float6_e3m2,
    _to_tc_fpx,
)
from torchao.prototype.mx_formats.custom_cast import f6_e3m2_unpacked_to_f32, f32_to_f6_e3m2_unpacked
from torchao.quantization.quant_api import quantize


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_FPx_DTYPES = [(3, 2), (2, 2)]


class TestQuantLlmLinear(TestCase):
    @parametrize("device", _DEVICES)
    def test_to_tc_float6_e3m2_correctness(self, device):
        x = torch.randn(256, 64, device=device)

        expected = _to_tc_fpx(x, 3, 2)
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
    @parametrize("ebits,mbits", _FPx_DTYPES)
    @parametrize("leading_dims", [(4,), (2, 4)])
    @parametrize("bias", [False, True])
    def test_quant_llm_linear_weight(self, ebits, mbits, bias, leading_dims):
        OC, IC = 256, 64
        device = "cuda"

        fp16_weight = torch.randn(OC, IC, device=device, dtype=torch.half)
        fp16_bias = torch.randn(OC, device=device, dtype=torch.half) if bias else None

        fpx_weight = QuantLlmLinearWeight.from_float(fp16_weight, ebits, mbits)

        x = torch.randn(*leading_dims, IC, device=device, dtype=torch.half)
        out = torch.nn.functional.linear(x, fpx_weight, fp16_bias)
        assert out.shape == leading_dims + (OC,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", _FPx_DTYPES)
    @parametrize("bias", [False, True])
    def test_quant_llm_quantize(self, ebits, mbits, bias):
        N, OC, IC = 4, 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, bias=bias, device=device)
        fpx_linear = copy.deepcopy(linear)
        quantize(fpx_linear, quant_llm_fpx_weight_only(ebits, mbits))

        x = torch.randn(N, IC, device=device, dtype=torch.half)
        expected = fpx_linear(x)
        actual = torch.compile(fpx_linear, fullgraph=True)(x)
        torch.testing.assert_close(actual, expected)


instantiate_parametrized_tests(TestQuantLlmLinear)


if __name__ == "__main__":
    run_tests()
