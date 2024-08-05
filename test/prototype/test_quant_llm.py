import copy

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.prototype.quant_llm import (
    QuantLlmLinearWeight,
    quant_llm_fpx_weight_only,
    fp6_llm_weight_only,
    to_scaled_tc_fpx,
    from_scaled_tc_fpx,
)
from torchao.prototype.quant_llm.quant_llm import _pack_tc_fpx, _pack_tc_fp6
from torchao.prototype.custom_fp_utils import _f32_to_fpx_unpacked, _fpx_unpacked_to_f32
from torchao.quantization.quant_api import quantize_


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_FPx_DTYPES = [(3, 2), (2, 2)]


class TestQuantLlmLinearWeight(TestCase):
    @parametrize("device", _DEVICES)
    def test_pack_tc_fp6_correctness(self, device):
        x = torch.randint(256, size=(256, 64), dtype=torch.uint8, device=device)

        expected = _pack_tc_fpx(x, 6)
        actual = _pack_tc_fp6(x)
        torch.testing.assert_close(actual, expected)

    @parametrize("ebits,mbits", _FPx_DTYPES)
    @parametrize("device", _DEVICES)
    def test_to_scaled_tc_fpx_compile(self, ebits, mbits, device):
        x = torch.randn(256, 64, device=device)

        expected = to_scaled_tc_fpx(x, ebits, mbits)
        actual = torch.compile(to_scaled_tc_fpx, fullgraph=True)(x, ebits, mbits)
        torch.testing.assert_close(actual, expected)

    @parametrize("ebits,mbits", _FPx_DTYPES)
    @parametrize("device", _DEVICES)
    def test_from_tc_fpx_correctness(self, ebits, mbits, device):
        x = torch.randn(256, 64, device=device) * 100

        # quantize and dequantize so that the values are exactly representable in FPx
        x = _fpx_unpacked_to_f32(_f32_to_fpx_unpacked(x, ebits, mbits), ebits, mbits)

        tc_fpx, scale = to_scaled_tc_fpx(x, ebits, mbits)
        actual = from_scaled_tc_fpx(tc_fpx, ebits, mbits, scale=scale)
        torch.testing.assert_close(actual, x)

    @parametrize("ebits,mbits", _FPx_DTYPES)
    @parametrize("device", _DEVICES)
    def test_from_scaled_tc_fpx_compile(self, ebits, mbits, device):
        M, N = 256, 64
        nbits = 1 + ebits + mbits
        x = torch.randint(256, size=(M, N // 8 * nbits), dtype=torch.uint8, device=device)
        scale = torch.randn(M, device=device)

        expected = from_scaled_tc_fpx(x, ebits, mbits, scale)
        actual = torch.compile(from_scaled_tc_fpx, fullgraph=True)(x, ebits, mbits, scale)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", _FPx_DTYPES)
    def test_to_copy_device(self, ebits, mbits):
        x = torch.randn(256, 64)
        fpx = QuantLlmLinearWeight.from_float(x, ebits, mbits).cuda()
        assert fpx.device.type == "cuda"
        fpx = fpx.cpu()
        assert fpx.device.type == "cpu"

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
        quantize_(fpx_linear, quant_llm_fpx_weight_only(ebits, mbits))

        x = torch.randn(N, IC, device=device, dtype=torch.half)
        expected = fpx_linear(x)
        actual = torch.compile(fpx_linear, fullgraph=True)(x)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fp6_llm_quantize(self):
        N, OC, IC = 4, 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, device=device)
        fpx_linear = copy.deepcopy(linear)
        quantize_(fpx_linear, fp6_llm_weight_only())

        x = torch.randn(N, IC, device=device, dtype=torch.half)
        expected = fpx_linear(x)
        actual = torch.compile(fpx_linear, fullgraph=True)(x)
        torch.testing.assert_close(actual, expected)


instantiate_parametrized_tests(TestQuantLlmLinearWeight)


if __name__ == "__main__":
    run_tests()
