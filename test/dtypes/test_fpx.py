import copy

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.dtypes.fpx import (
    FpxTensorCoreAQTLayout,
    FpxTensorCoreLayoutType,
    to_scaled_tc_fpx,
    from_scaled_tc_fpx,
)
from torchao.dtypes.fpx.fpx import _pack_tc_fpx, _pack_tc_fp6
from torchao.prototype.custom_fp_utils import _f32_to_fpx_unpacked, _fpx_unpacked_to_f32
from torchao.quantization import (
    quantize_,
    fpx_weight_only,
)

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_FPx_DTYPES = [(3, 2), (2, 2)]


class TestFpxTensorCoreAQTLayout(TestCase):
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
        from torchao.quantization.quant_primitives import (
            choose_qparams_affine_fpx,
            quantize_affine_fpx,
        )

        x = torch.randn(256, 64)
        scale = choose_qparams_affine_fpx(x, ebits, mbits)
        x = quantize_affine_fpx(x, scale, ebits, mbits)
        layout_type = FpxTensorCoreLayoutType(ebits, mbits)
        fpx_layout_tensor = FpxTensorCoreAQTLayout.from_plain(x, scale, None, layout_type).cuda()
        assert fpx_layout_tensor.device.type == "cuda"
        fpx_layout_tensor = fpx_layout_tensor.cpu()
        assert fpx_layout_tensor.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_5, reason="quantization only works with torch.compile for 2.5+")
    @parametrize("ebits,mbits", _FPx_DTYPES)
    @parametrize("bias", [False, True])
    def test_fpx_weight_only(self, ebits, mbits, bias):
        N, OC, IC = 4, 256, 64
        device = "cuda"

        linear = torch.nn.Linear(IC, OC, bias=bias, device=device, dtype=torch.half)
        fpx_linear = copy.deepcopy(linear)
        quantize_(fpx_linear, fpx_weight_only(ebits, mbits))

        x = torch.randn(N, IC, device=device, dtype=torch.half)
        expected = fpx_linear(x)
        actual = torch.compile(fpx_linear, fullgraph=True)(x)
        # somehow compile now changes the result a bit
        torch.testing.assert_close(actual, expected)


instantiate_parametrized_tests(TestFpxTensorCoreAQTLayout)


if __name__ == "__main__":
    run_tests()
