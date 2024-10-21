import copy

import pytest
import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torchao.dtypes.floatx import (
    FloatxTensorCoreAQTTensorImpl,
    FloatxTensorCoreLayout,
    to_scaled_tc_floatx,
    from_scaled_tc_floatx,
)
from torchao.dtypes.floatx.floatx import _pack_tc_floatx, _pack_tc_fp6
from torchao.prototype.custom_fp_utils import _f32_to_floatx_unpacked, _floatx_unpacked_to_f32
from torchao.quantization import (
    quantize_,
    fpx_weight_only,
)

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, is_fbcode


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_Floatx_DTYPES = [(3, 2), (2, 2)]


class TestFloatxTensorCoreAQTTensorImpl(TestCase):
    @parametrize("device", _DEVICES)
    def test_pack_tc_fp6_correctness(self, device):
        x = torch.randint(256, size=(256, 64), dtype=torch.uint8, device=device)

        expected = _pack_tc_floatx(x, 6)
        actual = _pack_tc_fp6(x)
        torch.testing.assert_close(actual, expected)

    @parametrize("ebits,mbits", _Floatx_DTYPES)
    @parametrize("device", _DEVICES)
    def test_to_scaled_tc_floatx_compile(self, ebits, mbits, device):
        x = torch.randn(256, 64, device=device)

        expected = to_scaled_tc_floatx(x, ebits, mbits)
        actual = torch.compile(to_scaled_tc_floatx, fullgraph=True)(x, ebits, mbits)
        torch.testing.assert_close(actual, expected)

    @parametrize("ebits,mbits", _Floatx_DTYPES)
    @parametrize("device", _DEVICES)
    def test_from_tc_floatx_correctness(self, ebits, mbits, device):
        x = torch.randn(256, 64, device=device) * 100

        # quantize and dequantize so that the values are exactly representable in Floatx
        x = _floatx_unpacked_to_f32(_f32_to_floatx_unpacked(x, ebits, mbits), ebits, mbits)

        tc_floatx, scale = to_scaled_tc_floatx(x, ebits, mbits)
        actual = from_scaled_tc_floatx(tc_floatx, ebits, mbits, scale=scale)
        torch.testing.assert_close(actual, x)

    @parametrize("ebits,mbits", _Floatx_DTYPES)
    @parametrize("device", _DEVICES)
    def test_from_scaled_tc_floatx_compile(self, ebits, mbits, device):
        M, N = 256, 64
        nbits = 1 + ebits + mbits
        x = torch.randint(256, size=(M, N // 8 * nbits), dtype=torch.uint8, device=device)
        scale = torch.randn(M, device=device)

        expected = from_scaled_tc_floatx(x, ebits, mbits, scale)
        actual = torch.compile(from_scaled_tc_floatx, fullgraph=True)(x, ebits, mbits, scale)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", _Floatx_DTYPES)
    def test_to_copy_device(self, ebits, mbits):
        from torchao.quantization.quant_primitives import (
            choose_qparams_affine_floatx,
            quantize_affine_floatx,
        )

        x = torch.randn(256, 64)
        scale = choose_qparams_affine_floatx(x, ebits, mbits)
        x = quantize_affine_floatx(x, scale, ebits, mbits)
        _layout = FloatxTensorCoreLayout(ebits, mbits)
        floatx_tensor_impl = FloatxTensorCoreAQTTensorImpl.from_plain(x, scale, None, _layout).cuda()
        assert floatx_tensor_impl.device.type == "cuda"
        floatx_tensor_impl = floatx_tensor_impl.cpu()
        assert floatx_tensor_impl.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TORCH_VERSION_AT_LEAST_2_5, reason="quantization only works with torch.compile for 2.5+")
    @parametrize("ebits,mbits", _Floatx_DTYPES)
    @parametrize("bias", [False, True])
    @pytest.mark.skipif(is_fbcode(), reason="broken in fbcode")
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


instantiate_parametrized_tests(TestFloatxTensorCoreAQTTensorImpl)


if __name__ == "__main__":
    run_tests()
