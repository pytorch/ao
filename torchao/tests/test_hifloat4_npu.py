import unittest

import torch

from torchao.float8.hifloat8_utils import (
    hifloat4_max_abs,
    hifloat4_min_max,
    hifloat4_scales_to_npu,
    is_hifloat4_tensor,
)
from torchao.quantization.quant_primitives import _choose_scale_float8_impl


def _npu_hifloat4_available() -> bool:
    try:
        import torch_npu  # type: ignore

        return torch_npu.npu.is_available() and hasattr(torch_npu, "float4_e2m1fn_x2")
    except Exception:
        return False


def _get_hifloat4_dtype():
    import torch_npu  # type: ignore

    return torch_npu.float4_e2m1fn_x2


class TestHiFloat4Utils(unittest.TestCase):
    def test_hifloat4_bounds(self):
        mn, mx = hifloat4_min_max()
        self.assertLess(mn, 0.0)
        self.assertGreater(mx, 0.0)
        self.assertAlmostEqual(hifloat4_max_abs(), max(abs(mn), abs(mx)))

    def test_hifloat4_scale_mapping_shapes(self):
        a_scale = torch.ones((4, 1), dtype=torch.float32)
        b_scale = torch.ones((1, 6), dtype=torch.float32)
        scale, pertoken_scale = hifloat4_scales_to_npu(a_scale, b_scale, out_features=6)
        self.assertEqual(scale.shape, (6,))
        self.assertEqual(pertoken_scale.shape, (4,))

        a_scale = torch.tensor(2.0, dtype=torch.float32)
        b_scale = torch.tensor(4.0, dtype=torch.float32)
        scale, pertoken_scale = hifloat4_scales_to_npu(a_scale, b_scale, out_features=1)
        self.assertEqual(scale.shape, (1,))
        self.assertEqual(pertoken_scale.shape, (1,))


@unittest.skipUnless(_npu_hifloat4_available(), "torch_npu float4_e2m1fn_x2 not available")
class TestHiFloat4NPUIntegration(unittest.TestCase):
    def test_choose_scale_float8_impl_hifloat4(self):
        hifloat4_dtype = _get_hifloat4_dtype()
        x = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device="npu", dtype=torch.float16)
        scale = _choose_scale_float8_impl(
            x,
            block_size=[],
            float8_dtype=hifloat4_dtype,
        )
        expected = torch.tensor(
            4.0 / hifloat4_max_abs(), device="npu", dtype=torch.float32
        )
        self.assertTrue(torch.allclose(scale, expected))

    def test_float8tensor_from_hp_hifloat4(self):
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )
        from torchao.quantization.granularity import PerRow

        hifloat4_dtype = _get_hifloat4_dtype()
        x = torch.randn(4, 8, device="npu", dtype=torch.bfloat16)
        qt = Float8Tensor.from_hp(
            x,
            float8_dtype=hifloat4_dtype,
            granularity=PerRow(),
        )
        self.assertTrue(is_hifloat4_tensor(qt.qdata))
        dq = qt.dequantize()
        self.assertEqual(dq.shape, x.shape)

    def test_float8trainingtensor_mm_hifloat4(self):
        from torchao.float8.config import ScalingGranularity
        from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
        from torchao.float8.float8_training_tensor import (
            GemmInputRole,
            LinearMMConfig,
        )

        hifloat4_dtype = _get_hifloat4_dtype()
        x = torch.randn(4, 8, device="npu", dtype=torch.float16)
        w = torch.randn(8, 16, device="npu", dtype=torch.float16)
        mm_cfg = LinearMMConfig()

        a = hp_tensor_to_float8_dynamic(
            x,
            hifloat4_dtype,
            mm_cfg,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        b = hp_tensor_to_float8_dynamic(
            w,
            hifloat4_dtype,
            mm_cfg,
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        self.assertTrue(is_hifloat4_tensor(a._data))
        self.assertTrue(is_hifloat4_tensor(b._data))

        out = torch.mm(a, b)
        self.assertEqual(out.shape, (4, 16))
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device.type, "npu")


if __name__ == "__main__":
    unittest.main()
