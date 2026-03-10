import unittest

import torch

from torchao.float8.hifloat8_utils import (
    hifloat8_max_abs,
    hifloat8_min_max,
    hifloat8_scales_to_npu,
    is_hifloat8_tensor,
)
from torchao.quantization.quant_primitives import _choose_scale_float8_impl


def _npu_available() -> bool:
    try:
        import torch_npu  # type: ignore

        return torch_npu.npu.is_available()
    except Exception:
        return False


class TestHiFloat8Utils(unittest.TestCase):
    def test_hifloat8_bounds(self):
        mn, mx = hifloat8_min_max()
        self.assertAlmostEqual(mn, -32768.0)
        self.assertAlmostEqual(mx, 32769.0)
        self.assertAlmostEqual(hifloat8_max_abs(), 32769.0)

    def test_hifloat8_scale_mapping_shapes(self):
        a_scale = torch.ones((4, 1), dtype=torch.float32)
        b_scale = torch.ones((1, 6), dtype=torch.float32)
        scale, pertoken_scale = hifloat8_scales_to_npu(a_scale, b_scale, out_features=6)
        self.assertEqual(scale.shape, (6,))
        self.assertEqual(pertoken_scale.shape, (4,))

        a_scale = torch.tensor(2.0, dtype=torch.float32)
        b_scale = torch.tensor(4.0, dtype=torch.float32)
        scale, pertoken_scale = hifloat8_scales_to_npu(a_scale, b_scale, out_features=1)
        self.assertEqual(scale.shape, (1,))
        self.assertEqual(pertoken_scale.shape, (1,))


@unittest.skipUnless(_npu_available(), "torch_npu/npu not available")
class TestHiFloat8NPUIntegration(unittest.TestCase):
    def test_choose_scale_float8_impl_hifloat8(self):
        import torch_npu  # type: ignore

        x = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device="npu", dtype=torch.float16)
        scale = _choose_scale_float8_impl(
            x,
            block_size=[],
            float8_dtype=torch_npu.hifloat8,
        )
        expected = torch.tensor(
            4.0 / hifloat8_max_abs(), device="npu", dtype=torch.float32
        )
        self.assertTrue(torch.allclose(scale, expected))

    def test_float8tensor_from_hp_hifloat8(self):
        import torch_npu  # type: ignore
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )
        from torchao.quantization.granularity import PerRow

        x = torch.randn(4, 8, device="npu", dtype=torch.bfloat16)
        qt = Float8Tensor.from_hp(
            x,
            float8_dtype=torch_npu.hifloat8,
            granularity=PerRow(),
        )
        self.assertTrue(is_hifloat8_tensor(qt.qdata))
        dq = qt.dequantize()
        self.assertEqual(dq.shape, x.shape)

    def test_float8_fake_quantizer_hifloat8(self):
        import torch_npu  # type: ignore
        from torchao.quantization.qat.fake_quantize_config import (
            Float8FakeQuantizeConfig,
        )
        from torchao.quantization.qat.fake_quantizer import Float8FakeQuantizer
        from torchao.quantization.granularity import PerRow

        x = torch.randn(4, 8, device="npu", dtype=torch.float16)
        cfg = Float8FakeQuantizeConfig(dtype=torch_npu.hifloat8, granularity=PerRow())
        fq = Float8FakeQuantizer(cfg)
        y = fq(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)

    def test_float8trainingtensor_mm_uses_npu_quant_matmul(self):
        import torch_npu  # type: ignore
        from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
        from torchao.float8.float8_training_tensor import (
            GemmInputRole,
            LinearMMConfig,
        )
        from torchao.float8.config import ScalingGranularity

        x = torch.randn(4, 8, device="npu", dtype=torch.float16)
        w = torch.randn(8, 16, device="npu", dtype=torch.float16)
        mm_cfg = LinearMMConfig()

        a = hp_tensor_to_float8_dynamic(
            x,
            torch_npu.hifloat8,
            mm_cfg,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        b = hp_tensor_to_float8_dynamic(
            w,
            torch_npu.hifloat8,
            mm_cfg,
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        self.assertTrue(is_hifloat8_tensor(a._data))
        self.assertTrue(is_hifloat8_tensor(b._data))

        out = torch.mm(a, b)
        self.assertEqual(out.shape, (4, 16))
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device.type, "npu")

    def test_hifloat8_npu_matmul_numerics(self):
        import torch_npu  # type: ignore
        from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
        from torchao.float8.float8_training_tensor import (
            GemmInputRole,
            LinearMMConfig,
        )
        from torchao.float8.config import ScalingGranularity

        torch.manual_seed(0)
        x = torch.randn(16, 32, device="npu", dtype=torch.float16) * 0.1
        w = torch.randn(32, 24, device="npu", dtype=torch.float16) * 0.1
        mm_cfg = LinearMMConfig()

        a = hp_tensor_to_float8_dynamic(
            x,
            torch_npu.hifloat8,
            mm_cfg,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        b = hp_tensor_to_float8_dynamic(
            w,
            torch_npu.hifloat8,
            mm_cfg,
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )

        out = torch.mm(a, b)
        self.assertEqual(out.device.type, "npu")

        a_fp = a._data.float() / a._scale
        b_fp = b._data.float() / b._scale
        ref = torch.mm(a_fp, b_fp).to(out.dtype)

        diff = (out - ref).float()
        rel_err = diff.norm() / ref.float().norm().clamp_min(1e-6)
        max_abs_err = diff.abs().max()

        self.assertLess(rel_err.item(), 0.15)
        self.assertLess(max_abs_err.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
