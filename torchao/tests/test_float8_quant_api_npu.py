import unittest
import warnings
from unittest.mock import patch

import torch

import torchao.float8.inference as float8_inference
from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
from torchao.float8.float8_training_tensor import (
    GemmInputRole,
    LinearMMConfig,
)
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8StaticActivationFloat8WeightConfig,
    quantize_,
)


def _npu_available() -> bool:
    try:
        import torch_npu  # type: ignore

        return torch_npu.npu.is_available()
    except Exception:
        return False


@unittest.skipUnless(_npu_available(), "torch_npu/npu not available")
class TestFloat8QuantApiNPU(unittest.TestCase):
    @staticmethod
    def _make_linear():
        model = torch.nn.Linear(
            32, 32, bias=False, device="npu", dtype=torch.bfloat16
        ).eval()
        x = torch.randn(4, 32, device="npu", dtype=torch.bfloat16)
        return model, x

    def _assert_standard_fp8_matmul_kwargs(self, call_kwargs):
        self.assertIsNone(call_kwargs.get("x1_dtype"))
        self.assertIsNone(call_kwargs.get("x2_dtype"))

    def _assert_standard_fp8_inputs(self, call_args):
        x1 = call_args[0]
        x2 = call_args[1]

        expected_std_fp8_dtypes = {
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        }
        expected_std_fp8_dtypes.discard(None)
        self.assertIn(x1.dtype, expected_std_fp8_dtypes)
        self.assertIn(x2.dtype, expected_std_fp8_dtypes)

    def test_dynamic_activation_float8_weight_on_npu(self):
        import torch_npu  # type: ignore

        model, x = self._make_linear()
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(set_inductor_config=False),
        )
        with patch(
            "torch_npu.npu_quant_matmul", wraps=torch_npu.npu_quant_matmul
        ) as qmm_mock:
            with patch(
                "torch._scaled_mm",
                side_effect=RuntimeError("_scaled_mm is forbidden on NPU fp8 path"),
            ):
                out = model(x)
        self.assertGreater(qmm_mock.call_count, 0)
        self._assert_standard_fp8_matmul_kwargs(qmm_mock.call_args.kwargs)
        self._assert_standard_fp8_inputs(qmm_mock.call_args.args)
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(out.device.type, "npu")
        self.assertEqual(out.dtype, x.dtype)

    def test_static_activation_float8_weight_on_npu(self):
        import torch_npu  # type: ignore

        model, x = self._make_linear()
        quantize_(
            model,
            Float8StaticActivationFloat8WeightConfig(
                scale=torch.tensor([1.0], device="npu", dtype=torch.float32),
                set_inductor_config=False,
            ),
        )
        with patch(
            "torch_npu.npu_quant_matmul", wraps=torch_npu.npu_quant_matmul
        ) as qmm_mock:
            with patch(
                "torch._scaled_mm",
                side_effect=RuntimeError("_scaled_mm is forbidden on NPU fp8 path"),
            ):
                out = model(x)
        self.assertGreater(qmm_mock.call_count, 0)
        self._assert_standard_fp8_matmul_kwargs(qmm_mock.call_args.kwargs)
        self._assert_standard_fp8_inputs(qmm_mock.call_args.args)
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(out.device.type, "npu")
        self.assertEqual(out.dtype, x.dtype)

    def test_float8_training_mm_on_npu_without_scaled_mm(self):
        import torch_npu  # type: ignore

        x = torch.randn(4, 16, device="npu", dtype=torch.float16)
        w = torch.randn(16, 32, device="npu", dtype=torch.float16)
        mm_cfg = LinearMMConfig()

        a = hp_tensor_to_float8_dynamic(
            x,
            torch.float8_e4m3fn,
            mm_cfg,
            gemm_input_role=GemmInputRole.INPUT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )
        b = hp_tensor_to_float8_dynamic(
            w,
            torch.float8_e4m3fn,
            mm_cfg,
            gemm_input_role=GemmInputRole.WEIGHT,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        )

        with patch(
            "torch_npu.npu_quant_matmul", wraps=torch_npu.npu_quant_matmul
        ) as qmm_mock:
            with patch(
                "torch._scaled_mm",
                side_effect=RuntimeError("_scaled_mm is forbidden on NPU fp8 path"),
            ):
                out = torch.mm(a, b)
        self.assertGreater(qmm_mock.call_count, 0)
        self._assert_standard_fp8_matmul_kwargs(qmm_mock.call_args.kwargs)
        self._assert_standard_fp8_inputs(qmm_mock.call_args.args)
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(out.device.type, "npu")

    def test_standard_float8_fallback_warns_when_npu_quant_matmul_unsupported(self):
        model, x = self._make_linear()
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(set_inductor_config=False),
        )

        float8_inference._WARNED_STANDARD_FP8_NPU_FALLBACK = False
        with patch(
            "torch_npu.npu_quant_matmul",
            side_effect=RuntimeError("mock unsupported standard float8 dtype"),
        ):
            with patch(
                "torch._scaled_mm",
                side_effect=RuntimeError("_scaled_mm is forbidden on NPU fp8 path"),
            ):
                with warnings.catch_warnings(record=True) as warns:
                    warnings.simplefilter("always")
                    out = model(x)
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(out.device.type, "npu")
        self.assertTrue(
            any(
                "falls back to explicit dequantize + torch.mm" in str(w.message)
                for w in warns
            )
        )


if __name__ == "__main__":
    unittest.main()
