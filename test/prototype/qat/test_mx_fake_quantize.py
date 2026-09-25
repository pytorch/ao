# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.qat import (
    MXFakeQuantizeConfig,
    mx_fake_quantize,
    mx_fake_quantized_grouped_mm,
)
from torchao.quantization.quantize_.common import KernelPreference


class MXFakeQuantizeTest(unittest.TestCase):
    def _assert_matches_reference(
        self,
        value: torch.Tensor,
        config: MXFakeQuantizeConfig,
    ) -> torch.Tensor:
        expected = MXTensor.to_mx(
            value.detach().contiguous(),
            elem_dtype=config.dtype,
            block_size=config.block_size,
            scaling_mode=config.scaling_mode,
            kernel_preference=config.kernel_preference,
        ).dequantize(value.dtype)
        actual = mx_fake_quantize(value, config)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(actual.shape, value.shape)
        self.assertEqual(actual.dtype, value.dtype)
        return actual

    def test_forward_matches_mxtensor_for_2d_and_3d_inputs(self) -> None:
        for elem_dtype in (
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            config = MXFakeQuantizeConfig(dtype=elem_dtype)
            for shape in ((4, 64), (3, 4, 64)):
                with self.subTest(elem_dtype=elem_dtype, shape=shape):
                    value = torch.randn(shape, dtype=torch.bfloat16)
                    self._assert_matches_reference(value, config)

    def test_identity_ste_preserves_parameter_and_gradient(self) -> None:
        for dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
            for shape in ((3, 64), (2, 3, 64)):
                with self.subTest(dtype=dtype, shape=shape):
                    parameter = torch.nn.Parameter(torch.randn(shape))
                    parameter_id = id(parameter)
                    gradient = torch.randn_like(parameter)
                    output = self._assert_matches_reference(
                        parameter,
                        MXFakeQuantizeConfig(dtype=dtype),
                    )
                    output.backward(gradient)

                    self.assertEqual(id(parameter), parameter_id)
                    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(torch.count_nonzero(parameter.grad), 0)

    def test_grouped_mm_supports_empty_and_ragged_experts(self) -> None:
        activation = torch.randn(5, 64, requires_grad=True)
        weight = torch.randn(3, 32, 64, requires_grad=True)
        offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
        output = mx_fake_quantized_grouped_mm(
            activation,
            weight,
            offsets,
            MXFakeQuantizeConfig(dtype=torch.float8_e4m3fn),
            MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2),
        )
        self.assertEqual(output.shape, (5, 32))
        output.sum().backward()
        self.assertTrue(torch.isfinite(activation.grad).all())
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertEqual(torch.count_nonzero(weight.grad[0]), 0)
        self.assertGreater(torch.count_nonzero(weight.grad[1]), 0)
        self.assertGreater(torch.count_nonzero(weight.grad[2]), 0)

    def test_grouped_mm_supports_all_empty_input(self) -> None:
        activation = torch.randn(0, 64, requires_grad=True)
        weight = torch.randn(2, 32, 64, requires_grad=True)
        output = mx_fake_quantized_grouped_mm(
            activation,
            weight,
            torch.tensor([0, 0], dtype=torch.int32),
            MXFakeQuantizeConfig(dtype=torch.float8_e4m3fn),
            MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2),
        )
        self.assertEqual(output.shape, (0, 32))
        output.sum().backward()
        self.assertEqual(torch.count_nonzero(weight.grad), 0)

    def test_grouped_mm_matches_independent_forward_and_gradient_reference(self):
        activation = torch.randn(5, 64, requires_grad=True)
        weight = torch.randn(3, 32, 64, requires_grad=True)
        offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
        ac = MXFakeQuantizeConfig(dtype=torch.float8_e4m3fn)
        wc = MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2)
        actual = mx_fake_quantized_grouped_mm(activation, weight, offsets, ac, wc)
        aq = MXTensor.to_mx(
            activation.detach(),
            elem_dtype=ac.dtype,
            block_size=32,
            scaling_mode=ac.scaling_mode,
        ).dequantize(torch.float32)
        wq = MXTensor.to_mx(
            weight.detach(),
            elem_dtype=wc.dtype,
            block_size=32,
            scaling_mode=wc.scaling_mode,
        ).dequantize(torch.float32)
        gradient = torch.randn_like(actual)
        expected = torch.cat(
            [aq[:0] @ wq[0].t(), aq[:2] @ wq[1].t(), aq[2:] @ wq[2].t()]
        )
        grad_input = torch.cat([gradient[:2] @ wq[1], gradient[2:] @ wq[2]])
        grad_weight = torch.stack(
            [
                torch.zeros_like(wq[0]),
                gradient[:2].t() @ aq[:2],
                gradient[2:].t() @ aq[2:],
            ]
        )
        actual.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(activation.grad, grad_input, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, grad_weight, rtol=0, atol=0)

    def _compare_grouped_mm(self, device, preference, *, compile_fn=False):
        from dataclasses import replace

        for counts in ((3, 0, 7), (0, 0, 0)):
            with self.subTest(counts=counts, preference=preference):
                torch.manual_seed(0)
                activation = torch.randn(
                    sum(counts),
                    64,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                weight = torch.randn(
                    3, 32, 64, device=device, dtype=torch.bfloat16, requires_grad=True
                )
                offsets = torch.tensor(counts, device=device, dtype=torch.int32).cumsum(
                    0, dtype=torch.int32
                )
                ac = MXFakeQuantizeConfig(
                    dtype=torch.float8_e4m3fn, kernel_preference=preference
                )
                wc = MXFakeQuantizeConfig(
                    dtype=torch.float4_e2m1fn_x2, kernel_preference=preference
                )
                ref_input = activation.detach().cpu().requires_grad_()
                ref_weight = weight.detach().cpu().requires_grad_()
                reference = mx_fake_quantized_grouped_mm(
                    ref_input,
                    ref_weight,
                    offsets.cpu(),
                    replace(ac, kernel_preference=KernelPreference.EMULATED),
                    replace(wc, kernel_preference=KernelPreference.EMULATED),
                )
                fn = mx_fake_quantized_grouped_mm
                if compile_fn:
                    fn = torch.compile(fn, fullgraph=True)
                actual = fn(activation, weight, offsets, ac, wc)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                reference.backward(gradient.cpu())
                torch.testing.assert_close(
                    actual.cpu(), reference, rtol=0.02, atol=0.125
                )
                torch.testing.assert_close(
                    activation.grad.cpu(), ref_input.grad, rtol=0.02, atol=0.125
                )
                torch.testing.assert_close(
                    weight.grad.cpu(), ref_weight.grad, rtol=0.02, atol=0.125
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_real_grouped_mm_forward_and_backward(self):
        self._compare_grouped_mm("cuda", KernelPreference.EMULATED)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_compiled_grouped_mm_forward_and_backward(self):
        self._compare_grouped_mm("cuda", KernelPreference.EMULATED, compile_fn=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_native_mxfp8_grouped_mm_forward_and_backward(self):
        from torchao.prototype.moe_training.mxfp8_grouped_mm import (
            _SM100_KERNELS_AVAILABLE,
        )

        if not _SM100_KERNELS_AVAILABLE or torch.cuda.get_device_capability()[0] < 10:
            self.skipTest(
                "native MXFP8 grouped kernels require SM100 and kernel dependencies"
            )
        for compile_fn in (False, True):
            self._compare_grouped_mm(
                "cuda", KernelPreference.AUTO, compile_fn=compile_fn
            )

    def test_kernel_preference_validation(self):
        from dataclasses import replace

        ac = MXFakeQuantizeConfig(dtype=torch.float8_e4m3fn)
        wc = MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2)
        activation, weight = torch.randn(2, 64), torch.randn(1, 32, 64)
        offsets = torch.tensor([2], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "must match"):
            mx_fake_quantized_grouped_mm(
                activation,
                weight,
                offsets,
                ac,
                replace(wc, kernel_preference=KernelPreference.AUTO),
            )
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            mx_fake_quantized_grouped_mm(
                activation,
                weight,
                offsets,
                replace(ac, kernel_preference=KernelPreference.AUTO),
                replace(wc, kernel_preference=KernelPreference.AUTO),
            )

    def test_invalid_block_size_and_last_dimension_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            MXFakeQuantizeConfig(block_size=0)
        with self.assertRaisesRegex(ValueError, "last dimension"):
            mx_fake_quantize(
                torch.randn(2, 33),
                MXFakeQuantizeConfig(block_size=32),
            )

    def test_special_values_match_mxtensor(self) -> None:
        values = torch.tensor(
            [
                0.0,
                -0.0,
                torch.finfo(torch.float32).tiny,
                -torch.finfo(torch.float32).tiny,
                1.0e20,
                -1.0e20,
                float("nan"),
                float("inf"),
            ]
            * 4,
            dtype=torch.float32,
        ).reshape(1, 32)
        self._assert_matches_reference(
            values,
            MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_mxfp4_forward_and_ste_on_cuda(self) -> None:
        value = torch.randn(
            4,
            8,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        gradient = torch.randn_like(value)
        output = self._assert_matches_reference(
            value,
            MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2),
        )
        output.backward(gradient)

        torch.testing.assert_close(value.grad, gradient, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(value.grad).all())


if __name__ == "__main__":
    unittest.main()
