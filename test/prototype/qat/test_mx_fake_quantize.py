# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.qat import MXFakeQuantizeConfig, mx_fake_quantize


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
        parameter = torch.nn.Parameter(torch.randn(2, 3, 64))
        parameter_id = id(parameter)
        gradient = torch.randn_like(parameter)
        output = self._assert_matches_reference(
            parameter,
            MXFakeQuantizeConfig(dtype=torch.float4_e2m1fn_x2),
        )
        output.backward(gradient)

        self.assertEqual(id(parameter), parameter_id)
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(torch.count_nonzero(parameter.grad), 0)

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
