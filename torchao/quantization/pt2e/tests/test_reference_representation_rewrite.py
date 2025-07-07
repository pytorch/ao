# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.quantization.pt2e.reference_representation_rewrite import (
    _qdq_dynamic_quantized_linear_4bit_groupwise,
    _reference_dynamic_quantized_linear_4bit_groupwise,
)


class TestReferenceRepresentationRewrite(unittest.TestCase):
    """Test cases for dynamically quantized linear 4-bit groupwise implementations."""

    def _get_default_quantization_params(self):
        """Get default quantization parameters."""
        return {
            "x_quant_min": -128,
            "x_quant_max": 127,
            "x_eps": torch.finfo(torch.float32).eps,
            "x_scales_type": torch.float32,
            "x_zero_points_type": torch.int8,
            "weight_quant_min": -8,  # 4-bit range
            "weight_quant_max": 7,  # 4-bit range
        }

    def _create_test_tensors(
        self, batch_size, in_features, out_features, group_size, seed=42, bias=True
    ):
        """Create test tensors for the given dimensions."""
        torch.manual_seed(seed)

        # Create input activation
        x_fp32 = torch.randn(batch_size, in_features, dtype=torch.float32)

        # Create 4-bit quantized weight (stored as int8 with values in [-8, 7])
        weight_i4 = torch.randint(-8, 7, (out_features, in_features), dtype=torch.int8)

        # Create groupwise scales and zero points
        num_groups = in_features // group_size
        weight_scale = (
            torch.randn(out_features, num_groups, dtype=torch.float32).abs() + 0.01
        )
        weight_zero_point = torch.zeros(
            out_features, num_groups, dtype=torch.int8
        )  # Symmetric quantization

        # Create bias if requested
        bias_fp32 = torch.randn(out_features, dtype=torch.float32) if bias else None

        return {
            "x_fp32": x_fp32,
            "weight_i4": weight_i4,
            "weight_scale": weight_scale,
            "weight_zero_point": weight_zero_point,
            "bias_fp32": bias_fp32,
        }

    def _run_qdq_implementation(self, tensors, quant_params, group_size):
        """Run the QDQ implementation with given tensors and parameters."""
        return _qdq_dynamic_quantized_linear_4bit_groupwise(
            x_fp32=tensors["x_fp32"],
            x_quant_min=quant_params["x_quant_min"],
            x_quant_max=quant_params["x_quant_max"],
            x_eps=quant_params["x_eps"],
            x_scales_type=quant_params["x_scales_type"],
            x_zero_points_type=quant_params["x_zero_points_type"],
            weight_i4=tensors["weight_i4"],
            weight_scale=tensors["weight_scale"],
            weight_zero_point=tensors["weight_zero_point"],
            weight_quant_min=quant_params["weight_quant_min"],
            weight_quant_max=quant_params["weight_quant_max"],
            bias_fp32=tensors["bias_fp32"],
            group_size=group_size,
        )

    def _run_reference_implementation(self, tensors, quant_params, group_size):
        """Run the reference implementation with given tensors and parameters."""
        return _reference_dynamic_quantized_linear_4bit_groupwise(
            x_fp32=tensors["x_fp32"],
            x_quant_min=quant_params["x_quant_min"],
            x_quant_max=quant_params["x_quant_max"],
            x_eps=quant_params["x_eps"],
            x_scales_type=quant_params["x_scales_type"],
            x_zero_points_type=quant_params["x_zero_points_type"],
            weight_i4=tensors["weight_i4"],
            weight_scale=tensors["weight_scale"],
            weight_zero_point=tensors["weight_zero_point"],
            weight_quant_min=quant_params["weight_quant_min"],
            weight_quant_max=quant_params["weight_quant_max"],
            bias_fp32=tensors["bias_fp32"],
            group_size=group_size,
        )

    def _assert_basic_properties(self, result, expected_shape):
        """Assert basic properties of the result tensor."""
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.dtype, torch.float32)

    def _assert_implementations_close(
        self, qdq_result, ref_result, atol=5e-2, rtol=5e-2, msg_suffix=""
    ):
        """Assert that QDQ and reference implementations produce similar results."""
        torch.testing.assert_close(
            qdq_result,
            ref_result,
            atol=atol,
            rtol=rtol,
            msg=f"QDQ and reference results differ significantly{msg_suffix}",
        )

    def test_qdq_dynamic_quantized_linear_4bit_groupwise_basic(self):
        """Test that QDQ implementation runs without errors and produces reasonable output."""
        # Test-specific parameters
        batch_size, in_features, out_features, group_size = 2, 32, 8, 8

        quant_params = self._get_default_quantization_params()
        tensors = self._create_test_tensors(
            batch_size, in_features, out_features, group_size
        )

        result = self._run_qdq_implementation(tensors, quant_params, group_size)
        self._assert_basic_properties(result, (batch_size, out_features))

    def test_reference_dynamic_quantized_linear_4bit_groupwise_basic(self):
        """Test that reference implementation runs without errors and produces reasonable output."""
        # Test-specific parameters
        batch_size, in_features, out_features, group_size = 2, 32, 8, 8

        quant_params = self._get_default_quantization_params()
        tensors = self._create_test_tensors(
            batch_size, in_features, out_features, group_size
        )

        result = self._run_reference_implementation(tensors, quant_params, group_size)
        self._assert_basic_properties(result, (batch_size, out_features))

    def test_both_implementations_no_bias(self):
        """Test both implementations without bias."""
        # Test-specific parameters
        batch_size, in_features, out_features, group_size = 1, 16, 4, 8
        seed = 123

        quant_params = self._get_default_quantization_params()
        tensors = self._create_test_tensors(
            batch_size, in_features, out_features, group_size, seed=seed, bias=False
        )

        qdq_result = self._run_qdq_implementation(tensors, quant_params, group_size)
        ref_result = self._run_reference_implementation(
            tensors, quant_params, group_size
        )

        self._assert_basic_properties(qdq_result, (batch_size, out_features))
        self._assert_basic_properties(ref_result, (batch_size, out_features))
        self._assert_implementations_close(
            qdq_result, ref_result, msg_suffix=" for no-bias case"
        )

    def test_edge_cases_group_size_validation(self):
        """Test edge cases and error conditions."""
        # Test-specific parameters
        batch_size, in_features, out_features = 1, 32, 8

        quant_params = self._get_default_quantization_params()
        tensors = self._create_test_tensors(
            batch_size, in_features, out_features, 8
        )  # Valid group size for tensor creation

        # Test with group_size that doesn't divide in_features evenly
        with self.assertRaises(AssertionError):
            self._run_qdq_implementation(
                tensors, quant_params, 7
            )  # 32 is not divisible by 7

        # Test with zero group_size
        with self.assertRaises(AssertionError):
            self._run_qdq_implementation(tensors, quant_params, 0)

    def test_weight_dimension_validation(self):
        """Test weight dimension validation."""
        # Test-specific parameters
        batch_size, in_features, out_features, group_size = 1, 32, 8, 8

        quant_params = self._get_default_quantization_params()
        tensors = self._create_test_tensors(
            batch_size, in_features, out_features, group_size
        )

        # Create 1D weight tensor (should fail)
        tensors["weight_i4"] = torch.randint(-8, 7, (in_features,), dtype=torch.int8)

        with self.assertRaises((AssertionError, IndexError)):
            self._run_qdq_implementation(tensors, quant_params, group_size)

    def test_different_group_sizes(self):
        """Test with different valid group sizes."""
        # Test-specific parameters
        batch_size, in_features, out_features = 2, 64, 16
        seed = 456
        group_sizes = [8, 16, 32, 64]

        quant_params = self._get_default_quantization_params()

        for group_size in group_sizes:
            with self.subTest(group_size=group_size):
                tensors = self._create_test_tensors(
                    batch_size, in_features, out_features, group_size, seed=seed
                )

                qdq_result = self._run_qdq_implementation(
                    tensors, quant_params, group_size
                )
                ref_result = self._run_reference_implementation(
                    tensors, quant_params, group_size
                )

                self._assert_basic_properties(qdq_result, (batch_size, out_features))
                self._assert_basic_properties(ref_result, (batch_size, out_features))
                self._assert_implementations_close(
                    qdq_result, ref_result, msg_suffix=f" for group_size={group_size}"
                )

    def test_qdq_vs_reference_implementation_comparison(self):
        """Test that QDQ and reference implementations produce similar results with various configurations."""
        # Test-specific parameters
        test_cases = [
            (1, 32, 8, 8),
            (2, 64, 16, 16),
            (4, 128, 32, 32),
        ]

        quant_params = self._get_default_quantization_params()

        for batch_size, in_features, out_features, group_size in test_cases:
            with self.subTest(
                batch_size=batch_size,
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
            ):
                seed = 42 + batch_size + in_features  # Deterministic but varied seed

                # Test with bias
                tensors_with_bias = self._create_test_tensors(
                    batch_size,
                    in_features,
                    out_features,
                    group_size,
                    seed=seed,
                    bias=True,
                )

                qdq_result = self._run_qdq_implementation(
                    tensors_with_bias, quant_params, group_size
                )
                ref_result = self._run_reference_implementation(
                    tensors_with_bias, quant_params, group_size
                )

                self.assertEqual(qdq_result.shape, ref_result.shape)
                self.assertEqual(qdq_result.shape, (batch_size, out_features))

                self._assert_implementations_close(
                    qdq_result,
                    ref_result,
                    msg_suffix=f" for shape ({batch_size}, {in_features}, {out_features}) with group_size={group_size}",
                )

                # Test without bias
                tensors_no_bias = self._create_test_tensors(
                    batch_size,
                    in_features,
                    out_features,
                    group_size,
                    seed=seed,
                    bias=False,
                )

                qdq_result_no_bias = self._run_qdq_implementation(
                    tensors_no_bias, quant_params, group_size
                )
                ref_result_no_bias = self._run_reference_implementation(
                    tensors_no_bias, quant_params, group_size
                )

                self._assert_implementations_close(
                    qdq_result_no_bias,
                    ref_result_no_bias,
                    msg_suffix=f" for no-bias case with shape ({batch_size}, {in_features}, {out_features}) and group_size={group_size}",
                )


if __name__ == "__main__":
    unittest.main()
