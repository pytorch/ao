# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import unittest

import torch
from parameterized import parameterized

# Need to import to load the ops
import torchao.experimental.ops.mps  # noqa: F401
from torchao.experimental.quant_api import (
    UIntxChooseQParamsAlgorithm,
    UIntxWeightOnlyConfig,
    UIntxWeightOnlyQuantizedLinear,
    _quantize,
)
from torchao.quantization.quant_api import quantize_


class TestUIntxWeightOnlyLinearQuantizer(unittest.TestCase):
    BITWIDTHS = range(1, 8)
    GROUPSIZES = [32, 64, 128, 256]

    # Currently, the quantization code in quant_api.py only supports K values
    # multiple of group_size.
    # TODO(mcandales): Generalize the code in quant_api.py and add tests to
    # cover values of K not multiple of group_size.
    def _model_setup(self):
        group_size = 32
        k0 = 96
        k1 = 224
        k2 = 160
        n = 44
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, k2, bias=False),
            torch.nn.Linear(k2, n, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        return model, group_size, k0, n

    def _quantize_model(self, model, precision, nbit, group_size):
        config = UIntxWeightOnlyConfig(
            bitwidth=nbit,
            group_size=group_size,
        )
        quantized_model = copy.deepcopy(model)
        quantized_model = quantized_model.to(device="mps", dtype=precision)
        quantize_(quantized_model, config)
        return quantized_model

    @parameterized.expand(BITWIDTHS)
    def test_export(self, nbit):
        model, group_size, k0, n = self._model_setup()
        m = 3
        activations = torch.randn(m, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        exported = torch.export.export(quantized_model, (activations,), strict=True)

        for node in exported.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(
                    str(node.target)
                    == f"torchao._linear_fp_act_{nbit}bit_weight.default"
                )

    @parameterized.expand(BITWIDTHS)
    def test_export_accuracy(self, nbit):
        group_size = 32
        m = 3
        n = 12
        k = 64
        with torch.no_grad():
            activations = torch.rand(m, k, dtype=torch.float32, device="mps")
            model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

            # Compute expected result
            weight_cpu = model[0].weight.data
            weight_qvals_cpu, weight_scales_cpu, weight_zeros_cpu = _quantize(
                weight_cpu, group_size, nbit, True, torch.uint8
            )
            weight_zeros_cpu = -weight_zeros_cpu * weight_scales_cpu
            expected = self._reference_linear_lowbit_quant_weights(
                activations.cpu(),
                weight_qvals_cpu,
                group_size,
                weight_scales_cpu,
                weight_zeros_cpu,
            )

            quantized_model = self._quantize_model(
                model, torch.float32, nbit, group_size
            )

            ep = torch.export.export(quantized_model, (activations,), strict=True)
            path = torch._inductor.aoti_compile_and_package(ep)
            compiled_model = torch._inductor.aoti_load_package(path)
            result = compiled_model(activations)

            # Compare results
            torch.testing.assert_close(result.cpu(), expected, rtol=0.001, atol=0.001)

    @parameterized.expand(BITWIDTHS)
    def test_2d_output_device_and_shape(self, nbit):
        model, group_size, k0, n = self._model_setup()
        m = 3
        activations = torch.randn(m, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        result = quantized_model(activations)
        self.assertTrue(result.is_mps)
        self.assertTrue(result.shape == (m, n))

    @parameterized.expand(BITWIDTHS)
    def test_3d_output_device_and_shape(self, nbit):
        model, group_size, k0, n = self._model_setup()
        leading_shape = (3, 5)
        activations = torch.randn(*leading_shape, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        result = quantized_model(activations)
        self.assertTrue(result.is_mps)
        self.assertTrue(result.shape == (*leading_shape, n))

    @parameterized.expand(itertools.product(BITWIDTHS, GROUPSIZES))
    def test_valid_groupsizes(self, nbit, group_size):
        k0 = 3 * group_size
        k1 = 7 * group_size
        n = 44
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, n, bias=False),
        ]
        model = torch.nn.Sequential(*layers)
        m = 5
        activations = torch.randn(m, k0, dtype=torch.float32, device="mps")

        quantized_model = self._quantize_model(model, torch.float32, nbit, group_size)
        result = quantized_model(activations)
        self.assertTrue(result.is_mps)
        self.assertTrue(result.shape == (m, n))

    @parameterized.expand(BITWIDTHS)
    def test_invalid_groupsizes(self, nbit):
        group_size = 16
        k0 = 3 * group_size
        k1 = 7 * group_size
        n = 44
        layers = [
            torch.nn.Linear(k0, k1, bias=False),
            torch.nn.Linear(k1, n, bias=False),
        ]
        model = torch.nn.Sequential(*layers)

        with self.assertRaises(ValueError):
            self._quantize_model(model, torch.float32, nbit, group_size)

    # TODO(mcandales): Consolidate with the reference impl in test_lowbit.py
    def _reference_linear_lowbit_quant_weights(self, A, W, group_size, S, Z):
        N = W.shape[0]
        K = W.shape[1]
        W = W.to(torch.float32)
        scales = S.unsqueeze(2).repeat(1, 1, group_size).view(N, -1)[:, :K]
        zeros = Z.unsqueeze(2).repeat(1, 1, group_size).view(N, -1)[:, :K]
        W = scales * W + zeros
        return torch.mm(A, W.t())

    @parameterized.expand(BITWIDTHS)
    def test_accuracy(self, nbit):
        print(f"nbit: {nbit}")
        group_size = 32
        m = 3
        n = 12
        k = 64
        with torch.no_grad():
            activations = torch.rand(m, k, dtype=torch.float32, device="mps")
            model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])
            quantized_model = self._quantize_model(
                model, torch.float32, nbit, group_size
            )
            result = quantized_model(activations)

            # Compute expected result
            weight_cpu = model[0].weight.data
            weight_qvals_cpu, weight_scales_cpu, weight_zeros_cpu = _quantize(
                weight_cpu, group_size, nbit, True, torch.uint8
            )
            weight_zeros_cpu = -weight_zeros_cpu * weight_scales_cpu
            expected = self._reference_linear_lowbit_quant_weights(
                activations.cpu(),
                weight_qvals_cpu,
                group_size,
                weight_scales_cpu,
                weight_zeros_cpu,
            )

            # Compare results
            torch.testing.assert_close(result.cpu(), expected, rtol=0.01, atol=0.01)

    @parameterized.expand([4])  # HQQ is optimized for 4-bit
    def test_hqq_accuracy(self, nbit):
        """Test that HQQ quantization produces results consistent with the kernel contract.

        The kernel expects: W_dequant = W_q * scale + zeros
        HQQ with raw_output=True gives: W_dequant = (W_q - zero) * scale
        We convert: zeros = -zero * scale to match the kernel format.

        This test verifies that:
        1. HQQ quantization runs without error
        2. The quantized model produces output reasonably close to the original
        3. The output is not corrupted (which would indicate format mismatch)
        """
        group_size = 32
        m = 3
        n = 12
        k = 64
        with torch.no_grad():
            torch.manual_seed(42)
            activations = torch.rand(m, k, dtype=torch.float32)
            model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

            # Get original unquantized output for reference
            original_output = model(activations)

            # Quantize with HQQ
            config = UIntxWeightOnlyConfig(
                bitwidth=nbit,
                group_size=group_size,
                uintx_choose_qparams_algorithm=UIntxChooseQParamsAlgorithm.HQQ,
            )
            quantized_model = copy.deepcopy(model)
            quantized_model = quantized_model.to(device="mps", dtype=torch.float32)
            quantize_(quantized_model, config)

            result = quantized_model(activations.to("mps"))

            # Verify the quantized layer has the expected attributes
            qlinear = quantized_model[0]
            self.assertIsInstance(qlinear, UIntxWeightOnlyQuantizedLinear)
            self.assertEqual(qlinear.nbit, nbit)
            self.assertEqual(qlinear.group_size, group_size)

            # Verify output is valid (not NaN or Inf)
            self.assertFalse(torch.isnan(result).any(), "HQQ output contains NaN")
            self.assertFalse(torch.isinf(result).any(), "HQQ output contains Inf")

            # Verify output is reasonably close to original
            # 4-bit quantization typically has ~1-5% relative error
            torch.testing.assert_close(
                result.cpu(), original_output, rtol=0.1, atol=0.1
            )

    def test_hqq_vs_default_quantization(self):
        """Test that HQQ produces different (typically better) quantization than default."""
        nbit = 4
        group_size = 32
        m = 3
        n = 12
        k = 64

        with torch.no_grad():
            # Use a fixed seed for reproducibility
            torch.manual_seed(42)
            activations = torch.rand(m, k, dtype=torch.float32)
            model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

            # Quantize with default (no HQQ)
            config_default = UIntxWeightOnlyConfig(
                bitwidth=nbit,
                group_size=group_size,
                uintx_choose_qparams_algorithm=UIntxChooseQParamsAlgorithm.MIN_MAX,
            )
            model_default = copy.deepcopy(model).to(device="mps", dtype=torch.float32)
            quantize_(model_default, config_default)
            result_default = model_default(activations.to("mps"))

            # Quantize with HQQ
            config_hqq = UIntxWeightOnlyConfig(
                bitwidth=nbit,
                group_size=group_size,
                uintx_choose_qparams_algorithm=UIntxChooseQParamsAlgorithm.HQQ,
            )
            model_hqq = copy.deepcopy(model).to(device="mps", dtype=torch.float32)
            quantize_(model_hqq, config_hqq)
            result_hqq = model_hqq(activations.to("mps"))

            # Both should produce valid results (not NaN or Inf)
            self.assertFalse(torch.isnan(result_default).any())
            self.assertFalse(torch.isnan(result_hqq).any())
            self.assertFalse(torch.isinf(result_default).any())
            self.assertFalse(torch.isinf(result_hqq).any())

            # HQQ and default should produce different results
            # (unless the weights happen to be perfectly distributed)
            # We don't assert which is better, just that they're different
            self.assertFalse(
                torch.allclose(
                    result_default.cpu(), result_hqq.cpu(), rtol=1e-5, atol=1e-5
                ),
                "HQQ and default quantization should produce different results",
            )

    def test_hqq_better_accuracy_than_default(self):
        """Test that HQQ produces better quantization accuracy than default min-max.

        HQQ uses an iterative optimization to find better scale/zero parameters,
        which should result in lower quantization error compared to simple min-max.
        """
        nbit = 4
        group_size = 32
        m = 10
        n = 64
        k = 128

        # Run multiple trials to ensure HQQ is consistently better
        hqq_wins = 0
        num_trials = 5

        for trial in range(num_trials):
            with torch.no_grad():
                torch.manual_seed(trial * 100)
                activations = torch.randn(m, k, dtype=torch.float32)
                model = torch.nn.Sequential(*[torch.nn.Linear(k, n, bias=False)])

                # Get original unquantized output
                original_output = model(activations)

                # Quantize with default (no HQQ)
                config_default = UIntxWeightOnlyConfig(
                    bitwidth=nbit,
                    group_size=group_size,
                    uintx_choose_qparams_algorithm=UIntxChooseQParamsAlgorithm.MIN_MAX,
                )
                model_default = copy.deepcopy(model).to(
                    device="mps", dtype=torch.float32
                )
                quantize_(model_default, config_default)
                result_default = model_default(activations.to("mps"))

                # Quantize with HQQ
                config_hqq = UIntxWeightOnlyConfig(
                    bitwidth=nbit,
                    group_size=group_size,
                    uintx_choose_qparams_algorithm=UIntxChooseQParamsAlgorithm.HQQ,
                )
                model_hqq = copy.deepcopy(model).to(device="mps", dtype=torch.float32)
                quantize_(model_hqq, config_hqq)
                result_hqq = model_hqq(activations.to("mps"))

                # Compute mean squared error for each method
                mse_default = torch.mean(
                    (result_default.cpu() - original_output) ** 2
                ).item()
                mse_hqq = torch.mean((result_hqq.cpu() - original_output) ** 2).item()

                if mse_hqq < mse_default:
                    hqq_wins += 1

        # HQQ should win in majority of trials (at least 3 out of 5)
        self.assertGreaterEqual(
            hqq_wins,
            3,
            f"HQQ should have lower error than default in most trials, "
            f"but only won {hqq_wins}/{num_trials} times",
        )

    def test_config_accepts_string_algorithm(self):
        """Test that UIntxWeightOnlyConfig accepts string values for algorithm."""
        # Test "hqq" string
        config_hqq = UIntxWeightOnlyConfig(uintx_choose_qparams_algorithm="hqq")
        self.assertEqual(
            config_hqq.uintx_choose_qparams_algorithm,
            UIntxChooseQParamsAlgorithm.HQQ,
        )

        # Test "min_max" string
        config_min_max = UIntxWeightOnlyConfig(uintx_choose_qparams_algorithm="min_max")
        self.assertEqual(
            config_min_max.uintx_choose_qparams_algorithm,
            UIntxChooseQParamsAlgorithm.MIN_MAX,
        )

        # Test enum values directly (should also work)
        config_enum = UIntxWeightOnlyConfig(
            uintx_choose_qparams_algorithm=UIntxChooseQParamsAlgorithm.HQQ
        )
        self.assertEqual(
            config_enum.uintx_choose_qparams_algorithm,
            UIntxChooseQParamsAlgorithm.HQQ,
        )

    def test_config_rejects_invalid_algorithm(self):
        """Test that UIntxWeightOnlyConfig raises ValueError for invalid algorithm."""
        with self.assertRaises(ValueError):
            UIntxWeightOnlyConfig(uintx_choose_qparams_algorithm="invalid_algorithm")

    def test_config_default_algorithm_is_min_max(self):
        """Test backward compatibility: default algorithm is MIN_MAX when not specified."""
        # Default config without specifying algorithm
        config = UIntxWeightOnlyConfig()
        self.assertEqual(
            config.uintx_choose_qparams_algorithm,
            UIntxChooseQParamsAlgorithm.MIN_MAX,
        )

        # Config with only other params specified
        config_with_bitwidth = UIntxWeightOnlyConfig(bitwidth=3, group_size=64)
        self.assertEqual(
            config_with_bitwidth.uintx_choose_qparams_algorithm,
            UIntxChooseQParamsAlgorithm.MIN_MAX,
        )


if __name__ == "__main__":
    unittest.main()
