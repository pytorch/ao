# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from copy import deepcopy

import torch
from torch.testing._internal import common_utils

from torchao.prototype.smoothquant import (
    SmoothQuantConfig,
    SmoothQuantObservedLinear,
)
from torchao.prototype.smoothquant.core import (
    RunningAbsMaxSmoothQuantObserver,
    SmoothQuantObserver,
)
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerRow, PerTensor
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
)
from torchao.quantization.quantize_.common import SupportsActivationPreScaling
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.utils import (
    compute_error as SQNR,
)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 64, bias=False)

    def example_inputs(
        self,
        batch_size,
        sequence_length=10,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        # For SmoothQuant tests, we intentionally insert some outliers to input features
        x = torch.randn(
            batch_size,
            sequence_length,
            self.linear1.in_features,
            dtype=dtype,
            device=device,
        )
        n_outliers = max(1, int(x.size(-1) * 0.1))
        # Randomly select outlier features
        outlier_indices = torch.randperm(x.size(-1))[:n_outliers]
        x[:, :, outlier_indices] *= 10.0
        return (x,)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


device_list = ["cpu"]
if torch.cuda.is_available():
    device_list.append("cuda")

if torch.xpu.is_available():
    device_list.append("xpu")


@unittest.skipIf(torch.version.hip is not None, "Skipping tests in ROCm")
class TestSmoothQuant(unittest.TestCase):
    """SmoothQuant tests using only supported quantization configs."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level configuration for tests."""
        # This test case will trigger recompilation many times, so set a large cache_size_limit here
        torch._dynamo.config.cache_size_limit = 128

    @common_utils.parametrize("alpha", [0.5, 0.75])
    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            Int8StaticActivationInt8WeightConfig(granularity=PerRow()),
            Int8StaticActivationInt8WeightConfig(granularity=PerTensor()),
            # Note: float8_static_activation_float8_weight is broken after recent PyTorch update.
            # TODO(#1639): Fix for supporting more API in torchao/quantization/quant_api.py
        ],
    )
    @common_utils.parametrize("device", device_list)
    @common_utils.parametrize("input_dtype", [torch.bfloat16])
    def test_smoothquant_accuracy(self, alpha, base_config, device, input_dtype):
        """Test if SmoothQuant achieves lower loss than basic quantization."""
        # Create linear layer
        m = ToyLinearModel().eval().to(device).to(input_dtype)
        x = m.example_inputs(batch_size=16, dtype=input_dtype, device=device)

        # Reference output
        out_ref = m(*x)

        # Step 1. Basic quantization
        basic_model = deepcopy(m)
        if isinstance(base_config, Int8StaticActivationInt8WeightConfig):
            quantize_(
                basic_model,
                Int8DynamicActivationInt8WeightConfig(
                    version=2, granularity=base_config.granularity
                ),
            )
        else:
            quantize_(basic_model, base_config)
        out_basic = basic_model(*x)
        loss_base = torch.nn.functional.mse_loss(out_basic, out_ref).item()

        # Step 2. SmoothQuant
        model = deepcopy(m)
        config = SmoothQuantConfig(
            base_config=base_config,
            step=QuantizationStep.PREPARE,
            alpha=alpha,
        )
        quantize_(model, config)

        # Perform calibration with test data
        model(*x)

        config.step = QuantizationStep.CONVERT
        quantize_(model, config)
        assert isinstance(model.linear1.weight, SupportsActivationPreScaling)
        assert isinstance(model.linear2.weight, SupportsActivationPreScaling)
        assert model.linear1.weight.act_pre_scale is not None
        assert model.linear2.weight.act_pre_scale is not None

        out_smoothquant = model(*x)
        loss_smoothquant = torch.nn.functional.mse_loss(out_smoothquant, out_ref).item()

        assert loss_smoothquant < loss_base, (
            f"SmoothQuant loss ({loss_smoothquant:.6f}) should not be higher than basic loss ({loss_base:.6f})"
        )
        # Make sure the result is reasonable
        self.assertGreater(SQNR(out_ref, out_smoothquant), 20.0)

    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            # TODO: Check more quantization APIs
        ],
    )
    def test_observer_insertion(self, base_config):
        """Test that PREPARE step correctly inserts SmoothQuantObservedLinear."""

        m = ToyLinearModel().eval()

        # Before quantization - should be regular Linear
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # PREPARE step - should insert observers
        config = SmoothQuantConfig(
            base_config=base_config,
            step=QuantizationStep.PREPARE,
        )
        quantize_(m, config)

        # After PREPARE - should be SmoothQuantObservedLinear
        self.assertIsInstance(m.linear1, SmoothQuantObservedLinear)
        self.assertTrue(hasattr(m.linear1, "obs"))

        # Test calibration
        test_data = torch.randn(2, 512)
        m(test_data)

        # CONVERT step - should produce regular Linear with quantized weights
        config.step = QuantizationStep.CONVERT
        quantize_(m, config)

        # After CONVERT - should be regular Linear again (but quantized)
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(version=2),
            # TODO: Check more quantization APIs
        ],
    )
    def test_prepare_for_loading(self, base_config):
        """Test PREPARE_FOR_LOADING step for loading pre-quantized checkpoints."""

        m = ToyLinearModel().eval()

        # Before quantization - should be regular Linear
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # PREPARE_FOR_LOADING step - should create quantized model ready for loading
        config = SmoothQuantConfig(
            base_config=base_config,
            step=QuantizationStep.PREPARE_FOR_LOADING,
            alpha=0.5,
        )
        quantize_(m, config)

        # After PREPARE_FOR_LOADING - should be regular Linear with quantized weights
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # Test that model can run inference
        test_data = torch.randn(2, 512)
        with torch.inference_mode():
            output = m(test_data)

            # Validate output
            self.assertIsNotNone(
                output, "PREPARE_FOR_LOADING model output should not be None"
            )
            self.assertFalse(
                torch.isnan(output).any(), "Model should not produce NaN values"
            )
            self.assertEqual(
                output.shape, (2, 64), "Output shape should match expected dimensions"
            )


common_utils.instantiate_parametrized_tests(TestSmoothQuant)


class SmoothQuantObserverTest(unittest.TestCase):
    """Tests for SmoothQuantObserver and RunningAbsMaxSmoothQuantObserver."""

    def test_smoothing_factor_equivalence_single_batch(self):
        """Both observers should produce identical smoothing factors for a single input batch."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        input_batch = torch.randn(8, in_features)

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        regular_obs(input_batch)
        running_obs(input_batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, _, _ = running_obs.calculate_qparams()

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should be identical for single batch",
        )

    def test_smoothing_factor_equivalence_multiple_batches(self):
        """Both observers should produce identical smoothing factors across multiple batches."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        for batch in batches:
            regular_obs(batch)
            running_obs(batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, _, _ = running_obs.calculate_qparams()

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should be identical across multiple batches",
        )

    def test_smoothing_factor_equivalence_3d_input(self):
        """Both observers should handle 3D inputs (batch, seq, features) correctly."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(4, 16, in_features) for _ in range(3)]

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        for batch in batches:
            regular_obs(batch)
            running_obs(batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, _, _ = running_obs.calculate_qparams()

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should be identical for 3D inputs",
        )

    def test_smoothing_factor_with_alpha_none(self):
        """Both observers should return ones when alpha is None."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        input_batch = torch.randn(8, in_features)

        regular_obs = SmoothQuantObserver(weight=weight, alpha=None)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=None)

        regular_obs(input_batch)
        running_obs(input_batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, _, _ = running_obs.calculate_qparams()

        expected = torch.ones(in_features)
        torch.testing.assert_close(regular_sf, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(running_sf, expected, rtol=1e-5, atol=1e-5)

    def test_smoothing_factor_with_different_alphas(self):
        """Both observers should produce identical results for various alpha values."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(3)]

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            regular_obs = SmoothQuantObserver(weight=weight, alpha=alpha)
            running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=alpha)

            for batch in batches:
                regular_obs(batch)
                running_obs(batch)

            regular_sf, _, _ = regular_obs.calculate_qparams()
            running_sf, _, _ = running_obs.calculate_qparams()

            torch.testing.assert_close(
                regular_sf,
                running_sf,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Smoothing factors should be identical for alpha={alpha}",
            )

    def test_running_observer_memory_efficiency(self):
        """RunningAbsMaxSmoothQuantObserver should not store all inputs."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        for _ in range(100):
            batch = torch.randn(32, in_features)
            running_obs(batch)

        self.assertEqual(running_obs.calibration_count, 100)
        self.assertIsNotNone(running_obs.x_abs_max)
        self.assertEqual(running_obs.x_abs_max.shape, (in_features,))

    def test_regular_observer_stores_all_inputs(self):
        """SmoothQuantObserver should store all inputs for reference."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)

        num_batches = 10
        for _ in range(num_batches):
            batch = torch.randn(32, in_features)
            regular_obs(batch)

        self.assertEqual(len(regular_obs.inputs), num_batches)

    def test_observers_raise_without_calibration(self):
        """Both observers should raise assertion error if calculate_qparams called without calibration."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        with self.assertRaises(AssertionError):
            regular_obs.calculate_qparams()

        with self.assertRaises(AssertionError):
            running_obs.calculate_qparams()

    def test_observers_forward_returns_input_unchanged(self):
        """Forward pass should return the input tensor unchanged."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        input_batch = torch.randn(8, in_features)

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        regular_output = regular_obs(input_batch)
        running_output = running_obs(input_batch)

        torch.testing.assert_close(regular_output, input_batch)
        torch.testing.assert_close(running_output, input_batch)

    def test_smoothing_factor_equivalence_large_scale(self):
        """Test equivalence with larger feature dimensions and more batches."""
        torch.manual_seed(42)
        in_features = 512
        out_features = 256

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(16, 32, in_features) for _ in range(20)]

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        for batch in batches:
            regular_obs(batch)
            running_obs(batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, _, _ = running_obs.calculate_qparams()

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should be identical for large-scale test",
        )

    def test_two_pass_calibration(self):
        """Test the two-pass calibration workflow for RunningAbsMaxSmoothQuantObserver."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        # First pass: collect x_abs_max
        for batch in batches:
            running_obs(batch)

        self.assertEqual(running_obs.calibration_count, 5)
        self.assertFalse(running_obs._in_second_pass)

        # Compute smoothing factor
        smoothing_factor = running_obs.compute_smoothing_factor()

        self.assertTrue(running_obs._in_second_pass)
        self.assertIsNotNone(smoothing_factor)
        self.assertEqual(smoothing_factor.shape, (in_features,))

        # Second pass: collect smoothed activation stats
        for batch in batches:
            running_obs(batch)

        self.assertEqual(running_obs._second_pass_count, 5)
        self.assertIsNotNone(running_obs._smooth_input_min)
        self.assertIsNotNone(running_obs._smooth_input_max)

    def test_two_pass_activation_scale_symmetric(self):
        """Test activation scale computation with symmetric quantization in two-pass mode."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        # Regular observer: single pass
        for batch in batches:
            regular_obs(batch)

        # Running observer: two passes
        for batch in batches:
            running_obs(batch)

        running_obs.compute_smoothing_factor()

        for batch in batches:
            running_obs(batch)

        # Compare smoothing factors
        weight_quant_kwargs = {
            "quant_min": -128,
            "quant_max": 127,
            "qscheme": torch.per_tensor_symmetric,
        }

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, running_scale, running_zp = running_obs.calculate_qparams(
            weight_quant_kwargs
        )

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should match between single-pass and two-pass",
        )

        # Verify activation scale and zero_point are computed
        self.assertIsNotNone(running_scale)
        self.assertIsNotNone(running_zp)

    def test_two_pass_activation_scale_affine(self):
        """Test activation scale computation with affine quantization in two-pass mode."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        # First pass
        for batch in batches:
            running_obs(batch)

        running_obs.compute_smoothing_factor()

        # Second pass
        for batch in batches:
            running_obs(batch)

        weight_quant_kwargs = {
            "quant_min": 0,
            "quant_max": 255,
            "is_symmetric": False,
        }

        _, running_scale, running_zp = running_obs.calculate_qparams(
            weight_quant_kwargs
        )

        self.assertIsNotNone(running_scale)
        self.assertGreater(running_scale.item(), 0)
        self.assertIsNotNone(running_zp)

    def test_no_second_pass_returns_none_scale(self):
        """Without second pass, calculate_qparams should return None for activation scale."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        # Only first pass
        for batch in batches:
            running_obs(batch)

        weight_quant_kwargs = {
            "quant_min": -128,
            "quant_max": 127,
            "qscheme": torch.per_tensor_symmetric,
        }

        sf, scale, zp = running_obs.calculate_qparams(weight_quant_kwargs)

        self.assertIsNotNone(sf)
        self.assertIsNone(scale)
        self.assertIsNone(zp)

    def test_compute_smoothing_factor_resets_state(self):
        """compute_smoothing_factor should reset second pass state."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(3)]

        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        # First pass
        for batch in batches:
            running_obs(batch)

        running_obs.compute_smoothing_factor()

        # Some second pass data
        running_obs(batches[0])
        self.assertEqual(running_obs._second_pass_count, 1)

        # Call compute_smoothing_factor again (should reset)
        running_obs.compute_smoothing_factor()

        self.assertEqual(running_obs._second_pass_count, 0)
        self.assertIsNone(running_obs._smooth_input_min)
        self.assertIsNone(running_obs._smooth_input_max)

    def test_two_pass_qparam_equivalence_symmetric(self):
        """Two-pass running observer should produce smoothing factors matching the regular observer,
        and should compute valid symmetric qparams."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        weight_quant_kwargs = {
            "quant_min": -128,
            "quant_max": 127,
            "qscheme": torch.per_tensor_symmetric,
        }

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        for batch in batches:
            regular_obs(batch)
            running_obs(batch)

        running_obs.compute_smoothing_factor()

        for batch in batches:
            running_obs(batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, running_scale, running_zp = running_obs.calculate_qparams(
            weight_quant_kwargs
        )

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should match",
        )

        self.assertIsNotNone(running_scale)
        self.assertGreater(running_scale.item(), 0)
        self.assertIsNotNone(running_zp)
        self.assertEqual(running_zp.item(), 0)

    def test_two_pass_qparam_equivalence_affine(self):
        """Two-pass running observer should produce smoothing factors matching the regular observer,
        and should compute valid affine qparams."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 32

        weight = torch.randn(out_features, in_features)
        batches = [torch.randn(8, in_features) for _ in range(5)]

        weight_quant_kwargs = {
            "quant_min": 0,
            "quant_max": 255,
            "is_symmetric": False,
        }

        regular_obs = SmoothQuantObserver(weight=weight, alpha=0.5)
        running_obs = RunningAbsMaxSmoothQuantObserver(weight=weight, alpha=0.5)

        for batch in batches:
            regular_obs(batch)
            running_obs(batch)

        running_obs.compute_smoothing_factor()

        for batch in batches:
            running_obs(batch)

        regular_sf, _, _ = regular_obs.calculate_qparams()
        running_sf, running_scale, running_zp = running_obs.calculate_qparams(
            weight_quant_kwargs
        )

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should match",
        )

        self.assertIsNotNone(running_scale)
        self.assertGreater(running_scale.item(), 0)
        self.assertIsNotNone(running_zp)


if __name__ == "__main__":
    unittest.main()
