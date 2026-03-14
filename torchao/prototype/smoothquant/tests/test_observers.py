# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchao.prototype.smoothquant.core import (
    RunningAbsMaxSmoothQuantObserver,
    SmoothQuantObserver,
)


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

        regular_sf, _ = regular_obs.calculate_qparams()
        running_sf, _ = running_obs.calculate_qparams()

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

        regular_sf, _ = regular_obs.calculate_qparams()
        running_sf, _ = running_obs.calculate_qparams()

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

        regular_sf, _ = regular_obs.calculate_qparams()
        running_sf, _ = running_obs.calculate_qparams()

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

        regular_sf, _ = regular_obs.calculate_qparams()
        running_sf, _ = running_obs.calculate_qparams()

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

            regular_sf, _ = regular_obs.calculate_qparams()
            running_sf, _ = running_obs.calculate_qparams()

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
        self.assertIsNotNone(running_obs._example_input)
        self.assertEqual(running_obs._example_input.shape, (32, in_features))

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

        regular_sf, _ = regular_obs.calculate_qparams()
        running_sf, _ = running_obs.calculate_qparams()

        torch.testing.assert_close(
            regular_sf,
            running_sf,
            rtol=1e-5,
            atol=1e-5,
            msg="Smoothing factors should be identical for large-scale test",
        )


if __name__ == "__main__":
    unittest.main()
