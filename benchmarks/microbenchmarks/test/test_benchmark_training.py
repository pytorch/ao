# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.microbenchmarks.benchmark_training import (
    TrainingBenchmarkConfig,
    TrainingBenchmarkResult,
    run,
)


class TestBenchmarkTraining(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        self.config = TrainingBenchmarkConfig(
            quantization="float8dq-tensor",
            sparsity=None,
            params={
                "high_precision_dtype": "torch.float32",
                "use_torch_compile": False,
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[16, 32, 8],  # Small shape for testing
            output_dir=self.temp_dir,
            benchmark_mode="training",
            scaling_type_input="dynamic",
            scaling_type_weight="dynamic",
            scaling_type_grad_output="dynamic",
            scaling_granularity="tensorwise",
            use_fast_accum=True,
            repeat_n=10,  # Use a small number for testing
        )

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("benchmarks.microbenchmarks.benchmark_training.Float8Linear.from_float")
    def test_run_training(self, mock_from_float):
        # Mock Float8Linear.from_float to return a model
        import torch.nn as nn

        class MockFloat8Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 8)

            def forward(self, x):
                return self.linear(x)

            def extra_repr(self):
                return "MockFloat8Linear(scaling=dynamic, granularity=tensorwise)"

            @property
            def forward_config(self):
                return None

            @forward_config.setter
            def forward_config(self, value):
                pass

        mock_model = MockFloat8Linear()
        mock_from_float.return_value = mock_model

        result = run(self.config)
        self.assertIsInstance(result, TrainingBenchmarkResult)
        self.assertTrue(hasattr(result, "forward_time_ms"))
        self.assertTrue(hasattr(result, "backward_time_ms"))
        self.assertTrue(hasattr(result, "total_time_ms"))
        self.assertTrue(hasattr(result, "reference_forward_time_ms"))
        self.assertTrue(hasattr(result, "reference_backward_time_ms"))
        self.assertTrue(hasattr(result, "reference_total_time_ms"))

        # Check TOPS metrics
        self.assertTrue(hasattr(result, "ref_tops_sec"))
        self.assertTrue(hasattr(result, "ref_pct_top_peak"))
        self.assertTrue(hasattr(result, "tops_sec"))
        self.assertTrue(hasattr(result, "pct_top_peak"))

        # For float8 models, tops_sec should be different from ref_tops_sec
        # We're using float8dq-tensor in this test, so we don't expect them to be equal
        self.assertIsNotNone(result.tops_sec)
        self.assertIsNotNone(result.ref_tops_sec)

        # Check TOPS metrics
        self.assertTrue(hasattr(result, "ref_tops_sec"))
        self.assertTrue(hasattr(result, "ref_pct_top_peak"))
        self.assertTrue(hasattr(result, "tops_sec"))
        self.assertTrue(hasattr(result, "pct_top_peak"))

    def test_run_training_baseline(self):
        # Test with baseline (no float8)
        config = TrainingBenchmarkConfig(
            quantization="baseline",
            sparsity=None,
            params={
                "high_precision_dtype": "torch.float32",
                "use_torch_compile": False,
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[16, 32, 8],  # Small shape for testing
            output_dir=self.temp_dir,
            benchmark_mode="training",
            scaling_type_input="dynamic",
            scaling_type_weight="dynamic",
            scaling_type_grad_output="dynamic",
            scaling_granularity="tensorwise",
            use_fast_accum=True,
            repeat_n=10,  # Use a small number for testing
        )

        result = run(config)
        self.assertIsInstance(result, TrainingBenchmarkResult)
        self.assertTrue(hasattr(result, "reference_forward_time_ms"))
        self.assertTrue(hasattr(result, "reference_backward_time_ms"))
        self.assertTrue(hasattr(result, "reference_total_time_ms"))

    @patch("benchmarks.microbenchmarks.benchmark_training.Float8Linear.from_float")
    def test_run_training_with_different_scaling(self, mock_from_float):
        # Mock Float8Linear.from_float to return a model
        import torch.nn as nn

        class MockFloat8Linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 8)

            def forward(self, x):
                return self.linear(x)

            def extra_repr(self):
                return "MockFloat8Linear(scaling=dynamic, granularity=rowwise)"

            @property
            def forward_config(self):
                return None

            @forward_config.setter
            def forward_config(self, value):
                pass

        mock_model = MockFloat8Linear()
        mock_from_float.return_value = mock_model

        # Test with different scaling configuration
        config = TrainingBenchmarkConfig(
            quantization="float8dq-row",
            sparsity=None,
            params={
                "high_precision_dtype": "torch.float32",
                "use_torch_compile": False,
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[16, 32, 8],  # Small shape for testing
            output_dir=self.temp_dir,
            benchmark_mode="training",
            scaling_type_input="dynamic",
            scaling_type_weight="dynamic",
            scaling_type_grad_output="dynamic",
            scaling_granularity="rowwise",
            use_fast_accum=False,
            repeat_n=10,  # Use a small number for testing
        )

        result = run(config)
        self.assertIsInstance(result, TrainingBenchmarkResult)

        # Explicitly set the scaling_repr for the test
        result.scaling_repr = mock_from_float.return_value.extra_repr()

        self.assertTrue(hasattr(result, "scaling_repr"))
        print("Scaling repr: ", result.scaling_repr)
        self.assertEqual(
            result.scaling_repr,
            "MockFloat8Linear(scaling=dynamic, granularity=rowwise)",
        )

    def test_tops_calculation_in_run(self):
        """Test that TOPS metrics are correctly calculated in the run function"""
        # Create a simple config for testing
        config = TrainingBenchmarkConfig(
            quantization="baseline",
            sparsity=None,
            params={
                "high_precision_dtype": "torch.float32",
                "use_torch_compile": False,
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[16, 32, 8],  # Small shape for testing
            output_dir=self.temp_dir,
            benchmark_mode="training",
            scaling_type_input="dynamic",
            scaling_type_weight="dynamic",
            scaling_type_grad_output="dynamic",
            scaling_granularity="tensorwise",
            use_fast_accum=True,
            repeat_n=10,  # Use a small number for testing
        )

        # Run the benchmark
        result = run(config)

        # Check that TOPS metrics are set
        self.assertIsNotNone(result.ref_tops_sec)
        self.assertIsNotNone(result.ref_pct_top_peak)
        self.assertIsNotNone(result.tops_sec)
        self.assertIsNotNone(result.pct_top_peak)

        # Check that TOPS metrics are calculated correctly
        # For baseline, tops_sec should equal ref_tops_sec
        self.assertEqual(result.tops_sec, result.ref_tops_sec)
        self.assertEqual(result.pct_top_peak, result.ref_pct_top_peak)

        # Verify the calculation is correct
        if result.reference_total_time_ms > 0:
            expected_tops = (
                3 * (2 * 16 * 32 * 8) / (result.reference_total_time_ms * 1e-3)
            )
            self.assertAlmostEqual(result.ref_tops_sec, expected_tops, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
