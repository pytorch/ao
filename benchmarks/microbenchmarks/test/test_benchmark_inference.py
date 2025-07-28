# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.microbenchmarks.benchmark_inference import run
from benchmarks.microbenchmarks.utils import BenchmarkConfig, BenchmarkResult


class TestBenchmarkInference(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        self.config = BenchmarkConfig(
            quantization="baseline",
            sparsity="semi-sparse",
            params={
                "high_precision_dtype": "torch.float32",
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[16, 32, 8],  # Small shape for testing
            output_dir=self.temp_dir,
            benchmark_mode="inference",
        )

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("benchmarks.microbenchmarks.benchmark_inference.string_to_config")
    def test_run_inference(self, mock_string_to_config):
        # Mock string_to_config to return a valid config
        from torchao.sparsity.sparse_api import SemiSparseWeightConfig

        mock_string_to_config.return_value = SemiSparseWeightConfig()

        result = run(self.config)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertTrue(hasattr(result, "compile_model_inference_time_in_ms"))

    @patch("benchmarks.microbenchmarks.benchmark_inference.string_to_config")
    def test_run_inference_with_semi_sparse_marlin(self, mock_string_to_config):
        """Test running inference with sparsity configurations"""
        # Mock string_to_config to return valid configs
        from torchao.dtypes import MarlinSparseLayout
        from torchao.quantization import Int4WeightOnlyConfig

        # Test with semi-sparse config
        mock_string_to_config.return_value = Int4WeightOnlyConfig(
            layout=MarlinSparseLayout()
        )
        config = BenchmarkConfig(
            quantization="marlin",
            sparsity="semi-sparse",
            params={
                "high_precision_dtype": "torch.float32",
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[64, 64, 64],  # Use dimensions divisible by 64
            output_dir=self.temp_dir,
            benchmark_mode="inference",
        )
        result = run(config)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertTrue(hasattr(result, "compile_model_inference_time_in_ms"))

    @patch("benchmarks.microbenchmarks.benchmark_inference.string_to_config")
    def test_run_inference_with_block_sparsity(self, mock_string_to_config):
        """Test running inference with sparsity configurations"""
        # Mock string_to_config to return valid configs
        from torchao.sparsity.sparse_api import (
            BlockSparseWeightConfig,
        )

        # Test with block sparsity
        mock_string_to_config.return_value = BlockSparseWeightConfig()
        config = BenchmarkConfig(
            quantization="baseline",
            sparsity="block",
            params={
                "high_precision_dtype": "torch.float32",
                "device": "cpu",
                "model_type": "linear",
            },
            shape_name="custom",
            shape=[64, 64, 64],  # Use dimensions divisible by 64
            output_dir=self.temp_dir,
            benchmark_mode="inference",
        )
        result = run(config)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertTrue(hasattr(result, "compile_model_inference_time_in_ms"))


if __name__ == "__main__":
    unittest.main()
