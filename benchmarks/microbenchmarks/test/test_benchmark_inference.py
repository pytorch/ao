# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest

from benchmarks.microbenchmarks.benchmark_inference import run
from benchmarks.microbenchmarks.utils import BenchmarkConfig, BenchmarkResult


class TestBenchmarkInference(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        self.config = BenchmarkConfig(
            quantization="baseline",
            params={
                "high_precision_dtype": "torch.float32",
                "use_torch_compile": False,
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

    def test_run_inference(self):
        result = run(self.config)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertTrue(hasattr(result, "model_inference_time_in_ms"))


if __name__ == "__main__":
    unittest.main()
