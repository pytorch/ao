import unittest

from benchmarks.microbenchmarks.benchmark_inference import run
from benchmarks.microbenchmarks.utils import BenchmarkConfig


class TestBenchmarkInference(unittest.TestCase):
    def setUp(self):
        self.params = {
            "high_precision_dtype": "torch.float32",  # Use float32 for testing
            "compile": False,
            "device": "cpu",  # Use CPU for testing
            "model_type": "linear",
        }
        self.config = BenchmarkConfig(
            quantization="baseline",
            params=self.params,
            shape_name="test",
            shape=[16, 32, 8],  # Small shape for testing
            output_dir="benchmarks/microbenchmarks/test/test_output/",
        )

    def test_run_inference(self):
        result = run(self.config)

        # Check result contains all config attributes
        for key in self.config.to_dict():
            self.assertIn(key, result)

        # Check benchmark result is present and reasonable
        self.assertIn("benchmark_model_inference_in_microseconds", result)
        self.assertGreater(result["benchmark_model_inference_in_microseconds"], 0)


if __name__ == "__main__":
    unittest.main()
