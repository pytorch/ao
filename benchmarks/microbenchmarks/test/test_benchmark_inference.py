import unittest

from benchmarks.microbenchmarks.benchmark_inference import run
from benchmarks.microbenchmarks.utils import BenchmarkConfig


class TestBenchmarkInference(unittest.TestCase):
    def setUp(self):
        self.params = {
            "high_precision_dtype": "torch.float32",  # Use float32 for testing
            "use_torch_compile": False,
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

        # Check benchmark result is present and reasonable
        self.assertTrue(hasattr(result, "model_inference_time_in_ms"))
        self.assertGreater(result.model_inference_time_in_ms, 0)


if __name__ == "__main__":
    unittest.main()
