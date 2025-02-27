import os
import tempfile
import unittest

import yaml

from benchmarks.microbenchmarks.benchmark_runner import (
    get_shapes_for_config,
    load_benchmark_configs,
    run_benchmarks_from_config,
)


class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.config = {
            "quantization_config_recipe_names": ["baseline", "int8wo"],
            "output_dir": "benchmarks/microbenchmarks/test/test_output",
            "model_params": {
                "matrix_shapes": [
                    {
                        "name": "custom",
                        "shapes": [[16, 32, 8]],  # Small shape for testing
                    }
                ],
                "high_precision_dtype": "torch.float32",
                "compile": False,
                "device": "cpu",
                "model_type": "linear",
            },
        }

        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.config_path):
            os.unlink(self.config_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

        # Clean up test output directory
        results_file = os.path.join(self.config["output_dir"], "results.csv")
        if os.path.exists(results_file):
            os.unlink(results_file)
        if os.path.exists(self.config["output_dir"]):
            os.rmdir(self.config["output_dir"])

    def test_get_shapes_for_config(self):
        shape_config = {
            "name": "custom",
            "shapes": [[1024, 1024, 1024], [2048, 2048, 2048]],
        }
        shapes = get_shapes_for_config(shape_config)
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0], ("custom", [1024, 1024, 1024]))
        self.assertEqual(shapes[1], ("custom", [2048, 2048, 2048]))

        with self.assertRaises(NotImplementedError):
            get_shapes_for_config({"name": "unsupported", "shapes": []})

    def test_load_benchmark_configs(self):
        configs = load_benchmark_configs(self.config_path)
        self.assertEqual(len(configs), 2)  # 2 quantizations * 1 shape
        self.assertEqual(configs[0].quantization, "baseline")
        self.assertEqual(configs[1].quantization, "int8wo")

    def test_run_benchmarks_from_config(self):
        run_benchmarks_from_config(self.config_path)
        results_file = os.path.join(self.config["output_dir"], "results.csv")
        self.assertTrue(os.path.exists(results_file))


if __name__ == "__main__":
    unittest.main()
