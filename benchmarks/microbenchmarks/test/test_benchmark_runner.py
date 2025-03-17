# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmarks.microbenchmarks.benchmark_runner import (
    get_param_combinations,
    get_shapes_for_config,
    load_benchmark_configs,
    run_inference_benchmarks_from_config,
)


class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        self.test_config = {
            "benchmark_mode": "inference",
            "quantization_config_recipe_names": ["baseline", "int8wo"],
            "output_dir": self.temp_dir,  # Use temp directory
            "model_params": [
                {
                    "name": "test_model",
                    "matrix_shapes": [
                        {
                            "name": "custom",
                            "shapes": [[1024, 1024, 1024]],
                        }
                    ],
                    "high_precision_dtype": "torch.bfloat16",
                    "use_torch_compile": True,
                    "torch_compile_mode": "max-autotune",
                    "device": "cpu",
                    "model_type": "linear",
                }
            ],
        }
        self.config_path = Path(self.temp_dir) / "test_config.yml"
        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        # Clean up temporary directory and all its contents
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_get_shapes_for_config(self):
        shapes = get_shapes_for_config(
            self.test_config["model_params"][0]["matrix_shapes"]
        )
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], ("custom", [1024, 1024, 1024]))

    def test_get_param_combinations(self):
        model_param = self.test_config["model_params"][0]
        shapes, params = get_param_combinations(model_param)

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], ("custom", [1024, 1024, 1024]))
        self.assertEqual(params["high_precision_dtype"], "torch.bfloat16")
        self.assertEqual(params["use_torch_compile"], True)

    @patch("argparse.Namespace")
    def test_load_benchmark_configs(self, mock_args):
        mock_args.config = str(self.config_path)
        configs = load_benchmark_configs(mock_args)

        self.assertEqual(len(configs), 2)  # 2 quantization configs
        self.assertEqual(configs[0].benchmark_mode, "inference")
        self.assertEqual(configs[0].device, "cpu")

    def test_run_inference_benchmarks_from_config(self):
        configs = load_benchmark_configs(
            argparse.Namespace(config=str(self.config_path))
        )
        run_inference_benchmarks_from_config(configs)
        results_file = Path(self.temp_dir) / "results.csv"
        self.assertTrue(results_file.exists())


if __name__ == "__main__":
    unittest.main()
