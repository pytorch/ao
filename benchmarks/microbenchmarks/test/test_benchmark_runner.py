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
    get_quantization_sparsity_recipes,
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
        # Test custom shapes
        shapes = get_shapes_for_config(
            self.test_config["model_params"][0]["matrix_shapes"]
        )
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], ("custom", [1024, 1024, 1024]))

        # Test llama shapes
        llama_shapes = get_shapes_for_config([{"name": "llama"}])
        self.assertEqual(len(llama_shapes), 4)  # 4 LLaMa shapes
        self.assertTrue(
            any(name.startswith("llama_attn.wqkv") for name, _ in llama_shapes)
        )
        self.assertTrue(
            any(name.startswith("llama_attn.w0") for name, _ in llama_shapes)
        )
        self.assertTrue(
            any(name.startswith("llama_ffn.w13") for name, _ in llama_shapes)
        )
        self.assertTrue(
            any(name.startswith("llama_ffn.w2") for name, _ in llama_shapes)
        )

        # Test pow2 shapes
        pow2_shapes = get_shapes_for_config(
            [{"name": "pow2", "min_power": 10, "max_power": 12}]
        )
        self.assertEqual(len(pow2_shapes), 3)  # 3 powers of 2 (10, 11, 12)
        self.assertEqual(pow2_shapes[0], ("pow2_0", [1024, 1024, 1024]))  # 2^10
        self.assertEqual(pow2_shapes[1], ("pow2_1", [2048, 2048, 2048]))  # 2^11
        self.assertEqual(pow2_shapes[2], ("pow2_2", [4096, 4096, 4096]))  # 2^12

        # Test pow2_extended shapes
        pow2_extended_shapes = get_shapes_for_config(
            [{"name": "pow2_extended", "min_power": 10, "max_power": 11}]
        )
        self.assertEqual(
            len(pow2_extended_shapes), 4
        )  # 2 powers of 2, each with 2 variants
        self.assertEqual(
            pow2_extended_shapes[0], ("pow2_extended_0", [1024, 1024, 1024])
        )  # 2^10
        self.assertEqual(
            pow2_extended_shapes[1], ("pow2_extended_1", [1536, 1536, 1536])
        )  # 2^10 + 2^9
        self.assertEqual(
            pow2_extended_shapes[2], ("pow2_extended_2", [2048, 2048, 2048])
        )  # 2^11
        self.assertEqual(
            pow2_extended_shapes[3], ("pow2_extended_3", [3072, 3072, 3072])
        )  # 2^11 + 2^10

        # Test sweep shapes (limited to a small range for testing)
        sweep_shapes = get_shapes_for_config(
            [{"name": "sweep", "min_power": 8, "max_power": 9}]
        )
        # For min_power=8, max_power=9, we should have 8 shapes (2^3 = 8 combinations)
        self.assertEqual(len(sweep_shapes), 8)
        # Check that all shapes have the expected format
        for name, shape in sweep_shapes:
            self.assertTrue(name.startswith("sweep_"))
            self.assertEqual(len(shape), 3)  # [M, K, N]
            # Check that all dimensions are powers of 2 between 2^8 and 2^9
            for dim in shape:
                self.assertTrue(dim in [256, 512])  # 2^8, 2^9

    def test_get_param_combinations(self):
        model_param = self.test_config["model_params"][0]
        shapes, params = get_param_combinations(model_param)

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], ("custom", [1024, 1024, 1024]))
        self.assertEqual(params["high_precision_dtype"], "torch.bfloat16")

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

        # The results file is saved in the inference subdirectory with a timestamp-based name
        inference_dir = Path(self.temp_dir) / "inference"
        self.assertTrue(inference_dir.exists(), "Inference directory was not created")

        # Check if any CSV file was created in the inference directory
        csv_files = list(inference_dir.glob("results_*.csv"))
        self.assertTrue(len(csv_files) > 0, "No results CSV file was created")

    def test_get_quantization_sparsity_recipes(self):
        """Test generation of valid quantization and sparsity recipe combinations"""
        # Test basic combinations
        quant_recipes = ["baseline", "int8wo"]
        sparse_recipes = [None, "semi-sparse"]
        recipes = get_quantization_sparsity_recipes(quant_recipes, sparse_recipes)
        self.assertIn(("baseline", None), recipes)
        self.assertIn(("int8wo", None), recipes)
        self.assertIn(("baseline", "semi-sparse"), recipes)

        # Test marlin with semi-sparse
        quant_recipes = ["marlin", "baseline"]
        sparse_recipes = [None, "semi-sparse"]
        recipes = get_quantization_sparsity_recipes(quant_recipes, sparse_recipes)
        self.assertIn(("marlin", "semi-sparse"), recipes)
        self.assertIn(("baseline", None), recipes)

        # Test block sparsity
        quant_recipes = ["baseline"]
        sparse_recipes = [None, "block"]
        recipes = get_quantization_sparsity_recipes(quant_recipes, sparse_recipes)
        self.assertIn(("baseline", "block"), recipes)

    def test_none_string_raises_error(self):
        """Test that passing 'None' as a string raises an error"""
        quant_recipes = ["baseline"]
        sparse_recipes = ["None"]  # "None" as a string should raise an error
        with self.assertRaises(ValueError):
            get_quantization_sparsity_recipes(quant_recipes, sparse_recipes)

    def test_block_sparsity_with_quantization(self):
        """Test that block sparsity is only paired with baseline quantization"""
        quant_recipes = ["baseline", "int8wo", "int4wo", "marlin"]
        sparse_recipes = ["block"]
        recipes = get_quantization_sparsity_recipes(quant_recipes, sparse_recipes)

        # Block sparsity should only be paired with baseline
        self.assertIn(("baseline", "block"), recipes)
        self.assertNotIn(("int8wo", "block"), recipes)
        self.assertNotIn(("int4wo", "block"), recipes)
        self.assertNotIn(("marlin", "block"), recipes)

        # All quantization techniques should be run without sparsity
        self.assertIn(("baseline", None), recipes)
        self.assertIn(("int8wo", None), recipes)
        self.assertIn(("int4wo", None), recipes)
        self.assertIn(("marlin", None), recipes)

    def test_all_quantization_without_sparsity(self):
        """Test that all quantization techniques are run without sparsity"""
        quant_recipes = ["baseline", "int8wo", "int4wo", "marlin"]
        sparse_recipes = [None, "semi-sparse", "block"]
        recipes = get_quantization_sparsity_recipes(quant_recipes, sparse_recipes)

        # All quantization techniques should be run without sparsity
        for quant in quant_recipes:
            self.assertIn((quant, None), recipes)

    @patch(
        "benchmarks.microbenchmarks.benchmark_runner.get_quantization_sparsity_recipes"
    )
    def test_load_benchmark_configs_with_sparsity(self, mock_get_recipes):
        """Test loading benchmark configs with sparsity options"""
        # Mock get_quantization_sparsity_recipes to return a valid set of recipes
        mock_get_recipes.return_value = {("baseline", None), ("marlin", "semi-sparse")}

        test_config = {
            "benchmark_mode": "inference",
            "quantization_config_recipe_names": ["baseline", "marlin"],
            "sparsity_config_recipe_names": [
                None,
                "semi-sparse",
            ],  # Use None instead of "None"
            "output_dir": self.temp_dir,
            "model_params": [
                {
                    "matrix_shapes": [
                        {"name": "custom", "shapes": [[1024, 1024, 1024]]}
                    ],
                    "high_precision_dtype": "torch.bfloat16",
                    "device": "cpu",
                    "model_type": "linear",
                }
            ],
        }

        config_path = Path(self.temp_dir) / "test_sparsity_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)

        configs = load_benchmark_configs(argparse.Namespace(config=str(config_path)))

        # Check that we get configs for baseline and marlin with appropriate sparsity
        self.assertTrue(
            any(c.quantization == "baseline" and c.sparsity is None for c in configs)
        )
        self.assertTrue(
            any(
                c.quantization == "marlin" and c.sparsity == "semi-sparse"
                for c in configs
            )
        )


if __name__ == "__main__":
    unittest.main()
