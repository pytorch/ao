# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    BlockSparseWeightConfig,
    Float8DynamicActivationFloat8SemiSparseWeightConfig,
    Int4WeightOnlyConfig,
    SemiSparseWeightConfig,
    clean_caches,
    generate_results_csv,
    get_default_device,
    string_to_config,
)
from torchao.testing.model_architectures import (
    LNLinearActivationModel,
    ToyLinearModel,
    create_model_and_input_data,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_params = {
            "name": "test_model",
            "high_precision_dtype": "torch.bfloat16",
            "use_torch_compile": True,
            "torch_compile_mode": "max-autotune",
            "device": "cpu",
            "model_type": "linear",
        }
        self.test_shape = [1024, 1024, 1024]
        self.test_output_dir = Path("test_output")

    def test_benchmark_config(self):
        config = BenchmarkConfig(
            quantization="baseline",
            sparsity=None,
            params=self.test_params,
            shape_name="custom",
            shape=self.test_shape,
            output_dir=str(self.test_output_dir),
            benchmark_mode="inference",
        )

        self.assertEqual(config.quantization, "baseline")
        self.assertEqual(config.m, 1024)
        self.assertEqual(config.k, 1024)
        self.assertEqual(config.n, 1024)
        self.assertEqual(config.high_precision_dtype, torch.bfloat16)
        self.assertEqual(config.use_torch_compile, True)
        self.assertEqual(config.torch_compile_mode, "max-autotune")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.model_type, "linear")
        self.assertEqual(config.benchmark_mode, "inference")

    def test_benchmark_result(self):
        config = BenchmarkConfig(
            quantization="baseline",
            sparsity=None,
            params=self.test_params,
            shape_name="custom",
            shape=self.test_shape,
            output_dir=str(self.test_output_dir),
            benchmark_mode="inference",
        )
        result = BenchmarkResult(config=config)

        self.assertEqual(result.config, config)
        self.assertEqual(result.model_inference_time_in_ms, 0.0)

    def test_get_default_device(self):
        # Test CPU fallback
        device = get_default_device("not_a_real_device")
        self.assertEqual(device, "cpu")

        # Test explicit CPU request
        device = get_default_device("cpu")
        self.assertEqual(device, "cpu")

    def test_string_to_config(self):
        # Test baseline
        config = string_to_config("baseline", None)
        self.assertIsNone(config)

        # Test int8wo
        config = string_to_config("int8wo", None)
        self.assertIsNotNone(config)

        # Test invalid config
        config = string_to_config("not_a_real_config", None)
        self.assertIsNone(config)

    def test_string_to_config_sparsity(self):
        """Test sparsity config generation"""
        # Test semi-sparse config
        config = string_to_config(None, "semi-sparse")
        self.assertIsInstance(config, SemiSparseWeightConfig)

        # Test block sparse config
        config = string_to_config(None, "block")
        self.assertIsInstance(config, BlockSparseWeightConfig)

        # Test combined sparsity and quantization
        config = string_to_config("marlin", "semi-sparse")
        self.assertIsInstance(config, Int4WeightOnlyConfig)

        # Test float8 with semi-sparse
        config = string_to_config("float8dq", "semi-sparse")
        self.assertIsInstance(
            config, Float8DynamicActivationFloat8SemiSparseWeightConfig
        )

    def test_block_sparsity_with_baseline_quantization(self):
        """Test that block sparsity with baseline quantization returns BlockSparseWeightConfig"""
        config = string_to_config("baseline", "block")
        self.assertIsInstance(config, BlockSparseWeightConfig)

    def test_block_sparsity_with_non_baseline_quantization(self):
        """Test that block sparsity with non-baseline quantization still returns BlockSparseWeightConfig"""
        # Block sparsity should take precedence over any quantization method
        config = string_to_config("int8wo", "block")
        self.assertIsInstance(config, BlockSparseWeightConfig)

        config = string_to_config("int4wo", "block")
        self.assertIsInstance(config, BlockSparseWeightConfig)

        config = string_to_config("marlin", "block")
        self.assertIsInstance(config, BlockSparseWeightConfig)

    def test_invalid_sparsity(self):
        """Test invalid sparsity config generation"""
        from benchmarks.microbenchmarks.benchmark_runner import (
            get_quantization_sparsity_recipes,
        )

        with self.assertRaises(ValueError):
            get_quantization_sparsity_recipes(["baseline"], ["invalid_sparsity"])

    def test_toy_linear_model(self):
        model = ToyLinearModel(k=64, n=32, dtype=torch.float32)
        x = torch.randn(16, 64)
        out = model(x)
        self.assertEqual(out.shape, (16, 32))
        self.assertEqual(out.dtype, torch.float32)

    def test_ln_linear_sigmoid(self):
        model = LNLinearActivationModel(fc_dim1=64, fc_dim2=32, dtype=torch.float32)
        x = torch.randn(16, 64)
        out = model(x)
        self.assertEqual(out.shape, (16, 32))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(
            torch.all((out >= 0) & (out <= 1))
        )  # Check sigmoid output range

    def test_create_model_and_input_data(self):
        m, k, n = 16, 64, 32
        model, input_data = create_model_and_input_data(
            model_type="linear",
            m=m,
            k=k,
            n=n,
            high_precision_dtype=torch.float32,
            device="cpu",
        )
        self.assertIsInstance(model, ToyLinearModel)
        self.assertEqual(input_data.shape, (m, k))

        model, input_data = create_model_and_input_data(
            model_type="ln_linear_sigmoid",
            m=m,
            k=k,
            n=n,
            high_precision_dtype=torch.float32,
            device="cpu",
        )
        self.assertIsInstance(model, LNLinearActivationModel)
        self.assertEqual(input_data.shape, (m, k))

    def test_generate_results_csv(self):
        results = [
            BenchmarkResult(
                BenchmarkConfig(
                    quantization="int8wo",
                    sparsity=None,
                    params={},
                    shape_name="custom",
                    shape=[1024, 1024, 1024],
                    output_dir="test_output",
                    benchmark_mode="inference",
                ),
            ),
            BenchmarkResult(
                BenchmarkConfig(
                    quantization="int4wo",
                    sparsity=None,
                    params={},
                    shape_name="custom",
                    shape=[1024, 1024, 1024],
                    output_dir="test_output",
                    benchmark_mode="inference",
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_results_csv(results, tmp_dir)

            # Check if any CSV file with the timestamp-based naming pattern was created
            csv_files = list(Path(tmp_dir).glob("results_*.csv"))
            self.assertTrue(len(csv_files) > 0, "No results CSV file was created")

    def test_clean_caches(self):
        # Just test that it runs without error
        clean_caches()


if __name__ == "__main__":
    unittest.main()
