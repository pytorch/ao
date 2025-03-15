import os
import tempfile
import unittest

import torch

from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    LNLinearSigmoid,
    ToyLinearModel,
    clean_caches,
    create_model_and_input,
    generate_results_csv,
)


class TestUtils(unittest.TestCase):
    def test_benchmark_config(self):
        params = {
            "high_precision_dtype": "torch.bfloat16",
            "use_torch_compile": True,
            "torch_compile_mode": "max-autotune",
            "device": "cuda",
            "model_type": "linear",
        }
        config = BenchmarkConfig(
            quantization="int8wo",
            params=params,
            shape_name="custom",
            shape=[1024, 1024, 1024],
            output_dir="test_output",
        )

        self.assertEqual(config.quantization, "int8wo")
        self.assertEqual(config.m, 1024)
        self.assertEqual(config.k, 1024)
        self.assertEqual(config.n, 1024)
        self.assertEqual(config.high_precision_dtype, torch.bfloat16)
        self.assertEqual(config.use_torch_compile, True)
        self.assertEqual(config.torch_compile_mode, "max-autotune")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.model_type, "linear")
        self.assertEqual(config.output_dir, "test_output")
        self.assertEqual(
            config.name, "benchmark_int8wo_linear_m1024_k1024_n1024_compile"
        )

    def test_toy_linear_model(self):
        model = ToyLinearModel(k=64, n=32, dtype=torch.float32)
        x = torch.randn(16, 64)
        out = model(x)
        self.assertEqual(out.shape, (16, 32))
        self.assertEqual(out.dtype, torch.float32)

    def test_ln_linear_sigmoid(self):
        model = LNLinearSigmoid(fc_dim1=64, fc_dim2=32, dtype=torch.float32)
        x = torch.randn(16, 64)
        out = model(x)
        self.assertEqual(out.shape, (16, 32))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(
            torch.all((out >= 0) & (out <= 1))
        )  # Check sigmoid output range

    def test_create_model_and_input(self):
        m, k, n = 16, 64, 32
        model, input_data = create_model_and_input(
            model_type="linear",
            m=m,
            k=k,
            n=n,
            high_precision_dtype=torch.float32,
            device="cpu",
        )
        self.assertIsInstance(model, ToyLinearModel)
        self.assertEqual(input_data.shape, (m, k))

        model, input_data = create_model_and_input(
            model_type="ln_linear_sigmoid",
            m=m,
            k=k,
            n=n,
            high_precision_dtype=torch.float32,
            device="cpu",
        )
        self.assertIsInstance(model, LNLinearSigmoid)
        self.assertEqual(input_data.shape, (m, k))

    def test_generate_results_csv(self):
        results = [
            BenchmarkResult(
                BenchmarkConfig(
                    quantization="int8wo",
                    params={},
                    shape_name="custom",
                    shape=[1024, 1024, 1024],
                    output_dir="test_output",
                ),
            ),
            BenchmarkResult(
                BenchmarkConfig(
                    quantization="int4wo",
                    params={},
                    shape_name="custom",
                    shape=[1024, 1024, 1024],
                    output_dir="test_output",
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_results_csv(results, tmp_dir)
            csv_path = os.path.join(tmp_dir, "results.csv")
            self.assertTrue(os.path.exists(csv_path))

    def test_clean_caches(self):
        # Just test that it runs without error
        clean_caches()


if __name__ == "__main__":
    unittest.main()
