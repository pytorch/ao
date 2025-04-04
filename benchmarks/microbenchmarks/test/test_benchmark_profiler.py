# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import unittest

import torch

from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    ToyLinearModel,
    generate_memory_profile,
    generate_model_profile,
)


class TestBenchmarkProfiler(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.test_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Set up a simple model and input for testing
        self.m, self.k, self.n = 1024, 1024, 1024
        self.dtype = torch.bfloat16
        self.model = ToyLinearModel(k=self.k, n=self.n, dtype=self.dtype)
        self.input_data = torch.randn(1, self.k, dtype=self.dtype)

        # Move to appropriate device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.input_data = self.input_data.to(self.device)

    def tearDown(self):
        # Clean up any generated files
        import shutil

        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)

    def test_profiler_enabled(self):
        """Test that profiler works when enabled"""
        config = BenchmarkConfig(
            quantization=None,
            sparsity=None,
            params={
                "enable_profiler": True,
                "device": self.device,
            },
            shape_name="test",
            shape=[self.m, self.k, self.n],
            output_dir=self.results_dir,
            benchmark_mode="inference",
        )

        profile_path = os.path.join(
            self.results_dir,
            "profiler",
            f"{config.name}_{self.m}_{self.k}_{self.n}_profile.json",
        )

        # Generate profile
        result_path, _ = generate_model_profile(
            self.model, self.input_data, profile_path
        )

        # Check that profile file exists and is not empty
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

        # Verify it's valid JSON
        with open(result_path) as f:
            profile_data = json.load(f)
        self.assertIsInstance(profile_data, dict)

    def test_profiler_basic_output(self):
        """Test that profiler output contains expected basic fields"""
        config = BenchmarkConfig(
            quantization=None,
            sparsity=None,
            params={
                "enable_profiler": True,
                "device": self.device,
            },
            shape_name="test",
            shape=[self.m, self.k, self.n],
            output_dir=self.results_dir,
            benchmark_mode="inference",
        )

        profile_path = os.path.join(
            self.results_dir,
            "profiler",
            f"{config.name}_{self.m}_{self.k}_{self.n}_profile.json",
        )

        result_path, _ = generate_model_profile(
            self.model, self.input_data, profile_path
        )

        with open(result_path) as f:
            data = json.load(f)

        # Check for required Chrome Trace Event format fields
        self.assertIn("traceEvents", data)
        self.assertTrue(isinstance(data["traceEvents"], list))

        # Check that we have some events
        self.assertGreater(len(data["traceEvents"]), 0)

        # Check event format
        event = data["traceEvents"][0]
        self.assertIn("name", event)
        self.assertIn("ph", event)  # Phase
        self.assertIn("ts", event)  # Timestamp
        self.assertIn("pid", event)  # Process ID

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_profiling(self):
        """Test CUDA profiling when available"""
        config = BenchmarkConfig(
            quantization=None,
            sparsity=None,
            params={
                "enable_profiler": True,
                "device": "cuda",
            },
            shape_name="test",
            shape=[self.m, self.k, self.n],
            output_dir=self.results_dir,
            benchmark_mode="inference",
        )

        profile_path = os.path.join(
            self.results_dir,
            "profiler",
            f"{config.name}_{self.m}_{self.k}_{self.n}_profile.json",
        )

        result_path, _ = generate_model_profile(
            self.model.cuda(), self.input_data.cuda(), profile_path
        )

        with open(result_path) as f:
            data = json.load(f)

        # Check for CUDA events
        cuda_events = [
            event for event in data["traceEvents"] if "CUDA" in event.get("name", "")
        ]
        self.assertGreater(len(cuda_events), 0)

    def test_memory_profiling(self):
        """Test memory profiling functionality for CUDA"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        config = BenchmarkConfig(
            quantization=None,
            sparsity=None,
            params={
                "enable_memory_profile": True,
                "device": "cuda",
            },
            shape_name="test",
            shape=[self.m, self.k, self.n],
            output_dir=self.results_dir,
            benchmark_mode="inference",
        )

        profile_path = os.path.join(
            self.results_dir,
            "profiler",
            f"{config.name}_{self.m}_{self.k}_{self.n}_memory_profile.json",
        )

        result_path, memory_stats = generate_memory_profile(
            self.model.cuda(), self.input_data.cuda(), profile_path
        )

        # Check that JSON profile file exists and is not empty
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

        # Check that pickle profile file exists and is not empty
        pickle_path = result_path.replace(".json", ".pickle")
        self.assertTrue(os.path.exists(pickle_path))
        self.assertGreater(os.path.getsize(pickle_path), 0)

        # Verify memory stats structure
        self.assertIn("peak_memory_allocated", memory_stats)
        self.assertIn("peak_memory_reserved", memory_stats)
        self.assertIn("total_memory_allocated", memory_stats)
        self.assertIn("total_memory_reserved", memory_stats)

        # Verify memory values are reasonable
        self.assertGreaterEqual(memory_stats["peak_memory_allocated"], 0)
        self.assertGreaterEqual(memory_stats["peak_memory_reserved"], 0)
        self.assertGreaterEqual(memory_stats["total_memory_allocated"], 0)
        self.assertGreaterEqual(memory_stats["total_memory_reserved"], 0)

        # Verify pickle file can be loaded
        with open(pickle_path, "rb") as f:
            from pickle import load

            snapshot = load(f)
            self.assertIsNotNone(snapshot)

        # Check that HTML visualization was generated
        html_path = pickle_path.replace(".pickle", ".html")
        if os.path.exists(
            os.path.dirname(os.path.dirname(torch.__file__))
            + "/torch/cuda/_memory_viz.py"
        ):
            self.assertTrue(os.path.exists(html_path))
            self.assertGreater(os.path.getsize(html_path), 0)


if __name__ == "__main__":
    unittest.main()
