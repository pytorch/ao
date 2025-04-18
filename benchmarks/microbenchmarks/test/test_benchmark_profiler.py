# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import unittest

import torch

from benchmarks.microbenchmarks.profiler import (
    generate_model_profile,
)
from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
)
from torchao.testing.model_architectures import ToyLinearModel


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
        result_path = generate_model_profile(self.model, self.input_data, profile_path)

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

        result_path = generate_model_profile(self.model, self.input_data, profile_path)

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

        result_path = generate_model_profile(
            self.model.cuda(), self.input_data.cuda(), profile_path
        )

        with open(result_path) as f:
            data = json.load(f)

        # Check for CUDA events
        cuda_events = [
            event for event in data["traceEvents"] if "cuda" in event.get("name", "")
        ]
        self.assertGreater(len(cuda_events), 0)


if __name__ == "__main__":
    unittest.main()
