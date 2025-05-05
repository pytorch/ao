# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import pickle
import unittest

import torch
import torch.cuda

from benchmarks.microbenchmarks.profiler import (
    generate_memory_profile,
    generate_model_profile,
    visualize_memory_profile,
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

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_profiler_enabled(self):
        """Test that memory profiler works when enabled and CUDA is available"""
        config = BenchmarkConfig(
            quantization=None,
            sparsity=None,
            params={
                "enable_memory_profiler": True,
                "device": "cuda",
            },
            shape_name="test",
            shape=[self.m, self.k, self.n],
            output_dir=self.results_dir,
            benchmark_mode="inference",
        )

        memory_profile_path = os.path.join(
            self.results_dir,
            "memory_profiler",
            f"{config.name}_{self.m}_{self.k}_{self.n}_memory_profile.pickle",
        )

        # Generate memory profile
        result_path = generate_memory_profile(
            self.model, self.input_data, memory_profile_path
        )

        # Check that profile file exists and is not empty
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

        # Verify it's a valid pickle file
        try:
            with open(result_path, "rb") as f:
                pickle_data = pickle.load(f)
            self.assertIsNotNone(pickle_data)
        except Exception as e:
            self.fail(f"Failed to load pickle file: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_memory_profiler_visualization(self):
        """Test memory profile visualization"""
        config = BenchmarkConfig(
            quantization=None,
            sparsity=None,
            params={
                "enable_memory_profiler": True,
                "device": "cuda",
            },
            shape_name="test",
            shape=[self.m, self.k, self.n],
            output_dir=self.results_dir,
            benchmark_mode="inference",
        )

        memory_profile_path = os.path.join(
            self.results_dir,
            "memory_profiler",
            f"{config.name}_{self.m}_{self.k}_{self.n}_memory_profile.pickle",
        )

        # Create a simple mock memory profile as a pickle file
        # This is a simplified structure that mimics what torch.cuda.memory._dump_snapshot produces
        mock_profile_data = {
            "segments": [
                {
                    "device": 0,
                    "address": 1000,
                    "size": 1024 * 1024,  # 1MB
                    "stream": 0,
                }
            ],
            "external_annotations": [],
        }

        # Save mock profile as pickle
        os.makedirs(os.path.dirname(memory_profile_path), exist_ok=True)
        with open(memory_profile_path, "wb") as f:
            pickle.dump(mock_profile_data, f)

        # Generate visualization
        viz_path = visualize_memory_profile(memory_profile_path)

        # Check that visualization file exists
        self.assertTrue(os.path.exists(viz_path))
        self.assertTrue(viz_path.endswith(".html"))

    def test_memory_profiler_cuda_unavailable(self):
        """Test memory profiler behavior when CUDA is not available"""
        # Save original torch.cuda.is_available function
        original_is_available = torch.cuda.is_available

        try:
            # Mock torch.cuda.is_available to return False
            torch.cuda.is_available = lambda: False

            config = BenchmarkConfig(
                quantization=None,
                sparsity=None,
                params={
                    "enable_memory_profiler": True,
                    "device": "cpu",  # Force CPU to test CUDA unavailable case
                },
                shape_name="test",
                shape=[self.m, self.k, self.n],
                output_dir=self.results_dir,
                benchmark_mode="inference",
            )

            memory_profile_path = os.path.join(
                self.results_dir,
                "memory_profiler",
                f"{config.name}_{self.m}_{self.k}_{self.n}_memory_profile.json",
            )

            # Generate memory profile
            result = generate_memory_profile(
                self.model, self.input_data, memory_profile_path
            )

            # Should return None when CUDA is unavailable
            self.assertIsNone(result)

            # Should not create file when CUDA is unavailable
            self.assertFalse(os.path.exists(memory_profile_path))

        finally:
            # Restore original torch.cuda.is_available function
            torch.cuda.is_available = original_is_available


if __name__ == "__main__":
    unittest.main()
