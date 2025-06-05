#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CI Microbenchmark Runner for PyTorch OSS Benchmark Database

This script runs microbenchmarks for various quantization types (int8wo, int8dq, float8wo, float8dq)
and outputs results in the format required by the PyTorch OSS benchmark database.
It reuses functionality from benchmark_runner.py and only adds CI-specific code.

Usage:
    python run_ci_microbenchmarks.py --config benchmark_config.yml

The YAML file should contain all necessary configuration parameters for the benchmarks.
"""

import argparse
import json
import platform
from typing import Any, Dict, List

import torch

from benchmarks.microbenchmarks.benchmark_inference import run as run_inference
from benchmarks.microbenchmarks.benchmark_runner import (
    load_benchmark_configs,
)
from benchmarks.microbenchmarks.utils import clean_caches


def create_benchmark_result(
    benchmark_name: str,
    shape: List[int],
    metric_name: str,
    metric_values: List[float],
    quant_type: str,
    device: str,
) -> Dict[str, Any]:
    """Create a benchmark result in the PyTorch OSS benchmark database format.

    Args:
        benchmark_name: Name of the benchmark
        shape: List of shape dimensions [M, K, N]
        metric_name: Name of the metric
        metric_values: List of metric values
        quant_type: Quantization type
        device: Device type (cuda/cpu)

    Returns:
        Dictionary containing the benchmark result in the required format
    """
    print(
        f"Creating benchmark result for {benchmark_name} with shape {shape} and metric {metric_name}"
    )

    # Map device to benchmark device name
    benchmark_device = (
        torch.cuda.get_device_name(0)
        if device == "cuda"
        else platform.processor()
        if device == "cpu"
        else "unknown"
    )

    # Format shape as M-K-N
    mkn_name = f"{shape[0]}-{shape[1]}-{shape[2]}" if len(shape) == 3 else "unknown"

    return {
        "benchmark": {
            "name": "micro-benchmark api",
            "mode": "inference",
            "dtype": quant_type,
            "extra_info": {
                "device": device,
                "arch": benchmark_device,
            },
        },
        "model": {
            "name": mkn_name,  # name in M-K-N format
            "type": "micro-benchmark custom layer",  # type
            "origins": ["torchao"],
        },
        "metric": {
            "name": f"{metric_name}(wrt bf16)",  # name with unit
            "benchmark_values": metric_values,  # benchmark_values
            "target_value": 0.0,  # TODO: Will need to define the target value
        },
        "runners": [],
        "dependencies": {},
    }


def run_ci_benchmarks(config_path: str) -> List[Dict[str, Any]]:
    """Run benchmarks using configurations from YAML file and return results in OSS format.

    Args:
        config_path: Path to the benchmark configuration file

    Returns:
        List of benchmark results in the PyTorch OSS benchmark database format
    """
    # Load configuration using existing function
    configs = load_benchmark_configs(argparse.Namespace(config=config_path))
    results = []

    # Run benchmarks for each config
    for config in configs:
        # Run benchmark using existing function
        clean_caches()
        result = run_inference(config)

        if result is not None:
            # Create benchmark result in OSS format
            benchmark_result = create_benchmark_result(
                benchmark_name="TorchAO Quantization Benchmark",
                shape=[config.m, config.k, config.n],
                metric_name="speedup",
                metric_values=[result.speedup],
                quant_type=config.quantization,
                device=config.device,
            )
            results.append(benchmark_result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks and output results in PyTorch OSS benchmark database format"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    # Run benchmarks
    results = run_ci_benchmarks(args.config)

    # Save results to JSON file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
