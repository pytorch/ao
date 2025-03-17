# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Benchmark Runner

This is the main entry point for the benchmarking application. It reads the YAML configuration
file and orchestrates the entire benchmarking process by:
- Loading and validating benchmark configurations
- Executing benchmark scenarios
- Collecting and processing results
- Generating reports

Usage:
    python benchmark_runner.py [config.yaml]

The YAML file should contain all necessary configuration parameters for the benchmarks.
"""

import argparse
from itertools import product
from typing import Any, Dict, List, Tuple

import yaml

from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    generate_results_csv,
    print_results,
)


def get_shapes_for_config(
    shape_configs: List[Dict[str, Any]],
) -> List[Tuple[str, List[int]]]:
    """Get shapes for a given configuration.

    Args:
        shape_configs: List of shape configurations from YAML

    Returns:
        List of tuples containing (shape_name, shape)
    """
    shapes = []
    for shape_config in shape_configs:
        name = shape_config["name"]
        if name == "custom":
            shapes.extend([(name, shape) for shape in shape_config["shapes"]])
        else:
            raise NotImplementedError(
                f"Shape config {name} not supported. Currently only supports custom shapes."
            )
    return shapes


def get_param_combinations(model_param):
    """Extract all parameter combinations from a model config"""
    # Get all shapes
    shapes = get_shapes_for_config(model_param["matrix_shapes"])

    # Extract all other parameters (excluding matrix_shapes)
    base_params = {
        key: value for key, value in model_param.items() if key not in ["matrix_shapes"]
    }

    return shapes, base_params


def load_benchmark_configs(cli_args: argparse.Namespace) -> List[BenchmarkConfig]:
    """Load benchmark configurations from CLI arguments and YAML file."""
    with open(cli_args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "benchmarks/microbenchmarks/results")
    benchmark_mode = config.get("benchmark_mode", "inference")

    # Create all possible combinations
    configs = []
    for model_param in config["model_params"]:
        shapes, params = get_param_combinations(model_param)

        # Create configs for all combinations
        for quant_config, (shape_name, shape) in product(
            config.get("quantization_config_recipe_names", ["baseline"]), shapes
        ):
            configs.append(
                BenchmarkConfig(
                    quantization=quant_config,
                    params=params,
                    shape_name=shape_name,
                    shape=shape,
                    output_dir=output_dir,
                    benchmark_mode=benchmark_mode,
                )
            )

    return configs


def run_inference_benchmarks_from_config(configs: List[BenchmarkConfig]) -> None:
    """Run benchmarks using configurations from YAML file"""
    from benchmarks.microbenchmarks.benchmark_inference import run as run_inference

    results = []
    print("Benchmarking Inference ......")
    for config in configs:
        try:
            print(f"Running: {config.name}")
            result = run_inference(config)  # Pass the config object directly
            results.append(result)
        except Exception as e:
            print(f"Error running benchmark {config.name}: {e}")
            continue

    # Add results to csv
    generate_results_csv(results, configs[0].output_dir)

    # Print results
    print_results(results)

    # TODO: Process results: Speedups:
    # 1. For different shapes for same model and quantization
    # 2. For different quantizations for same model and shape
    # 3. For different models for same quantization


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmarks from config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark configuration file",
    )
    # TODO: Add support for args to override config values and run smaller benchmarks
    args = parser.parse_args()

    configs = load_benchmark_configs(cli_args=args)
    # Run benchmarks
    if configs[0].benchmark_mode == "inference":
        run_inference_benchmarks_from_config(configs)
    elif configs[0].benchmark_mode == "training":
        print("Training mode not implemented yet")
    else:
        raise ValueError(
            f"Invalid benchmark mode: {configs[0].benchmark_mode}, choose from inference or training"
        )

    # TODO: Add support for args to override config values and run smaller benchmarks
