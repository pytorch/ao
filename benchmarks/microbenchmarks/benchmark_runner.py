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
from typing import Any, Dict, List, Optional, Set, Tuple

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


def get_quantization_sparsity_recipes(
    quantization_recipes: List[str], sparsity_recipes: List[str]
) -> Set[Tuple[str, Optional[str]]]:
    """Generate valid quantization and sparsity recipes.

    Args:
        quantization_recipes: List of quantization recipes
        sparsity_recipes: List of sparsity recipes

    Returns:
        Set of tuples containing (quantization_recipe, sparsity_recipe)
        For block sparsity, quantization is always "baseline"
        All quantization techniques are also run without sparsity
    """
    config_recipes = set()

    # Always include baseline without sparsity
    config_recipes.add(("baseline", None))

    # Add all quantization techniques without sparsity
    for quant_config in quantization_recipes:
        config_recipes.add((quant_config, None))

    # Process combinations of quantization and sparsity
    for sparse_config in sparsity_recipes:
        if sparse_config is None:
            # Skip None sparsity as we've already added all quantization techniques without sparsity
            continue
        elif "block" in sparse_config:
            # For block sparsity, only pair with baseline quantization
            config_recipes.add(("baseline", sparse_config))
        elif "semi" in sparse_config or "2:4" in sparse_config:
            # For semi-sparse, only pair with compatible quantization methods
            for quant_config in quantization_recipes:
                if (
                    "marlin" in quant_config
                    or "int8dq" in quant_config
                    or "float8dq" in quant_config
                    or quant_config == "baseline"
                ):
                    config_recipes.add((quant_config, sparse_config))
        else:
            raise ValueError(f"Invalid sparsity recipe: {sparse_config}")

    return config_recipes


def load_benchmark_configs(cli_args: argparse.Namespace) -> List[BenchmarkConfig]:
    """Load benchmark configurations from CLI arguments and YAML file."""
    with open(cli_args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "benchmarks/microbenchmarks/results")
    benchmark_mode = config.get("benchmark_mode", "inference")

    # Create all possible combinations
    configs = []
    quantization_sparsity_recipes = get_quantization_sparsity_recipes(
        config.get("quantization_config_recipe_names", []),
        config.get("sparsity_config_recipe_names", []),
    )
    for model_param in config["model_params"]:
        shapes, params = get_param_combinations(model_param)

        # Create configs for all combinations
        for (quant_config, sparse_config), (shape_name, shape) in product(
            quantization_sparsity_recipes,
            shapes,
        ):
            configs.append(
                BenchmarkConfig(
                    quantization=quant_config,
                    sparsity=sparse_config,
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
    print("----------------- RUNNING BENCHMARKS FOR INFERENCE -----------------------")
    for config in configs:
        print("----------------------------------------")
        try:
            print(
                f"Running: {config.name} for Quantization: {config.quantization} and Sparsity: {config.sparsity}"
            )
            result = run_inference(config)  # Pass the config object directly
            results.append(result)
        except Exception:
            print(f"Error running benchmark {config.name}")
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
