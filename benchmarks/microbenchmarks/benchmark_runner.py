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
    TrainingBenchmarkConfig,
    generate_results_csv,
    print_results,
    print_training_results,
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
        elif name == "llama":
            # LLaMa 2 70B single-node weight shapes
            # assumes fused attn.wqkv and ffn.w13
            bsz, seq_len = 4, 4096
            M = bsz * seq_len
            llama_shapes = {
                "attn.wqkv": (M, 8192, 1280),
                "attn.w0": (M, 1024, 8192),
                "ffn.w13": (M, 8192, 7168),
                "ffn.w2": (M, 3584, 8192),
            }
            shapes.extend([(f"{name}_{k}", v) for k, v in llama_shapes.items()])
        elif name == "pow2":
            # Generate shapes with dimensions that are powers of 2
            min_power_of_2 = shape_config.get("min_power", 10)  # 1024
            max_power_of_2 = shape_config.get("max_power", 14)  # 16,384
            for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
                val = 2**power_of_2
                shapes.append((f"{name}_{idx}", [val, val, val]))
        elif name == "pow2_extended":
            # Generate shapes with dimensions that are powers of 2 and powers of 2 + half
            min_power_of_2 = shape_config.get("min_power", 10)  # 1024
            max_power_of_2 = shape_config.get("max_power", 14)  # 16,384
            for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
                val1 = 2**power_of_2
                val2 = 2**power_of_2 + 2 ** (power_of_2 - 1)
                shapes.append((f"{name}_{idx * 2}", [val1, val1, val1]))
                shapes.append((f"{name}_{idx * 2 + 1}", [val2, val2, val2]))
        elif name == "sweep":
            # Generate a sweep of shapes with different powers of 2 for M, K, N
            min_p2 = shape_config.get("min_power", 8)  # 256
            max_p2 = shape_config.get("max_power", 15)  # 32,768
            counter = 0
            for M_p2 in range(min_p2, max_p2 + 1):
                M = 2**M_p2
                for K_p2 in range(min_p2, max_p2 + 1):
                    K = 2**K_p2
                    for N_p2 in range(min_p2, max_p2 + 1):
                        N = 2**N_p2
                        shapes.append((f"{name}_{counter}", [M, K, N]))
                        counter += 1
        else:
            raise NotImplementedError(
                f"Shape config {name} not supported. Supported options: custom, llama, pow2, pow2_extended, sweep."
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
    quantization_recipes: List[str],
    sparsity_recipes: List[str],
    benchmark_mode: str = "inference",
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

    if benchmark_mode == "inference":
        # If inference, include baseline without sparsity
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


def load_benchmark_configs(cli_args: argparse.Namespace) -> List[Any]:
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
        benchmark_mode,
    )
    for model_param in config["model_params"]:
        shapes, params = get_param_combinations(model_param)

        # Create configs for all combinations
        for (quant_config, sparse_config), (shape_name, shape) in product(
            quantization_sparsity_recipes,
            shapes,
        ):
            if benchmark_mode == "inference":
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
            elif benchmark_mode == "training":
                # Extract training-specific parameters
                training_params = params.copy()
                scaling_type_input = config.get("scaling_type_input", "dynamic")
                scaling_type_weight = config.get("scaling_type_weight", "dynamic")
                scaling_type_grad_output = config.get(
                    "scaling_type_grad_output", "dynamic"
                )

                # Determine scaling granularity based on quantization string
                # If quantization contains "-row", use "rowwise", otherwise use the config value
                default_granularity = config.get("scaling_granularity", "tensorwise")
                if quant_config and ("row" in quant_config or "axis" in quant_config):
                    scaling_granularity = "axiswise"  # This will be mapped to AXISWISE in create_float8_config
                else:
                    scaling_granularity = default_granularity

                use_fast_accum = config.get("use_fast_accum", True)
                repeat_n = config.get("repeat_n", 100)

                # For training benchmarks, we don't use sparsity
                configs.append(
                    TrainingBenchmarkConfig(
                        quantization=quant_config,
                        sparsity=None,  # No sparsity for training
                        params=training_params,
                        shape_name=shape_name,
                        shape=shape,
                        output_dir=output_dir,
                        benchmark_mode=benchmark_mode,
                        scaling_type_input=scaling_type_input,
                        scaling_type_weight=scaling_type_weight,
                        scaling_type_grad_output=scaling_type_grad_output,
                        scaling_granularity=scaling_granularity,
                        use_fast_accum=use_fast_accum,
                        repeat_n=repeat_n,
                    )
                )
    return configs


def run_inference_benchmarks_from_config(configs: List[Any]) -> None:
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
            if result is not None:  # Only add successful results
                results.append(result)
        except Exception as e:
            print(f"Error running benchmark {config.name} with error: {e}")
            continue

    # Add results to csv if there are any
    if results:
        generate_results_csv(results, configs[0].output_dir)
        # Print results
        print_results(results)
    else:
        print("No benchmark results were collected. All benchmarks failed.")

    # TODO: Process results: Speedups:
    # 1. For different shapes for same model and quantization
    # 2. For different quantizations for same model and shape
    # 3. For different models for same quantization


def run_training_benchmarks_from_config(configs: List[Any]) -> None:
    """Run training benchmarks using configurations from YAML file"""
    from benchmarks.microbenchmarks.benchmark_training import run as run_training

    results = []
    print("----------------- RUNNING BENCHMARKS FOR TRAINING -----------------------")

    # Run all configs - each config will calculate its own reference baseline
    for config in configs:
        print("----------------------------------------")
        try:
            print(
                f"Running: {config.name} for Quantization: {config.quantization} and shape: {config.shape_name}: ({config.m}, {config.k}, {config.n})"
            )
            result = run_training(config)  # Pass the config object directly
            if result is not None:  # Only add successful results
                results.append(result)
        except Exception as e:
            print(f"Error running benchmark {config.name} with error: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Add results to csv if there are any
    if results:
        generate_results_csv(results, configs[0].output_dir)
        # Print results
        print_training_results(results)
    else:
        print("No benchmark results were collected. All benchmarks failed.")

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
        run_training_benchmarks_from_config(configs)
    else:
        raise ValueError(
            f"Invalid benchmark mode: {configs[0].benchmark_mode}, choose from inference or training"
        )

    # TODO: Add support for args to override config values and run smaller benchmarks
