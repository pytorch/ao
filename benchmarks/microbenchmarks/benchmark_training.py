# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Training benchmark runner

This script runs training benchmarks and generates a micro-benchmarking report for it.
- run() function is the main entry point for running training benchmarks.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.utils.benchmark as benchmark

from benchmarks.microbenchmarks.profiler import (
    generate_memory_profile,
    generate_model_profile,
    visualize_memory_profile,
)
from benchmarks.microbenchmarks.utils import (
    TrainingBenchmarkConfig,
    TrainingBenchmarkResult,
    clean_caches,
)

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 1979e12
h100_peak_tops_float8_tc = 3958e12

# Use strings as keys to avoid issues with torch.dtype objects
dtype_to_peak_tops = {
    "float32": h100_peak_flops_float32,
    "float16": h100_peak_flops_fp16_tc,
    "bfloat16": h100_peak_flops_fp16_tc,
    "float8_e4m3fn": h100_peak_tops_float8_tc,
    "float8_e5m2": h100_peak_tops_float8_tc,
}
from torchao.float8.config import (
    CastConfig,
    Float8LinearConfig,
    ScalingGranularity,
    ScalingType,
    e4m3_dtype,
    e5m2_dtype,
)
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_tensor import ScaledMMConfig
from torchao.testing.model_architectures import (
    create_model_and_input_data,
)

# Use the ScalingType and ScalingGranularity enums from torchao.float8.config
# instead of defining our own


def benchmark_torch_function_in_microseconds(
    func: Any,
    *args,
    **kwargs,
) -> float:
    """Benchmark a torch function and return the median time in microseconds"""
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.blocked_autorange().median * 1e6


def create_float8_config(config: TrainingBenchmarkConfig) -> Float8LinearConfig:
    """Create a Float8LinearConfig from the benchmark config"""

    # Map string values to ScalingType enum values
    def map_scaling_type(scaling_type_str):
        if scaling_type_str == "dynamic":
            return ScalingType.DYNAMIC
        elif scaling_type_str == "disabled":
            return ScalingType.DISABLED
        else:
            raise ValueError(
                f"Unsupported scaling type: {scaling_type_str}. Supported types are 'dynamic' and 'disabled'."
            )

    # Map string values to ScalingGranularity enum values
    def map_scaling_granularity(granularity_str):
        if granularity_str == "tensorwise":
            return ScalingGranularity.TENSORWISE
        elif granularity_str == "rowwise" or granularity_str == "axiswise":
            return ScalingGranularity.AXISWISE
        else:
            raise ValueError(
                f"Unsupported scaling granularity: {granularity_str}. Supported granularities are 'tensorwise' and 'rowwise'."
            )

    scaling_type_input = map_scaling_type(config.scaling_type_input)
    scaling_type_weight = map_scaling_type(config.scaling_type_weight)
    scaling_type_grad_output = map_scaling_type(config.scaling_type_grad_output)
    scaling_granularity = map_scaling_granularity(config.scaling_granularity)

    # Explicitly set target_dtype to avoid KeyError in CastConfig.short_str()
    cast_config_input = CastConfig(
        scaling_type=scaling_type_input,
        scaling_granularity=scaling_granularity,
        target_dtype=e4m3_dtype,  # Explicitly set the target_dtype
    )
    cast_config_weight = CastConfig(
        scaling_type=scaling_type_weight,
        scaling_granularity=scaling_granularity,
        target_dtype=e4m3_dtype,  # Explicitly set the target_dtype
    )
    cast_config_grad_output = CastConfig(
        scaling_type=scaling_type_grad_output,
        scaling_granularity=scaling_granularity,
        target_dtype=e5m2_dtype,  # Explicitly set the target_dtype
    )

    return Float8LinearConfig(
        cast_config_input=cast_config_input,
        cast_config_weight=cast_config_weight,
        cast_config_grad_output=cast_config_grad_output,
    )


def n_times(n: int, fn: Any, *args, **kwargs):
    """Wrap a function to execute it n times"""

    def wrapper(*args, **kwargs):
        for _ in range(n):
            fn(*args, **kwargs)

    return wrapper


def run_training_benchmark(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: TrainingBenchmarkConfig,
) -> Tuple[float, float, float]:
    """Run training benchmark and return forward, backward, and total times in milliseconds"""

    # Define benchmark functions similar to bench_linear_float8.py
    def forward_pass():
        return model(input_data).sum()

    def forward_backward_pass():
        model.zero_grad()  # Reset gradients before each run
        loss = model(input_data).sum()
        loss.backward()

    # Measure forward pass time
    forward_time = (
        benchmark_torch_function_in_microseconds(n_times(config.repeat_n, forward_pass))
        * 1e-3
        / config.repeat_n
    )  # Convert to ms

    # Measure forward+backward pass time
    total_time = (
        benchmark_torch_function_in_microseconds(
            n_times(config.repeat_n, forward_backward_pass)
        )
        * 1e-3
        / config.repeat_n
    )  # Convert to ms

    # Calculate backward time
    backward_time = total_time - forward_time

    return forward_time, backward_time, total_time


def run(config: TrainingBenchmarkConfig) -> TrainingBenchmarkResult:
    """Run training benchmarks"""
    try:
        # Check if model type is supported
        if config.model_type != "linear":
            print(
                f"Error: Model type '{config.model_type}' is not supported. Only 'linear' model type is currently implemented."
            )
            return None

        # Note: Sparsity is not supported for training benchmarks
        if config.sparsity:
            print(
                f"Warning: Sparsity '{config.sparsity}' is not supported for training benchmarks and will be ignored."
            )

        clean_caches()  # Clean caches

        # Create output directory if it doesn't exist
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Create the base model and input data
        base_model, input_data = create_model_and_input_data(
            config.model_type,
            config.m,
            config.k,
            config.n,
            high_precision_dtype=config.high_precision_dtype,
            device=config.device,
        )

        # Create a copy for reference benchmarking
        ref_model = deepcopy(base_model).to(config.device)

        # Create result object
        result = TrainingBenchmarkResult(config=config)

        # Benchmark reference model
        print(f"Benchmarking reference model ({config.high_precision_dtype})...")

        # Define benchmark functions similar to bench_linear_float8.py
        def ref_forw_backward():
            ref_model(input_data).sum().backward()
            ref_model.zero_grad()

        # Wrap the function to execute it multiple times
        REPEAT_N = config.repeat_n
        ref_forw_backward_repeated = n_times(REPEAT_N, ref_forw_backward)

        # Compile if requested
        if config.use_torch_compile:
            # Inductor settings
            torch._dynamo.config.cache_size_limit = 1000
            torch._dynamo.config.automatic_dynamic_shapes = False
            # torch._dynamo.config.recompile_limit = 10000
            # torch._dynamo.config.accumulated_recompile_limit = 10000

            print("Compiling reference model...")
            ref_forw_backward_compiled = torch.compile(ref_forw_backward_repeated)

            # Warmup
            for _ in range(5):
                ref_forw_backward_compiled()

            # Benchmark
            ref_time = (
                benchmark_torch_function_in_microseconds(ref_forw_backward_compiled)
                * 1e-6
                / REPEAT_N
            )
        else:
            # Warmup without compilation
            for _ in range(5):
                ref_forw_backward()

            # Benchmark
            ref_time = (
                benchmark_torch_function_in_microseconds(ref_forw_backward_repeated)
                * 1e-6
                / REPEAT_N
            )

        # Store reference time in milliseconds
        ref_total_time = ref_time * 1000  # Convert to ms

        # For simplicity, we'll estimate forward and backward times
        # Typically backward is ~2x the forward time for linear layers
        ref_forward_time = ref_total_time / 3
        ref_backward_time = ref_total_time * 2 / 3

        result.reference_forward_time_ms = ref_forward_time
        result.reference_backward_time_ms = ref_backward_time
        result.reference_total_time_ms = ref_total_time

        # Calculate reference TOPS metrics directly
        M, K, N = config.m, config.k, config.n
        if ref_total_time > 0:
            # 3 * (2 * M * K * N) accounts for forward and backward passes
            # 3 = 1 (forward) + 2 (backward: gradient wrt input + gradient wrt weight)
            # 2 * M * K * N is the number of FLOPs for a matrix multiplication
            ref_tops_sec = float(3 * (2 * M * K * N)) / (ref_total_time * 1e-3)
            # Convert torch.dtype to string
            dtype_str = str(config.high_precision_dtype).split(".")[-1]
            if dtype_str in dtype_to_peak_tops:
                ref_pct_top_peak = ref_tops_sec / dtype_to_peak_tops[dtype_str]
            else:
                ref_pct_top_peak = 0.0
        else:
            ref_tops_sec = 0.0
            ref_pct_top_peak = 0.0

        result.ref_tops_sec = ref_tops_sec
        result.ref_pct_top_peak = ref_pct_top_peak

        # For baseline or no quantization, use reference model results
        if not config.quantization or config.quantization == "baseline":
            print("Using reference model results for baseline...")
            # Copy reference model results to the float8 columns for baseline
            result.forward_time_ms = ref_forward_time
            result.backward_time_ms = ref_backward_time
            result.total_time_ms = ref_total_time
            result.speedup = 1.0  # Speedup is 1.0 for baseline

            # For baseline, TOPS metrics are the same as reference
            result.tops_sec = ref_tops_sec
            result.pct_top_peak = ref_pct_top_peak
            # Set scaling representation to the actual dtype being used
            dtype_str = str(config.high_precision_dtype).split(".")[
                -1
            ]  # Extract 'float32' or 'bfloat16' from torch.dtype
            result.scaling_repr = f"baseline ({dtype_str})"

        # Create Float8 model if requested
        if config.quantization and "float8" in config.quantization:
            print(f"Creating Float8 model with {config.scaling_granularity} scaling...")

            # Create a fresh copy of the base model
            float8_model = deepcopy(base_model).to(config.device)

            # Create Float8 configuration
            float8_config = create_float8_config(config)

            # TODO: Add support for other models also, currently this only works for ToyLinearModel
            # Since we only support linear models, we know it has a linear1 attribute
            float8_model.linear1 = Float8Linear.from_float(
                float8_model.linear1,
                config=float8_config,
            )

            # Store scaling representation for reporting
            if hasattr(float8_model.linear1, "extra_repr"):
                result.scaling_repr = float8_model.linear1.extra_repr()
            else:
                result.scaling_repr = f"float8 ({config.scaling_granularity})"

            # # For test cases with mocked Float8Linear, get the scaling_repr from the mock
            # print("Checking for mock Float8Linear...")
            # mock_from_float = getattr(Float8Linear.from_float, "__self__", None)
            # print(f"mock_from_float: {mock_from_float}")

            # if mock_from_float is not None:
            #     print(f"Has return_value: {hasattr(mock_from_float, 'return_value')}")
            #     if hasattr(mock_from_float, "return_value"):
            #         print(f"return_value: {mock_from_float.return_value}")
            #         print(
            #             f"Has extra_repr: {hasattr(mock_from_float.return_value, 'extra_repr')}"
            #         )
            #         if hasattr(mock_from_float.return_value, "extra_repr"):
            #             result.scaling_repr = mock_from_float.return_value.extra_repr()
            #             print(f"Set scaling_repr to: {result.scaling_repr}")

            # Set fast accumulation if requested
            if hasattr(float8_model, "forward_config"):
                float8_model.forward_config = ScaledMMConfig(
                    False, config.use_fast_accum, False
                )

            # Store scaling representation for reporting
            if hasattr(float8_model, "extra_repr"):
                result.scaling_repr = float8_model.extra_repr()

            # Ensure scaling_repr is not empty
            if not result.scaling_repr:
                result.scaling_repr = f"float8 ({config.scaling_granularity})"

            # Define benchmark function for float8 model
            def float8_forw_backward():
                float8_model(input_data).sum().backward()
                float8_model.zero_grad()

            # Wrap the function to execute it multiple times
            float8_forw_backward_repeated = n_times(REPEAT_N, float8_forw_backward)

            # Compile if requested
            if config.use_torch_compile:
                # Inductor settings
                torch._dynamo.config.cache_size_limit = 1000
                torch._dynamo.config.automatic_dynamic_shapes = False
                # torch._dynamo.config.recompile_limit = 10000
                # torch._dynamo.config.accumulated_recompile_limit = 10000

                print("Compiling Float8 model...")
                float8_forw_backward_compiled = torch.compile(
                    float8_forw_backward_repeated
                )

                # Warmup
                for _ in range(5):
                    float8_forw_backward_compiled()

                # Benchmark
                float8_time = (
                    benchmark_torch_function_in_microseconds(
                        float8_forw_backward_compiled
                    )
                    * 1e-6
                    / REPEAT_N
                )
            else:
                # Warmup without compilation
                for _ in range(5):
                    float8_forw_backward()

                # Benchmark
                float8_time = (
                    benchmark_torch_function_in_microseconds(
                        float8_forw_backward_repeated
                    )
                    * 1e-6
                    / REPEAT_N
                )

            # Store float8 time in milliseconds
            total_time = float8_time * 1000  # Convert to ms

            # For simplicity, we'll estimate forward and backward times
            # Typically backward is ~2x the forward time for linear layers
            forward_time = total_time / 3
            backward_time = total_time * 2 / 3

            result.forward_time_ms = forward_time
            result.backward_time_ms = backward_time
            result.total_time_ms = total_time
            result.speedup = ref_forward_time / forward_time

            # Calculate TOPS metrics for float8 model directly
            if total_time > 0:
                tops_sec = float(3 * (2 * M * K * N)) / (total_time * 1e-3)
                # For float8 models, use float8 peak TOPS
                if config.quantization and "float8" in config.quantization:
                    pct_top_peak = tops_sec / dtype_to_peak_tops["float8_e4m3fn"]
                else:
                    # For other models, use the high precision dtype
                    dtype_str = str(config.high_precision_dtype).split(".")[-1]
                    if dtype_str in dtype_to_peak_tops:
                        pct_top_peak = tops_sec / dtype_to_peak_tops[dtype_str]
                    else:
                        pct_top_peak = 0.0
            else:
                tops_sec = 0.0
                pct_top_peak = 0.0

            result.tops_sec = tops_sec
            result.pct_top_peak = pct_top_peak

            # Run profiler if enabled
            if config.enable_profiler:
                print("Running profiler for Float8 model...")
                try:
                    profiler_json_path = generate_model_profile(
                        model=float8_model,
                        input_data=input_data,
                        profile_file_path=os.path.join(
                            config.output_dir,
                            "profiler",
                            f"{config._file_name}_profile.json",
                        ),
                    )
                    result.profiler_json_path = profiler_json_path
                except Exception as e:
                    print(f"Error running profiler: {e}")

            # Run memory profiler if enabled
            if config.enable_memory_profiler:
                print("Running memory profiler for Float8 model...")
                try:
                    result.memory_profile_path, result.memory_stats = (
                        generate_memory_profile(
                            model=float8_model,
                            input_data=input_data,
                            profile_file_path=os.path.join(
                                config.output_dir,
                                "memory_profiler/pickle",
                                f"{config._file_name}_memory_profile.pickle",
                            ),
                        )
                    )

                    if result.memory_profile_path:
                        result.memory_visualization_path = visualize_memory_profile(
                            result.memory_profile_path
                        )
                except Exception as e:
                    print(f"Error running memory profiler: {e}")

        return result
    except Exception as e:
        print(f"Error in benchmark run: {config.name} with error: {e}")
        import traceback

        traceback.print_exc()
        return None
