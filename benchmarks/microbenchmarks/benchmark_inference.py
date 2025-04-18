# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Inference benchmark runner

This script runs inference benchmarks and generates a micro-benchmarking report for it.
- run() function is the main entry point for running inference benchmarks.
"""

import os
from copy import deepcopy
from pathlib import Path

import torch

from benchmarks.microbenchmarks.profiler import (
    generate_memory_profile,
    generate_model_profile,
    visualize_memory_profile,
)
from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    clean_caches,
    model_inference_time_in_ms,
    string_to_config,
)
from torchao.quantization import quantize_
from torchao.sparsity.sparse_api import sparsify_
from torchao.testing.model_architectures import (
    create_model_and_input_data,
)


def run(config: BenchmarkConfig) -> BenchmarkResult:
    """Run inference benchmarks"""
    try:
        clean_caches()  # Clean caches

        # Create output directory if it doesn't exist
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        base_model, input_data = create_model_and_input_data(
            config.model_type,
            config.m,
            config.k,
            config.n,
            high_precision_dtype=config.high_precision_dtype,
            device=config.device,
        )

        # Use quantize_ to apply each quantization function to the model
        m_copy = deepcopy(base_model).eval().to(config.device)
        ao_base_config = string_to_config(
            config.quantization,
            config.sparsity,
            high_precision_dtype=config.high_precision_dtype,
        )

        # Check if sparsity is requested and if the device is CUDA (sparsity operations require CUDA)
        is_cuda = config.device == "cuda" and torch.cuda.is_available()

        if config.sparsity is not None and (
            config.quantization is None or "baseline" in config.quantization
        ):
            if is_cuda:
                print(f"Applying {config.sparsity} sparsity to model")
                sparsify_(m_copy, ao_base_config)
            else:
                print(
                    f"Warning: Skipping {config.sparsity} sparsity as it requires CUDA, but device is {config.device}"
                )
        elif config.sparsity is None and (
            config.quantization is None or "baseline" in config.quantization
        ):
            pass  # No quantization or sparsity specified, do nothing
        else:
            print("Quantizing model....")
            quantize_(m_copy, ao_base_config)

        if config.use_torch_compile:
            print("Compiling model....")
            m_copy = torch.compile(
                m_copy, mode=config.torch_compile_mode, fullgraph=True
            )

        # Run benchmarks
        result = BenchmarkResult(config=config)
        # Store result in model for memory profiling
        m_copy._benchmark_result = result

        # Benchmark time to run an inference call for quantized model
        result.model_inference_time_in_ms = model_inference_time_in_ms(
            model=m_copy, input_data=input_data
        )

        # Run profiler if enabled
        if config.enable_profiler:
            print("Running profiler...")
            try:
                profiler_json_path = generate_model_profile(
                    model=m_copy,
                    input_data=input_data,
                    profile_file_path=os.path.join(
                        config.output_dir, "profiler", f"{config.name}_profile.json"
                    ),
                )
                result.profiler_json_path = profiler_json_path
            except Exception as e:
                print(f"Error running profiler: {e}")

        # Run memory profiler if enabled
        if config.enable_memory_profiler:
            print("Running memory profiler...")
            try:
                result.memory_profile_path, result.memory_stats = (
                    generate_memory_profile(
                        model=m_copy,
                        input_data=input_data,
                        profile_file_path=os.path.join(
                            config.output_dir,
                            "memory_profiler/pickle",
                            f"{config.name}_quant_{config.quantization}_sparsity_{config.sparsity}_memory_profile.pickle",
                        ),
                    )
                )

                if result.memory_profile_path:
                    result.memory_visualization_path = visualize_memory_profile(
                        result.memory_profile_path
                    )
            except ValueError as e:
                if "not enough values to unpack" in e:
                    print(
                        "Failed due to existing bugs, re-run the code to generate memory profile. Please raise an issue if it persists."
                    )
            except Exception as e:
                print(f"Error running memory profiler: {e}")
                import traceback

                traceback.print_exc()

        return result
    except Exception as e:
        print(f"Error in benchmark run: {config.name} with error: {e}")
        return None
