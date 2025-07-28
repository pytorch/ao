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
from typing import Dict, Tuple

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

# -----------------------------------------------------------------------------
# Baseline caching
#
# ``_BASELINE_CACHE`` maps a unique key to a tuple
# ``(eager_baseline_time, compile_baseline_time)``.  See ``_make_cache_key`` for the key
# construction.  Users should not access this cache directly; it is
# internal to this module.  The cache intentionally holds the
# uncompiled base model so that quantized versions can be derived
# without mutating the cached copy.

_BASELINE_CACHE: Dict[Tuple, Tuple[float, float]] = {}


def _make_cache_key(config: BenchmarkConfig) -> Tuple:
    """Create a key for caching based on benchmark configuration.

    Parameters that affect baseline performance are included:

    * model type (e.g. ``linear`` or ``transformer_block``)
    * shape dimensions (m, k, n)
    * high precision dtype (bf16, fp16, etc.)
    * device (cuda, cpu, mps)
    * compile settings (whether compile is enabled and compile mode)

    Sparsity and quantization settings are deliberately excluded
    because the baseline (non‑quantized, non‑sparse) performance is
    independent of those attributes.
    """
    return (
        config.model_type,
        config.m,
        config.k,
        config.n,
        config.high_precision_dtype,
        config.device,
        config.torch_compile_mode,
    )


def run(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run inference benchmarks.

    The function first checks if a baseline for the given configuration
    already exists in the internal cache.  If not, it measures the baseline
    inference time and stores the result.  When the baseline is cached,
    the function reuses the stored model and input data to
    benchmark quantized variants, avoiding redundant baseline measurements.

    Args:
        config (BenchmarkConfig): Benchmark configuration.

    Returns:
        BenchmarkResult: Result of the benchmark.
    """
    try:
        clean_caches()  # Clean caches

        # Create output directory if it doesn't exist
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Prepare result container
        result = BenchmarkResult(config=config)

        # Create model and input data
        base_model, input_data = create_model_and_input_data(
            config.model_type,
            config.m,
            config.k,
            config.n,
            high_precision_dtype=config.high_precision_dtype,
            device=config.device,
        )

        # Generate a cache key for the current configuration
        cache_key = _make_cache_key(config)

        # Check if the baseline for this configuration has been computed
        if cache_key not in _BASELINE_CACHE:
            # Switch model to eval and move to device
            base_model = base_model.eval().to(config.device)
            print("Benchmarking eager baseline inference.....")
            eager_baseline_time = model_inference_time_in_ms(
                model=base_model, input_data=input_data
            )

            print("Benchmarking compile baseline inference.....")
            base_model = torch.compile(
                base_model, mode=config.torch_compile_mode, fullgraph=True
            )
            compile_baseline_time = model_inference_time_in_ms(
                model=base_model, input_data=input_data
            )

            # Store uncompiled model, input and baseline time
            _BASELINE_CACHE[cache_key] = (eager_baseline_time, compile_baseline_time)

            result.eager_baseline_inference_time_in_ms = eager_baseline_time
            result.compile_baseline_inference_time_in_ms = compile_baseline_time
        else:
            # Retrieve cached values
            cached_eager_time, cached_compile_time = _BASELINE_CACHE[cache_key]
            result.eager_baseline_inference_time_in_ms = cached_eager_time
            result.compile_baseline_inference_time_in_ms = cached_compile_time

        # At this point, ``base_model`` is an uncompiled model ready for quantization,
        # and ``input_data`` is the corresponding input tensor.  The baseline time
        # has been stored in ``result.baseline_inference_time_in_ms``.

        # Copy base model for quantizing/sparsifying
        m_copy = deepcopy(base_model)

        # Determine quantization/sparsity configuration
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
            m_copy = m_copy.eval().to(config.device)
            quantize_(m_copy, ao_base_config)

        # Store result in model for memory profiling
        m_copy._benchmark_result = result

        # Measure inference time for quantized model
        print("Benchmarking eager quantized model.....")
        result.eager_model_inference_time_in_ms = model_inference_time_in_ms(
            model=m_copy, input_data=input_data
        )

        # Measure inference time for compiled quantized model
        print("Benchmarking quantized model.....")
        m_copy = torch.compile(m_copy, mode=config.torch_compile_mode, fullgraph=True)
        result.compile_model_inference_time_in_ms = model_inference_time_in_ms(
            model=m_copy, input_data=input_data
        )

        # Compute eager speedup relative to baseline
        result.eager_speedup_on_baseline = round(
            result.eager_baseline_inference_time_in_ms
            / result.eager_model_inference_time_in_ms,
            2,
        )
        # Compute compile speedup relative to baseline
        result.compile_speedup_on_baseline = round(
            result.compile_baseline_inference_time_in_ms
            / result.compile_model_inference_time_in_ms,
            2,
        )
        # Compute compile speedup for quantized model relative to eager quantized model
        result.compile_speedup_on_eager = round(
            result.eager_model_inference_time_in_ms
            / result.compile_model_inference_time_in_ms,
            2,
        )

        # Run profiler if enabled
        if config.enable_profiler:
            print("Running profiler...")
            try:
                profiler_json_path = generate_model_profile(
                    model=m_copy,
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
            print("Running memory profiler...")
            try:
                # Create memory profiler directory if it doesn't exist
                memory_profiler_dir = os.path.join(
                    config.output_dir, "memory_profiler/pickle"
                )
                os.makedirs(memory_profiler_dir, exist_ok=True)

                # Save memory profile with .pickle extension
                result.memory_profile_path, result.memory_stats = (
                    generate_memory_profile(
                        model=m_copy,
                        input_data=input_data,
                        profile_file_path=os.path.join(
                            memory_profiler_dir,
                            f"{config._file_name}_memory_profile.pickle",
                        ),
                    )
                )

                if result.memory_profile_path:
                    result.memory_visualization_path = visualize_memory_profile(
                        result.memory_profile_path
                    )
            except ValueError as e:
                if "not enough values to unpack" in str(e):
                    print(
                        "Failed due to existing bugs, re‑run the code to generate memory profile. Please raise an issue if it persists."
                    )
            except Exception as e:
                print(f"Error running memory profiler: {e}")
                import traceback

                traceback.print_exc()

        return result
    except Exception as e:
        print(f"Error in benchmark run: {config.name} with error: {e}")
        return None
