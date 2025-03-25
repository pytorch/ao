# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Inference benchmark runner

This script runs inference benchmarks and generates a micro-benchmarking report for it.
- run() function is the main entry point for running inference benchmarks.
"""

from copy import deepcopy
from pathlib import Path

import torch

from benchmarks.microbenchmarks.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    clean_caches,
    create_model_and_input,
    model_inference_time_in_ms,
    string_to_config,
)
from torchao.quantization import quantize_
from torchao.sparsity.sparse_api import sparsify_


def run(config: BenchmarkConfig) -> BenchmarkResult:
    """Run inference benchmarks"""
    clean_caches()  # Clean caches

    # Create output directory if it doesn't exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    base_model, input_data = create_model_and_input(
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
        m_copy = torch.compile(m_copy, mode=config.torch_compile_mode, fullgraph=True)

    # Run benchmarks
    result = BenchmarkResult(config=config)

    # Benchmark time to run an inference call for quantized model
    result.model_inference_time_in_ms = model_inference_time_in_ms(
        model=m_copy, input_data=input_data
    )

    # TODO: Benchmark time using profiler
    # Profile dtype model evaluation
    # prof_dtype = benchmark_model_op_with_profiler_in_microseconds(m_copy, input_data, quantized_dtype)
    # prof_dtype.export_chrome_trace(f"{quantization}_model_{input_data[0].size()[0]}.json")  # Save profiling details

    # TODO: Benchmark gemm time using cuda graph
    # gemm_time = benchmark_torch_function_in_microseconds(gemm_op, *args, **kwargs)

    # TODO: Benchmark op with cuda graph
    # time = benchmark_op_with_cuda_graph(op, args)

    return result
