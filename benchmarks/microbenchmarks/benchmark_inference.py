"""
Inference benchmark runner

This script runs inference benchmarks and generates a micro-benchmarking report for it.
- run() function is the main entry point for running inference benchmarks.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch

from utils import (
    BenchmarkConfig,
    benchmark_model_inference_in_microseconds,
    clean_caches,
    create_model_and_input,
    quantization_string_to_quantization_config,
)
from torchao.quantization import quantize_


def run(config: BenchmarkConfig) -> Dict[str, float]:
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
    quantization_config = quantization_string_to_quantization_config(
        config.quantization,
        config.sparsity,
        high_precision_dtype=config.high_precision_dtype
    )
    if quantization_config:
        quantize_(m_copy, quantization_config)
    if config.compile:
        print("Compiling model....")
        m_copy = torch.compile(m_copy, mode=config.compile_mode, fullgraph=True)

    # Run benchmarks
    result = {**config.to_dict()}

    # Benchmark time to run an inference call for quantized model
    model_time = benchmark_model_inference_in_microseconds(
        model=m_copy, input_data=input_data
    )
    result["benchmark_model_inference_in_microseconds"] = model_time

    # TODO: Benchmark time using profiler
    # Profile dtype model evaluation
    # prof_dtype = benchmark_model_op_with_profiler_in_microseconds(m_copy, input_data, quantized_dtype)
    # prof_dtype.export_chrome_trace(f"{quantization}_model_{input_data[0].size()[0]}.json")  # Save profiling details

    # TODO: Benchmark gemm time using cuda graph
    # gemm_time = benchmark_torch_function_in_microseconds(gemm_op, *args, **kwargs)

    # TODO: Benchmark op with cuda graph
    # time = benchmark_op_with_cuda_graph(op, args)

    return result
