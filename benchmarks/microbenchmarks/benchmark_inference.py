"""
Inference benchmark runner

This script runs inference benchmarks and generates a micro-benchmarking report for it.
- run() function is the main entry point for running inference benchmarks.
"""
from copy import deepcopy
import json
from pathlib import Path

import torch
from utils import (
    benchmark_model_inference_in_seconds,
    clean_caches,
    create_model_and_input,
    quantize_model,
    BenchmarkConfig,
)

def run(config: BenchmarkConfig) -> None:
    """Run inference benchmarks"""
    clean_caches()  # Clean caches
    
    # Create output directory if it doesn't exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    base_model, input_data = create_model_and_input(
        config.model_type,
        config.m,
        config.k,
        config.n,
        dtype=config.precision,
        device=config.device,
    )
    print(
        f"Starting benchmarking for model: {base_model.__class__.__name__} for quantization: {config.quantization}"
    )
    
    # Use quantize_ to apply each quantization function to the model
    m_copy = deepcopy(base_model).eval().to(config.device)
    m_copy = quantize_model(m_copy, config.quantization)

    if config.compile:
        print("Compiling model....")
        m_copy = torch.compile(m_copy, mode=config.compile, fullgraph=True)

    # Run benchmarks
    results = {}
    
    # Benchmark time to run an inference call for quantized model
    model_time = benchmark_model_inference_in_seconds(
        model=m_copy, input_data=input_data
    )
    results[f"benchmark_model_inference_in_seconds"] = model_time
    print(
        f"Time to run a {base_model.__class__.__name__}: {model_time:.2f} seconds quantized with {config.quantization}"
    )

    # 2. Benchmark time using profiler
    # Profile dtype model evaluation
    # prof_dtype = benchmark_model_op_with_profiler_in_microseconds(m_copy, input_data, quantized_dtype)
    # prof_dtype.export_chrome_trace(f"{quantization}_model_{input_data[0].size()[0]}.json")  # Save profiling details

    # 3. Benchmark gemm time using cuda graph
    # gemm_time = benchmark_torch_function_in_microseconds(gemm_op, *args, **kwargs)

    # 4. Benchmark op with cuda graph
    # time = benchmark_op_with_cuda_graph(op, args)

    return results
