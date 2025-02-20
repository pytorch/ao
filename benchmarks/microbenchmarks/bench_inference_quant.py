"""Script to compare multiple quantization techniques for inference, for a particular matrix shape, and model type"""

import argparse
from copy import deepcopy
from typing import List

import torch
from utils import (
    benchmark_model_inference_in_microseconds,
    benchmark_model_op_with_profiler_in_microseconds,
    create_model_and_input,
    get_default_device,
    quantize_model,
    clean_caches,
)


def run(
    quantization: str,
    m,
    k,
    n,
    precision,
    model_type: str = "linear",
    compile: bool = False,
    device=get_default_device(),
    output_dir: str = "benchmarks/microbenchmarks/results/",
) -> None:
    # TODO: Add more model types here
    clean_caches()
    base_model, input_data = create_model_and_input(
        model_type, m, k, n,
        dtype=precision,
        device=device,
    )
    print(f"Starting benchmarking for model: {base_model.__class__.__name__} for quantization: {quantization}")
    # Use quantize_ to apply each quantization function to the model
    m_copy = deepcopy(base_model).eval().to(device)
    m_copy = quantize_model(m_copy, quantization)
    # quantized_dtype = .....

    if compile:
        print("Compiling model....")
        m_copy = torch.compile(m_copy, mode=compile, fullgraph=True)

    # Run benchmarks
    # 1. Benchmark time to run an inference call for quantized model
    model_time = benchmark_model_inference_in_microseconds(model=m_copy, input_data=input_data)
    print(f"Time to run a {base_model.__class__.__name__}: {model_time * 1e6:.2f} microseconds quantized with {quantization}")

    # 2. Benchmark time using profiler
    # Profile dtype model evaluation
    # prof_dtype = benchmark_model_op_with_profiler_in_microseconds(m_copy, input_data, quantized_dtype)
    # prof_dtype.export_chrome_trace(f"{quantization}_model_{input_data[0].size()[0]}.json")  # Save profiling details

    # 3. Benchmark gemm time using cuda graph
    # gemm_time = benchmark_torch_function_in_microseconds(gemm_op, *args, **kwargs)

    # 4. Benchmark op with cuda graph
    # time = benchmark_op_with_cuda_graph(op, args)

    # Last: Create csv file with all the results
    # generate_csv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run microbenchmarks")

    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        help=(
            "Pass the quantization technique for benchmarking: "
            + "int8wo, int4wo-<groupsize>, int4wo-<groupsize>-hqq, float8wo"
        ),
    )

    parser.add_argument(
        "-m",
        type=int,
        help="M dimension of the matrix",
    )

    parser.add_argument(
        "-k",
        type=int,
        help="M dimension of the matrix",
    )

    parser.add_argument(
        "-n",
        type=int,
        help="M dimension of the matrix",
    )

    parser.add_argument(
        "--precision",
        type=lambda x: getattr(torch, x.split(".")[-1]),
        default=torch.bfloat16,
        help="dtype precision to use",
    )

    parser.add_argument(
        "--compile",
        type=str,
        nargs='?',
        const="default",
        default=None,
        help="Whether to compile the model and optionally specify compile mode (default: max-autotune)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmarks/microbenchmarks/results/",
        help="Output directory to save results",
    )

    args = parser.parse_args()
    print(args)

    # Run benchmarks
    run(
        quantization=args.quantization,
        m=args.m,
        k=args.k,
        n=args.n,
        precision=args.precision,
        compile=args.compile,
        device=args.device,
        output_dir=args.output_dir,
    )
