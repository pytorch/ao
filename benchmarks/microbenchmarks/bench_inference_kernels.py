"""Script to compare multiple quantization techniques for inference, for a particular matrix shape, and model type"""

import argparse
from copy import deepcopy
from typing import List

import torch
from utils import (
    ToyLinearModel,
    benchmark_model_run,
    create_model_and_input,
    get_default_device,
    quantize_model,
)


def main(
    quantizations: List[str],
    m,
    k,
    n,
    precision,
    model_type: str = "linear",
    compile: bool = False,
    device=get_default_device(),
) -> None:
    # TODO: Add more model types here
    base_model, input_data = create_model_and_input(model_type, m, k, n, precision, device)
    # base_model = ToyLinearModel(k, n, precision).eval().to(device)

    for quant in quantizations:
        # Use quantize_ to apply each quantization function to the model
        m_copy = deepcopy(base_model).eval().to(device)
        m_copy = quantize_model(m_copy, quant)
        print(f"Quantized model: {m_copy}")

        if compile:
            print("Compiling model...")
            m_copy = torch.compile(m_copy)

        # Run benchmarks
        # 1. Benchmark time to run an inference call for quantized model
        model_time = benchmark_model_run(m_copy, input_data)
        print(f"Time to run a {model_type}: {model_time}")

        # 2. Benchmark memory usage of quantized model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run microbenchmarks")

    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        nargs="+",
        help=(
            "Pass all the quantization techniques for benchmarking: "
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
        action="store_true",
        help="Whether to compile the model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on",
    )

    args = parser.parse_args()
    print(args)

    # Run benchmarks
    main(
        quantizations=args.quantization,
        m=args.m,
        k=args.k,
        n=args.n,
        precision=args.precision,
        compile=args.compile,
        device=args.device,
    )
