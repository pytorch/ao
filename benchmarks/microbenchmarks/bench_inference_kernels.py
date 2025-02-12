import torch
import argparse
from copy import deepcopy
from typing import Callable, List, Optional
from benchmarks.microbenchmarks.utils import (
    ToyLinearModel,
    get_default_device,
)
from torchao import quantization
from torchao.quantization import (
    int4_weight_only,
    int8_weight_only,
    float8_weight_only,
    quantize_,
)
import re


def parse_quantization_arg(quantization_input: List[str]):
    # Define regex patterns for quantization techniques
    patterns = {
        r"^int4wo-(\d+)(-hqq)?$": int4_weight_only,
        r"^int8wo$": int8_weight_only,
        r"^float8wo$": float8_weight_only,
        # Add other patterns and corresponding functions here
    }
    for quant_technique in quantization_input:
        # Iterate over patterns and functions
        for pattern, func in patterns.items():
            match = re.match(pattern, quant_technique)
            if match:
                # Extract parameters from the match
                groups = match.groups()
                if func == int4_weight_only:
                    kwargs = {
                        "group_size": int(groups[0]),
                        "use_hqq": bool(groups[1]),
                    }
                    yield func, kwargs
                elif func == int8_weight_only:
                    yield func
                elif func == float8_weight_only:
                    yield func
                # TODO: Add other function calls with parameters here

                # raise ValueError(f"Unsupported quantization technique: {quant_technique}")


def main(
    quant_func: Callable,
    quant_kwargs: Optional[dict],
    # matrix_sizes,
    # m,
    # k,
    # n,
    # precision,
    device=get_default_device(),
    compile: bool = False,
) -> None:
    # TODO: Add more model types here
    base_model = ToyLinearModel().eval().to(device)

    # Use quantize_ to apply each quantization function to the model
    print(f"Running benchmark for {quant_func} {kwargs} quantization")
    m_copy = deepcopy(base_model).to(device)
    quantize_(m_copy, quant_func(**kwargs))
    print(f"Quantized model: {m_copy}")

    if compile:
        print("Compiling model...")
        m_copy = torch.compile(m_copy)

    # TODO: Run benchmark on the quantized model
    # Will add benchmarking code here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks"
    )

    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        nargs='+',
        help=(
            "Pass all the quantization techniques for benchmarking: "
            + "int8dq, int8wo, fp6, int4wo-<groupsize>, int4wo-<groupsize>-hqq, autoquant, "
            + "autoquant-int4, autoquant-gemlite-int4, autoquant-float8, autoquant-sparse, autoquant-all, uintx-<nbits>-<groupsize>, uintx-<nbits>-<groupsize>-hqq, sparse-marlin, spinquant, "
            + "embed-int8wo, marlin_qqq, gemlite-<pack_bitwidth>-<nbits>-<groupsize>, int8adq-int4w-symm"
        ),
    )

    # parser.add_argument(
    #     "--matrix_sizes",
    #     type=str,
    #     nargs='+',
    #     help=(
    #         "Pass all the matrix sizes for benchmarking."
    #     ),
    # )

    # parser.add_argument(
    #     "-m",
    #     type=int,
    #     help="M dimension of the matrix",
    # )

    # parser.add_argument(
    #     "-k",
    #     type=int,
    #     help="M dimension of the matrix",
    # )

    # parser.add_argument(
    #     "-n",
    #     type=int,
    #     help="M dimension of the matrix",
    # )

    # parser.add_argument(
    #     "--precision",
    #     type=str,
    #     choices=["float32", "float16", "bfloat16"],
    # )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model",
    )

    args = parser.parse_args()
    print(args)

    # Process arguments
    quantization_funcs = list(parse_quantization_arg(args.quantization))

    # Run benchmarks
    for func, kwargs in quantization_funcs:
        main(
            quant_func=func,
            quant_kwargs=kwargs,
        )
