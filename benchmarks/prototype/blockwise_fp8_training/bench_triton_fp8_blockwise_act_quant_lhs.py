# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Tuple

import torch
from tabulate import tabulate
from tqdm import tqdm

# Assuming these imports based on the kernel location
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.blockwise_fp8_training.kernels import (
    torch_blockwise_scale_act_quant_lhs,
    triton_fp8_blockwise_act_quant_lhs,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: Tuple[int, int]  # (M, K)
    block_size: int


@dataclass(frozen=True)
class ExperimentResult:
    # time
    naive_us: float
    triton_us: float
    # mem bw
    naive_gbps: float
    triton_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    """
    Test configurations for typical transformer activation shapes.
    Format: (batch_size * seq_len, hidden_dim)
    """
    # Llama-style shapes: various batch*seq_len sizes with typical hidden dims
    input_shapes = [
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (8192, 4096),
    ]

    configs = []
    block_sizes = [128]  # Standard block size for FP8

    for shape in input_shapes:
        for block_size in block_sizes:
            configs.append(
                ExperimentConfig(
                    input_shape=shape,
                    block_size=block_size,
                )
            )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M, K = config.input_shape
    block_size = config.block_size

    def verify_outputs(
        y_naive: torch.Tensor,
        s_naive: torch.Tensor,
        y_triton: torch.Tensor,
        s_triton: torch.Tensor,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        """Verify that Triton and naive implementations produce similar results."""

        # Convert FP8 back to float for comparison
        y_naive_float = y_naive.to(torch.float32)
        y_triton_float = y_triton.to(torch.float32)

        # Check quantized values are close
        try:
            torch.testing.assert_close(
                y_naive_float,
                y_triton_float,
                rtol=rtol,
                atol=atol,
                msg="Quantized values differ between naive and Triton implementations",
            )
        except AssertionError as e:
            max_diff = (y_naive_float - y_triton_float).abs().max().item()
            print(f"WARNING: Scales differ! Max diff: {max_diff}")
            print(
                f"  Naive scale range: [{y_naive_float.min():.6f}, {y_naive_float.max():.6f}]"
            )
            print(
                f"  Triton scale range: [{y_triton_float.min():.6f}, {y_triton_float.max():.6f}]"
            )
            print(f"  Error details: {e}")

        try:
            torch.testing.assert_close(
                s_naive,
                s_triton,
                rtol=rtol,
                atol=atol,
                msg="Scales differ between naive and Triton implementations",
            )
        except AssertionError as e:
            max_diff = (s_naive - s_triton).abs().max().item()
            print(f"WARNING: Scales differ! Max diff: {max_diff}")
            print(f"  Naive scale range: [{s_naive.min():.6f}, {s_naive.max():.6f}]")
            print(f"  Triton scale range: [{s_triton.min():.6f}, {s_triton.max():.6f}]")
            print(f"  Error details: {e}")

    input_tensor = torch.randn(
        M,
        K,
        dtype=torch.bfloat16,
        device=device,
    )

    # Benchmark naive implementation
    # naive_impl_c = torch.compile(torch_blockwise_scale_act_quant_lhs)
    y_naive, s_naive = torch_blockwise_scale_act_quant_lhs(input_tensor, block_size)
    naive_time_us = benchmark_cuda_function_in_microseconds(
        torch_blockwise_scale_act_quant_lhs,
        input_tensor,
        block_size,
    )

    # Benchmark Triton implementation
    triton_impl_c = torch.compile(triton_fp8_blockwise_act_quant_lhs)
    y_triton, s_triton = triton_impl_c(input_tensor, block_size)
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_impl_c,
        input_tensor,
        block_size,
    )

    # Verify correctness (optional, can comment out for pure benchmarking)
    verify_outputs(y_naive, s_naive, y_triton, s_triton)

    # Memory bandwidth calculations
    bytes_per_input_el = torch.finfo(torch.float32).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = 4  # float32

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = (
        y_triton.numel() * bytes_per_output_el + s_triton.numel() * bytes_per_scale_el
    )

    naive_gbps = ((read_bytes + write_bytes) / 1e9) / (naive_time_us / 1e6)
    triton_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        naive_us=naive_time_us,
        triton_us=triton_time_us,
        naive_gbps=naive_gbps,
        triton_gbps=triton_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape (M, K)",
        "block_size",
        "naive_us",
        "triton_us",
        "speedup",
        "naive_gbps",
        "triton_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.naive_us / experiment.result.triton_us
        rows.append(
            [
                f"{experiment.config.input_shape[0]}x{experiment.config.input_shape[1]}",
                experiment.config.block_size,
                f"{experiment.result.naive_us:.2f}",
                f"{experiment.result.triton_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.naive_gbps:.1f}",
                f"{experiment.result.triton_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []

    print(f"Running {len(configs)} benchmark configurations...\n")

    for config in tqdm(configs, desc="Benchmarking"):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80 + "\n")
    print_results(results)


if __name__ == "__main__":
    main()
