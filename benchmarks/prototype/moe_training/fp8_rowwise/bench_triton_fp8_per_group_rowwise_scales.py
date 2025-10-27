# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    triton_fp8_per_group_colwise_scales,
    triton_fp8_per_group_rowwise_scales,
)
from torchao.prototype.moe_training.utils import (
    generate_jagged_offs,
    torch_to_float8_per_group_rowwise,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple[int]
    n_groups: int


@dataclass(frozen=True)
class ExperimentResult:
    torch_loop_time_us: float
    triton_time_us: float
    triton_transpose_us: float
    torch_mem_bw_gbps: float
    triton_mem_bw_gbps: float
    triton_transpose_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [(16640, 8192)]  # (Mg, N)
    n_groups_list = [1, 16, 64]
    high_precision_dtypes = [torch.bfloat16]
    configs = []
    for input_shape, n_groups, high_precision_dtype in itertools.product(
        input_shapes, n_groups_list, high_precision_dtypes
    ):
        configs.append(
            ExperimentConfig(
                input_shape=input_shape,
                n_groups=n_groups,
                high_precision_dtype=high_precision_dtype,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # define test inputs
    Mg, N = config.input_shape

    # Left operand in grad_weight = grad_output_t @ input
    grad_out = torch.randn(
        Mg,
        N,
        dtype=config.high_precision_dtype,
        device=device,
    )
    grad_out_t = grad_out.transpose(-2, -1)

    # - configure input to be row-major with groups divided along the column dimension,
    #   representing the left operand of grad_weight = grad_output_t @ input
    #   that occurs in the backward pass of the differentiable scaled grouped mm.
    # - the transposed tensor in col-major format with groups along the row dimension,
    #    which represents the right operand.
    n_groups = config.n_groups
    offs = generate_jagged_offs(n_groups, Mg, multiple_of=16)

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    # Bench torch per group rowwise
    torch_to_float8_per_group_rowwise_c = torch.compile(
        torch_to_float8_per_group_rowwise
    )
    warmup(
        torch_to_float8_per_group_rowwise_c,
        grad_out_t,
        offs,
        target_dtype=torch.float8_e4m3fn,
    )
    torch_loop_time_us = benchmark_cuda_function_in_microseconds(
        torch_to_float8_per_group_rowwise_c,
        grad_out_t,
        offs,
        target_dtype=torch.float8_e4m3fn,
    )

    # Bench triton per group rowwise scaling kernel
    warmup(
        triton_fp8_per_group_rowwise_scales,
        grad_out_t,
        offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=True,
    )
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_per_group_rowwise_scales,
        grad_out_t,
        offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=True,
    )

    # Bench method where we compute colwise scales on grad_output (equivalent to rowwise scales on grad_output_t)
    def run_triton_transpose_method(
        grad_out, offs, output_dtype, round_scales_to_power_of_2
    ):
        # Restride input as column major.
        # Note this is the transpose of grad_output_t, which is what we are trying to compute per group rowwise scales for.
        grad_out = grad_out.t().contiguous().t()
        # Compute per group colwise scales, writing to column major format.
        fp8_data, scales = triton_fp8_per_group_colwise_scales(
            grad_out, offs, output_dtype, round_scales_to_power_of_2
        )
        return fp8_data.t(), scales.t()

    run_triton_c = torch.compile(run_triton_transpose_method)
    warmup(
        run_triton_c,
        grad_out,
        offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=True,
    )
    triton_transpose_us = benchmark_cuda_function_in_microseconds(
        run_triton_c,
        grad_out,
        offs,
        output_dtype=torch.float8_e4m3fn,
        round_scales_to_power_of_2=True,
    )

    # Mem bw calculations
    bytes_per_input_el = torch.finfo(config.high_precision_dtype).bits / 8
    num_elements = grad_out_t.numel()

    read_bytes = (
        2 * num_elements * bytes_per_input_el  # read input tensor twice
        + 4 * (n_groups * N)  # read scales tensor once, 4 bytes per fp32 scale
    )
    write_bytes = (
        # 1 byte per output elem in fp8
        num_elements
        +
        # write scales tensor, 4 bytes per fp32 scale (we actually do this write once per blong along the reduction dim using atomics, but this is an approximation)
        4 * (n_groups * N)
    )

    read_write_bytes = read_bytes + write_bytes
    torch_mem_bw_gbps = (read_write_bytes) / (torch_loop_time_us / 1e6) / 1e9
    triton_mem_bw_gbps = (read_write_bytes) / (triton_time_us / 1e6) / 1e9

    # Transpose method has extra reads/writes:
    to_col_major_read_write_bytes = (
        2 * num_elements * bytes_per_input_el
    )  # read once, write once when converting input to column major
    triton_transpose_mem_bw_gbps = (
        (read_write_bytes + to_col_major_read_write_bytes)
        / (triton_transpose_us / 1e6)
        / 1e9
    )
    return ExperimentResult(
        torch_loop_time_us=torch_loop_time_us,
        triton_time_us=triton_time_us,
        triton_transpose_us=triton_transpose_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
        triton_transpose_mem_bw_gbps=triton_transpose_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "Mg,N",
        "n_groups",
        "torch_loop_time_us",
        "triton_time_us",
        "triton_transpose_us",
        "torch_mem_bw_gbps",
        "triton_mem_bw_gbps",
        "triton_transpose_mem_bw_gbps",
        "triton_speedup",
        "triton_transpose_speedup",
    ]
    rows = []
    for experiment in experiments:
        input_shape = (
            f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]})"
        )
        rows.append(
            [
                input_shape,
                experiment.config.n_groups,
                experiment.result.torch_loop_time_us,
                experiment.result.triton_time_us,
                experiment.result.triton_transpose_us,
                round(experiment.result.torch_mem_bw_gbps, 3),
                round(experiment.result.triton_mem_bw_gbps, 3),
                round(experiment.result.triton_transpose_mem_bw_gbps, 3),
                f"{experiment.result.torch_loop_time_us / experiment.result.triton_time_us:.2f}x",
                f"{experiment.result.torch_loop_time_us / experiment.result.triton_transpose_us:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
