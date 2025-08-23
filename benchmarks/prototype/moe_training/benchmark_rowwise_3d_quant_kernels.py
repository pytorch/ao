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

from torchao.prototype.moe_training.kernels.float8_rowwise import (
    triton_fp8_rowwise_3d_transpose_rhs,
    triton_fp8_rowwise_3d_transpose_rhs_fused_reduction,
)
from torchao.prototype.moe_training.utils import (
    torch_to_3d_rowwise_float8_transpose_rhs,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple[int]
    power_of_2_scales: bool


@dataclass(frozen=True)
class ExperimentResult:
    torch_time_us: float
    triton_atomic_time_us: float
    triton_reduction_time_us: float
    torch_mem_bw_gbps: float
    triton_atomic_mem_bw_gbps: float
    triton_reduction_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes (E, N, K)
    input_shapes = [
        (1, 8192, 5120),  # w1, w3
        (1, 5120, 8192),  # w2
        (16, 8192, 5120),  # w1, w3
        (16, 5120, 8192),  # w2
        (128, 8192, 5120),  # w1, w3
        (128, 5120, 8192),  # w2
    ]
    high_precision_dtypes = [torch.bfloat16]
    power_of_2_scales = [True]
    configs = []
    for input_shape, high_precision_dtype, power_of_2_scale in itertools.product(
        input_shapes, high_precision_dtypes, power_of_2_scales
    ):
        configs.append(
            ExperimentConfig(
                input_shape=input_shape,
                high_precision_dtype=high_precision_dtype,
                power_of_2_scales=power_of_2_scale,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # Expert weights will be passed in transposed and column major in practice
    input_tensor = torch.randn(
        *config.input_shape,
        dtype=config.high_precision_dtype,
        device=device,
    ).transpose(-2, -1)

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    def run_torch(input_tensor: torch.Tensor):
        out = torch_to_3d_rowwise_float8_transpose_rhs(
            input_tensor,
            target_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=config.power_of_2_scales,
        )
        return out

    def run_triton_atomic(input_tensor: torch.Tensor):
        out = triton_fp8_rowwise_3d_transpose_rhs(
            input_tensor,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=config.power_of_2_scales,
        )
        return out

    def run_triton_reduction(input_tensor: torch.Tensor):
        out = triton_fp8_rowwise_3d_transpose_rhs_fused_reduction(
            input_tensor,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=config.power_of_2_scales,
        )
        return out

    # bench torch
    compiled_run_torch = torch.compile(run_torch)
    warmup(run_torch, input_tensor)
    torch_time_us = benchmark_cuda_function_in_microseconds(
        compiled_run_torch,
        input_tensor,
    )

    # bench triton atomic method
    run_triton_atomic_c = torch.compile(run_triton_atomic)
    warmup(run_triton_atomic_c, input_tensor)
    triton_atomic_time_us = benchmark_cuda_function_in_microseconds(
        run_triton_atomic_c,
        input_tensor,
    )

    # bench triton reduction method
    run_triton_reduction_c = torch.compile(run_triton_reduction)
    warmup(run_triton_reduction_c, input_tensor)
    triton_reduction_time_us = benchmark_cuda_function_in_microseconds(
        run_triton_reduction_c,
        input_tensor,
    )

    # mem bw calculations - excluding scales to simplify calculation
    # but still get an accurate estimate.
    bytes_per_input_el = torch.finfo(config.high_precision_dtype).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    num_elements = input_tensor.numel()

    read_bytes = num_elements * bytes_per_input_el
    write_bytes = num_elements * bytes_per_output_el

    # Both torch.compile codegen and the triton kernel read the input tensor twice
    # (once for scale calculations, once for scaling + casting).
    torch_mem_bw_gbps = ((read_bytes * 2 + write_bytes) / 1e9) / (torch_time_us / 1e6)
    triton_atomic_mem_bw_gbps = ((read_bytes * 2 + write_bytes) / 1e9) / (
        triton_atomic_time_us / 1e6
    )
    triton_reduction_mem_bw_gbps = ((read_bytes * 2 + write_bytes) / 1e9) / (
        triton_reduction_time_us / 1e6
    )

    return ExperimentResult(
        torch_time_us=torch_time_us,
        triton_atomic_time_us=triton_atomic_time_us,
        triton_reduction_time_us=triton_reduction_time_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        triton_atomic_mem_bw_gbps=triton_atomic_mem_bw_gbps,
        triton_reduction_mem_bw_gbps=triton_reduction_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "power_of_2_scales",
        "torch_time_us",
        "triton_atomic_time_us",
        "triton_reduction_time_us",
        "torch_mem_bw_gbps",
        "triton_atomic_mem_bw_gbps",
        "triton_reduction_mem_bw_gbps",
        "triton_atomic_speedup",
        "triton_reduction_speedup",
    ]
    rows = []
    for experiment in experiments:
        input_shape = f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1], experiment.config.input_shape[2]})"
        rows.append(
            [
                input_shape,
                experiment.config.power_of_2_scales,
                experiment.result.torch_time_us,
                experiment.result.triton_atomic_time_us,
                experiment.result.triton_reduction_time_us,
                round(experiment.result.torch_mem_bw_gbps, 3),
                round(experiment.result.triton_atomic_mem_bw_gbps, 3),
                round(experiment.result.triton_reduction_mem_bw_gbps, 3),
                f"{experiment.result.torch_time_us / experiment.result.triton_atomic_time_us:.2f}x",
                f"{experiment.result.torch_time_us / experiment.result.triton_reduction_time_us:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


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
