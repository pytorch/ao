# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    torch_to_blocked_per_group_3d,
    triton_mx_block_rearrange_per_group_3d,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    torch_time_us: float
    triton_time_us: float
    torch_mem_bw_gbps: float
    triton_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes. Input activations are scaled along K dim.
    block_size = 32
    input_shapes = [
        # w1, w3 scaled along K (fwd)
        (1, 8192, 5120 // block_size),
        (2, 8192, 5120 // block_size),
        (4, 8192, 5120 // block_size),
        (8, 8192, 5120 // block_size),
        (16, 8192, 5120 // block_size),
        # w2 scaled along K (fwd)
        (1, 5120, 8192 // block_size),
        (2, 5120, 8192 // block_size),
        (4, 5120, 8192 // block_size),
        (8, 5120, 8192 // block_size),
        (16, 5120, 8192 // block_size),
    ]
    configs = []
    for shape in input_shapes:
        configs.append(
            ExperimentConfig(
                input_shape=shape,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    input_tensor = torch.randint(
        low=0,
        high=256,
        size=config.input_shape,
        dtype=torch.uint8,
        device=device,
    )

    def warmup(fn, *args, **kwargs):
        for _ in range(5):
            fn(*args, **kwargs)

    E, N, K = config.input_shape

    # bench torch
    compiled_run_torch = torch.compile(torch_to_blocked_per_group_3d)
    warmup(compiled_run_torch, input_tensor)
    torch_time_us = benchmark_cuda_function_in_microseconds(
        compiled_run_torch,
        input_tensor,
    )

    # bench triton
    triton_out_scales = triton_mx_block_rearrange_per_group_3d(input_tensor)
    warmup(triton_mx_block_rearrange_per_group_3d, input_tensor)
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_mx_block_rearrange_per_group_3d,
        input_tensor,
    )

    # mem bw calculations
    bytes_per_input_el = torch.finfo(torch.float8_e8m0fnu).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = triton_out_scales.numel() * bytes_per_output_el

    torch_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (torch_time_us / 1e6)
    triton_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        torch_time_us=torch_time_us,
        triton_time_us=triton_time_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "torch_time_us",
        "triton_time_us",
        "torch_mem_bw_gbps",
        "triton_mem_bw_gbps",
        "triton_speedup",
    ]
    rows = []
    for experiment in experiments:
        input_shape = f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]}, {experiment.config.input_shape[2]})"
        rows.append(
            [
                input_shape,
                experiment.result.torch_time_us,
                experiment.result.triton_time_us,
                round(experiment.result.torch_mem_bw_gbps, 3),
                round(experiment.result.triton_mem_bw_gbps, 3),
                f"{experiment.result.torch_time_us / experiment.result.triton_time_us:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers))


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
