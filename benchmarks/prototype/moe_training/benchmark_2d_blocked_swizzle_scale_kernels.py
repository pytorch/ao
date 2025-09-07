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

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8_blocked_scales import (
    compute_per_group_blocked_scale_offsets,
    torch_to_blocked_per_group_2d,
    triton_mx_block_rearrange_per_group_2d,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]
    num_groups: int


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
        (16640, 5120 // block_size),
    ]
    num_groups = [16]
    configs = []
    for shape, groups in itertools.product(
        input_shapes,
        num_groups,
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                num_groups=groups,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    input_shape, num_groups = config.input_shape, config.num_groups
    input_tensor = torch.randint(
        low=0,
        high=256,
        size=input_shape,
        dtype=torch.uint8,
        device=device,
    )

    Mg, K = input_shape
    input_group_offsets = generate_jagged_offs(num_groups, Mg, multiple_of=32)

    # bench torch
    compiled_run_torch = torch.compile(torch_to_blocked_per_group_2d)
    torch_out_scales, torch_group_offs = compiled_run_torch(
        input_tensor, input_group_offsets, K
    )
    torch_time_us = benchmark_cuda_function_in_microseconds(
        compiled_run_torch,
        input_tensor,
        input_group_offsets,
        Mg,
        K,
    )

    # bench triton
    _, output_group_offsets = compute_per_group_blocked_scale_offsets(
        input_group_offsets
    )
    triton_out_scales = triton_mx_block_rearrange_per_group_2d(
        input_tensor,
        input_group_offsets,
        output_group_offsets,
    )
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_mx_block_rearrange_per_group_2d,
        input_tensor,
        input_group_offsets,
        output_group_offsets,
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
        input_shape = (
            f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]})"
        )
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
