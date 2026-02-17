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
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.grouped_mm.kernels import (
    torch_to_blocked_2d_K_groups,
    triton_mx_block_rearrange_2d_K_groups,
)

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
    block_size = 32
    # DeepSeekV3 671b shapes.
    # total tokens and groups are along the contracting dimension
    input_shapes = [
        (7168, 131072 // block_size),
        (2048, 131072 // block_size),
        (7168, 65536 // block_size),
        (2048, 65536 // block_size),
        (7168, 32768 // block_size),
        (2048, 32768 // block_size),
    ]
    num_groups_list = [4, 8]

    configs = []
    for shape, groups in itertools.product(
        input_shapes,
        num_groups_list,
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

    M, scale_cols = input_shape
    block_size = 32
    # total_K is the total number of elements along K dimension
    total_K = scale_cols * block_size

    # Generate group end offsets along total_K, then divide by block_size to get scale group end offsets
    input_group_offsets = generate_jagged_offs(
        num_groups, total_K, multiple_of=block_size
    )
    scale_group_offsets = input_group_offsets // block_size

    # bench torch (torch.compile hangson this, can't use)
    torch_out_scales, torch_group_offs = torch_to_blocked_2d_K_groups(
        input_tensor,
        scale_group_offsets,
        block_size=block_size,
    )
    torch_time_us = benchmark_cuda_function_in_microseconds(
        torch_to_blocked_2d_K_groups,
        input_tensor,
        scale_group_offsets,
        block_size=block_size,
    )

    # bench triton
    triton_out_scales = triton_mx_block_rearrange_2d_K_groups(
        input_tensor,
        scale_group_offsets,
    )
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_mx_block_rearrange_2d_K_groups,
        input_tensor,
        scale_group_offsets,
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
        "num_groups",
        "torch_time_us",
        "triton_time_us",
        "triton_speedup",
        "torch_mem_bw_gbps",
        "triton_mem_bw_gbps",
    ]
    rows = []
    for experiment in experiments:
        input_shape = (
            f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]})"
        )
        rows.append(
            [
                input_shape,
                experiment.config.num_groups,
                f"{experiment.result.torch_time_us:.2f}",
                f"{experiment.result.triton_time_us:.2f}",
                f"{experiment.result.torch_time_us / experiment.result.triton_time_us:.2f}x",
                f"{experiment.result.torch_mem_bw_gbps:.2f}",
                f"{experiment.result.triton_mem_bw_gbps:.2f}",
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
