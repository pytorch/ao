# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import argparse
import itertools
import time
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import profile_fn
from torchao.prototype.moe_training.kernels.mxfp8 import (
    torch_pad_token_groups,
    triton_pad_token_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    dim: int
    num_groups: int
    alignment_size: int


@dataclass(frozen=True)
class ExperimentResult:
    torch_uncompiled_time_us: float
    triton_time_us: float
    torch_mem_bw_gbps: float
    triton_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Various token group sizes and dimensions
    num_tokens_list = [32768, 131072]
    dim_list = [1024, 7168]
    num_groups_list = [8]
    alignment_size_list = [32]

    configs = []
    for num_tokens, dim, num_groups, alignment_size in itertools.product(
        num_tokens_list, dim_list, num_groups_list, alignment_size_list
    ):
        configs.append(
            ExperimentConfig(
                num_tokens=num_tokens,
                dim=dim,
                num_groups=num_groups,
                alignment_size=alignment_size,
            )
        )
    return configs


def benchmark_host_side_in_microseconds(fn, *args, num_iters=100, **kwargs):
    """
    Benchmark using host-side timing to capture device-to-host syncs.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return ((end - start) / num_iters) * 1e6  # Convert to microseconds


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    num_tokens, dim, num_groups, alignment_size = (
        config.num_tokens,
        config.dim,
        config.num_groups,
        config.alignment_size,
    )

    # Create input activations
    inputs = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device=device)

    # Create wrapper functions that generate fresh offsets on each call
    def torch_uncompiled_with_offsets():
        group_offsets = generate_jagged_offs(
            num_groups, num_tokens, multiple_of=1, device=device
        )
        return torch_pad_token_groups(inputs, group_offsets, alignment_size)

    def triton_with_offsets():
        group_offsets = generate_jagged_offs(
            num_groups, num_tokens, multiple_of=1, device=device
        )
        return triton_pad_token_groups(inputs, group_offsets, alignment_size)

    def warmup(fn):
        for _ in range(5):
            fn()

    # bench torch uncompiled (use host-side timing to capture .tolist() sync)
    warmup(torch_uncompiled_with_offsets)
    torch_uncompiled_time_us = benchmark_host_side_in_microseconds(
        torch_uncompiled_with_offsets
    )
    if args.profile:
        group_offsets = generate_jagged_offs(
            num_groups, num_tokens, multiple_of=1, device=device
        )
        profile_fn(
            torch_pad_token_groups,
            inputs,
            group_offsets,
            alignment_size,
            profile_name="torch_pad_token_groups_uncompiled",
        )

    # bench triton (use host-side timing for consistency)
    warmup(triton_with_offsets)
    triton_time_us = benchmark_host_side_in_microseconds(triton_with_offsets)
    if args.profile:
        group_offsets = generate_jagged_offs(
            num_groups, num_tokens, multiple_of=1, device=device
        )
        profile_fn(
            triton_pad_token_groups,
            inputs,
            group_offsets,
            alignment_size,
            profile_name="triton_pad_token_groups",
        )

    # mem bw calculations - run once to get output sizes
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    )
    torch_padded_tokens, torch_padded_offsets = torch_pad_token_groups(
        inputs, group_offsets, alignment_size
    )

    bytes_per_el = torch.finfo(torch.bfloat16).bits / 8

    # Read all input tokens + group offsets
    read_bytes = inputs.numel() * bytes_per_el + group_offsets.numel() * 4  # int32

    # Write padded tokens + padded offsets
    write_bytes = (
        torch_padded_tokens.numel() * bytes_per_el + torch_padded_offsets.numel() * 4
    )

    torch_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (
        torch_uncompiled_time_us / 1e6
    )
    triton_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        torch_uncompiled_time_us=torch_uncompiled_time_us,
        triton_time_us=triton_time_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "num_tokens",
        "dim",
        "num_groups",
        "torch_uncompiled_us",
        "triton_us",
        "torch_mem_bw_gbps",
        "triton_mem_bw_gbps",
        "triton_vs_torch",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.num_tokens,
                experiment.config.dim,
                experiment.config.num_groups,
                f"{experiment.result.torch_uncompiled_time_us:.2f}",
                f"{experiment.result.triton_time_us:.2f}",
                f"{experiment.result.torch_mem_bw_gbps:.2f}",
                f"{experiment.result.triton_mem_bw_gbps:.2f}",
                f"{experiment.result.torch_uncompiled_time_us / experiment.result.triton_time_us:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers))


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling with PyTorch profiler"
    )
    args = parser.parse_args()
    main(args)
