# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import argparse
import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds, profile_fn
from torchao.prototype.moe_training.kernels.mxfp8 import (
    _mxfp8_cuda_kernels_available,
    fused_unpad_token_groups_cuda,
    torch_pad_token_groups,
    torch_unpad_token_groups,
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
    torch_eager_time_us: float
    cuda_time_us: float
    torch_mem_bw_gbps: float
    cuda_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Various token group sizes and dimensions
    num_tokens_list = [16384]
    dim_list = [1536, 2048, 5120, 7168]
    num_groups_list = [1, 4, 8, 16]
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


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    num_tokens, dim, num_groups, alignment_size = (
        config.num_tokens,
        config.dim,
        config.num_groups,
        config.alignment_size,
    )

    # Create inputs and pad them first
    inputs = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device=device)
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    )

    # Pad the inputs to get padded tensors for unpad benchmark
    padded_inputs, padded_group_start_offsets, padded_group_end_offsets = (
        torch_pad_token_groups(inputs, group_offsets, alignment_size)
    )

    def torch_eager_with_offsets():
        return torch_unpad_token_groups(
            padded_inputs,
            group_offsets,
            padded_group_start_offsets,
            num_tokens,
            alignment_size,
        )

    def warmup(fn):
        for _ in range(5):
            fn()

    # bench torch eager (includes buffer allocation overhead)
    warmup(torch_eager_with_offsets)
    torch_eager_time_us = benchmark_cuda_function_in_microseconds(
        torch_eager_with_offsets
    )
    if args.profile:
        profile_fn(
            torch_unpad_token_groups,
            padded_inputs,
            group_offsets,
            padded_group_start_offsets,
            alignment_size,
            profile_name="torch_unpad_token_groups_eager",
        )

    # bench CUDA kernel if available
    if _mxfp8_cuda_kernels_available:

        def cuda_with_offsets():
            return fused_unpad_token_groups_cuda(
                padded_inputs,
                group_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size,
            )

        warmup(cuda_with_offsets)
        cuda_time_us = benchmark_cuda_function_in_microseconds(cuda_with_offsets)
        if args.profile:
            profile_fn(
                fused_unpad_token_groups_cuda,
                padded_inputs,
                group_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size,
                profile_name="fused_unpad_token_groups_cuda",
            )
    else:
        cuda_time_us = float("inf")  # Not available

    # mem bw calculations
    bytes_per_el = torch.finfo(torch.bfloat16).bits / 8

    read_bytes = (
        padded_inputs.numel() * bytes_per_el  # Read padded input tokens
        + group_offsets.numel() * 4  # Read group offsets (int32)
        + padded_group_start_offsets.numel() * 4  # Read padded start offsets (int32)
    )

    write_bytes = (
        inputs.numel() * bytes_per_el  # Write unpadded data
    )

    total_bytes = read_bytes + write_bytes

    torch_mem_bw_gbps = (total_bytes / 1e9) / (torch_eager_time_us / 1e6)

    if _mxfp8_cuda_kernels_available and cuda_time_us != float("inf"):
        cuda_mem_bw_gbps = (total_bytes / 1e9) / (cuda_time_us / 1e6)
    else:
        cuda_mem_bw_gbps = 0.0

    return ExperimentResult(
        torch_eager_time_us=torch_eager_time_us,
        cuda_time_us=cuda_time_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        cuda_mem_bw_gbps=cuda_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "num_tokens",
        "dim",
        "num_groups",
        "torch_us",
        "cuda_us",
        "torch_mem_bw_gbps",
        "cuda_mem_bw_gbps",
        "cuda_vs_torch",
    ]
    rows = []
    for experiment in experiments:
        cuda_time = experiment.result.cuda_time_us
        cuda_vs_torch = (
            f"{experiment.result.torch_eager_time_us / cuda_time:.2f}x"
            if cuda_time != float("inf") and cuda_time > 0
            else "N/A"
        )
        cuda_bw_str = (
            f"{experiment.result.cuda_mem_bw_gbps:.2f}"
            if experiment.result.cuda_mem_bw_gbps > 0
            else "N/A"
        )

        rows.append(
            [
                experiment.config.num_tokens,
                experiment.config.dim,
                experiment.config.num_groups,
                experiment.result.torch_eager_time_us,
                experiment.result.cuda_time_us,
                f"{experiment.result.torch_mem_bw_gbps:.2f}",
                cuda_bw_str,
                cuda_vs_torch,
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
