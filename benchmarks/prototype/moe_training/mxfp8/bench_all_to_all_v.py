# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these benchmarks, use the following command:
#
# torchrun --nproc-per-node=8 --local-ranks-filter=0 benchmarks/prototype/moe_training/mxfp8/bench_all_to_all_v.py
#
#######################################################################
import os
import time
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from torch import distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single_autograd,
)
from tqdm import tqdm

from torchao.prototype.moe_training.kernels.mxfp8.comms import (
    mxfp8_on_device_all_to_all_v,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    bf16_us: float
    mxfp8_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # (batch_size, seq_len, dim)
    input_shapes = [
        (8, 8192, 5120),
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
    batch_size, seq_len, dim = config.input_shape
    x = torch.randn(
        (batch_size * seq_len, dim),
        dtype=torch.bfloat16,
        device=device,
    )
    ref_x = x.detach().clone()

    # Max output tokens per rank is worst case where one rank receives all tokens
    input_tokens_per_rank = batch_size * seq_len
    max_output_tokens_per_rank = input_tokens_per_rank * dist.get_world_size()

    def using_bf16(
        input_tensor: torch.Tensor, input_splits: torch.Tensor
    ) -> torch.Tensor:
        # Calculate output splits from input splits
        output_splits = torch.empty_like(input_splits)
        dist.all_to_all_single(output_splits, input_splits)

        # Perform all-to-all
        out = all_to_all_single_autograd(
            input_tensor,
            output_splits.tolist(),
            input_splits.tolist(),
            dist.group.WORLD,
        )
        out = torch.ops._c10d_functional.wait_tensor(out)
        return out

    def using_mxfp8(
        input_tensor: torch.Tensor, input_splits: torch.Tensor
    ) -> torch.Tensor:
        output, output_splits = mxfp8_on_device_all_to_all_v(
            input_tensor,
            input_splits,
            max_output_tokens_per_rank,
            dist.group.WORLD.group_name,
        )
        output = torch.ops._c10d_functional.wait_tensor(output)
        output_splits = torch.ops._c10d_functional.wait_tensor(output_splits)
        return output

    def warmup(func_no_args):
        for _ in range(2):
            func_no_args()

    num_splits = dist.get_world_size()
    input_splits = generate_split_sizes(
        num_splits, input_tokens_per_rank, device=device
    )

    print(
        "Benchmarking using bf16",
        "batch_size",
        batch_size,
        "seq_len",
        seq_len,
        "dim",
        dim,
        "input_tokens_per_rank",
        input_tokens_per_rank,
        "max_output_tokens_per_rank",
        max_output_tokens_per_rank,
    )
    warmup(lambda: using_bf16(ref_x, input_splits))
    start_ns = time.perf_counter()
    using_bf16(ref_x, input_splits)
    end_ns = time.perf_counter()
    bf16_us = (end_ns - start_ns) * 1e6

    print(
        "Benchmarking using_mxfp8",
        "batch_size",
        batch_size,
        "seq_len",
        seq_len,
        "dim",
        dim,
        "input_tokens_per_rank",
        input_tokens_per_rank,
        "max_output_tokens_per_rank",
        max_output_tokens_per_rank,
    )
    warmup(lambda: using_mxfp8(x, input_splits))
    start_ns = time.perf_counter()
    using_mxfp8(x, input_splits)
    end_ns = time.perf_counter()
    mxfp8_us = (end_ns - start_ns) * 1e6

    return ExperimentResult(
        bf16_us=bf16_us,
        mxfp8_us=mxfp8_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "num_splits",
        "bf16_us",
        "mxfp8_us",
    ]
    rows = []
    num_splits = dist.get_world_size()
    for experiment in experiments:
        rows.append(
            [
                str(experiment.config.input_shape),
                num_splits,
                experiment.result.bf16_us,
                experiment.result.mxfp8_us,
            ]
        )
    print(tabulate(rows, headers=headers))


def generate_split_sizes(K: int, N: int, device: str = "cuda") -> torch.Tensor:
    """
    Generates a tensor of K random non-negative integers that sum to N.
    Used for testing mxfp8_all_to_all_v implementation.
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if N < 0:
        raise ValueError("N must be a non-negative integer.")

    if K == 1:
        return torch.tensor([N], dtype=torch.long, device=device)

    # Generate K-1 random "dividers" in the range [0, N].
    dividers = torch.randint(0, N + 1, (K - 1,), device=device)

    # Add 0 and N to the set of dividers to form the boundaries.
    boundaries = torch.cat(
        [torch.tensor([0], device=device), dividers, torch.tensor([N], device=device)]
    )

    # Sort the boundaries to ensure they are in order
    sorted_boundaries = torch.sort(boundaries).values

    # The K integers are the differences between consecutive boundaries (will sum to N)
    result = sorted_boundaries[1:] - sorted_boundaries[:-1]

    return result.to(dtype=torch.int64)


def main():
    torch.random.manual_seed(123)

    # Set up process group
    setup_distributed()

    # Generate experiment configs
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)

    # Clean up process group
    dist.destroy_process_group()


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


if __name__ == "__main__":
    main()
