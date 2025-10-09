# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these benchmarks, use the following command:
#
# torchrun --nproc-per-node=4 --local-ranks-filter=0 benchmarks/prototype/moe_training/mxfp8/bench_all_to_all_v.py
#
#######################################################################
import argparse
import os
import time
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from torch import distributed as dist
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.nn import functional as F
from tqdm import tqdm

from benchmarks.utils import profile_fn
from torchao.prototype.moe_training.kernels.mxfp8.comms import (
    to_mxfp8_a2a_dequant,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    bf16_ms: float
    mxfp8_ms: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # (batch_size, seq_len, dim)
    input_shapes = [
        (16, 8192, 5120),
    ]
    configs = []
    for shape in input_shapes:
        configs.append(
            ExperimentConfig(
                input_shape=shape,
            )
        )
    return configs


def default_a2a_fwd_bwd(
    routed_input: torch.Tensor,
    labels: torch.Tensor,
    output_splits_list: list[int],
    input_splits_list: list[int],
    device_mesh: DeviceMesh,
):
    routed_input = all_to_all_single_autograd(
        routed_input,
        output_splits_list,
        input_splits_list,
        device_mesh.get_group(),
    )
    routed_input = torch.ops._c10d_functional.wait_tensor(routed_input)

    loss = F.mse_loss(routed_input, labels)
    loss.backward()

    torch.cuda.synchronize()
    return routed_input


def mxfp8_a2a_fwd_bwd(
    routed_input: torch.Tensor,
    labels: torch.Tensor,
    output_splits_list: list[int],
    input_splits_list: list[int],
    device_mesh: DeviceMesh,
):
    routed_input = to_mxfp8_a2a_dequant(
        routed_input,
        output_splits_list,
        input_splits_list,
        device_mesh.get_group(),
    )

    loss = F.mse_loss(routed_input, labels)
    loss.backward()
    torch.cuda.synchronize()
    return routed_input


# Compile target funcs
default_a2a_sync_compiled = torch.compile(default_a2a_fwd_bwd)
mxfp8_a2a_sync_compiled = torch.compile(mxfp8_a2a_fwd_bwd)


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    batch_size, seq_len, dim = config.input_shape
    x = torch.randn(
        (batch_size * seq_len, dim),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    ref_x = x.detach().clone().requires_grad_(True)

    # Set up device mesh
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Max output tokens per rank is worst case where one rank receives all tokens
    input_tokens_per_rank = batch_size * seq_len

    def warmup(func_no_args):
        for _ in range(2):
            func_no_args()

    num_experts_per_rank = 2
    num_splits = dist.get_world_size() * num_experts_per_rank
    input_splits = generate_split_sizes(
        num_splits, input_tokens_per_rank, device=device
    )
    input_splits_list, output_splits_list = get_split_lists(input_splits, mesh)

    # Generate labels
    labels_shape = (sum(output_splits_list), dim)
    labels = x.new_ones(*labels_shape)

    # Bench default a2a (exclude d2h sync from preparing input splits_list and output_splits_list)
    warmup(
        lambda: default_a2a_sync_compiled(
            ref_x, labels, output_splits_list, input_splits_list, mesh
        )
    )
    start_sec = time.perf_counter()
    default_a2a_sync_compiled(
        ref_x, labels, output_splits_list, input_splits_list, mesh
    )
    end_sec = time.perf_counter()
    bf16_ms = (end_sec - start_sec) * 1e3
    if args.profile:
        profile_fn(
            default_a2a_sync_compiled,
            ref_x,
            labels,
            output_splits_list,
            input_splits_list,
            mesh,
            distributed=True,
            profile_name="all_to_all_single_autograd",
        )

    # Bench mxfp8 sync a2a (exclude d2h sync from preparing input splits_list and output_splits_list)
    warmup(
        lambda: mxfp8_a2a_sync_compiled(
            x, labels, output_splits_list, input_splits_list, mesh
        )
    )
    start_sec = time.perf_counter()
    mxfp8_a2a_sync_compiled(x, labels, output_splits_list, input_splits_list, mesh)
    end_sec = time.perf_counter()
    mxfp8_ms = (end_sec - start_sec) * 1e3
    if args.profile:
        profile_fn(
            mxfp8_a2a_sync_compiled,
            x,
            labels,
            output_splits_list,
            input_splits_list,
            mesh,
            distributed=True,
            profile_name="to_mxfp8_a2a_dequant",
        )

    return ExperimentResult(
        bf16_ms=bf16_ms,
        mxfp8_ms=mxfp8_ms,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "num_splits",
        "bf16_ms",
        "mxfp8_ms",
    ]
    rows = []
    num_splits = dist.get_world_size()
    for experiment in experiments:
        rows.append(
            [
                str(experiment.config.input_shape),
                num_splits,
                experiment.result.bf16_ms,
                experiment.result.mxfp8_ms,
            ]
        )
    print(tabulate(rows, headers=headers))


def get_split_lists(
    num_tokens_per_expert: torch.Tensor, device_mesh: DeviceMesh
) -> tuple[list[int], list[int]]:
    ep_degree = device_mesh.size(0)

    # generate the input splits and output splits for sync-impls
    num_tokens_per_expert_group = all_to_all_single(
        num_tokens_per_expert,
        None,
        None,
        group=device_mesh.get_group(),
    )
    # Need to wait explicitly because it is used by a triton kernel later
    # which doesn't realize that AsyncCollectiveTensor needs unwrapping
    num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
        num_tokens_per_expert_group
    )
    input_splits = (
        num_tokens_per_expert.view(ep_degree, -1)
        .sum(dim=1)
        .to(torch.device("cpu"), non_blocking=True)
    )
    # NOTE: this would incur a device-to-host sync
    output_splits = (
        num_tokens_per_expert_group.view(ep_degree, -1)
        .sum(dim=1)
        .to(torch.device("cpu"), non_blocking=False)
    )

    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()

    return input_splits_list, output_splits_list


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


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)

    # Set up process group
    setup_distributed()

    # Generate experiment configs
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    main(args)
