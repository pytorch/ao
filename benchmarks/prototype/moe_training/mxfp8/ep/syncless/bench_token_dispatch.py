# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these benchmarks, use the following command:
#
# torchrun --nproc-per-node=4 --local-ranks-filter=0 <path to file>
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
from tqdm import tqdm

from benchmarks.utils import profile_fn
from torchao.prototype.moe_training.ep.permute import permute_and_pad
from torchao.prototype.moe_training.ep.syncless import (
    get_buffer_manager,
)
from torchao.prototype.moe_training.ep.syncless.token_dispatch import (
    mxfp8_token_dispatch,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    fwd_bf16_ms: float
    fwd_mxfp8_ms: float
    mxfp8_bandwidth_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # (batch_size, seq_len, dim)
    input_shapes = [(1, 8192, 7168), (4, 8192, 7168)]
    configs = []
    for shape in input_shapes:
        configs.append(
            ExperimentConfig(
                input_shape=shape,
            )
        )
    return configs


def default_a2a_fwd(
    routed_input: torch.Tensor,
    input_rank_splits: torch.Tensor,
    input_expert_splits: torch.Tensor,
    device_mesh: DeviceMesh,
    num_experts_per_rank: int,
):
    world_size = dist.get_world_size(device_mesh.get_group())

    # Step 1: Device-to-host sync to get splits for NCCL API
    input_splits_list = input_rank_splits.cpu().tolist()

    # Step 2: Compute output splits from input splits using all_to_all_single
    output_rank_splits = torch.empty_like(input_rank_splits)
    dist.all_to_all_single(
        output_rank_splits, input_rank_splits, group=device_mesh.get_group()
    )
    output_splits_list = output_rank_splits.cpu().tolist()  # Another d2h sync

    # Step 3: Do actual all_to_all_single to exchange token data
    routed_output = all_to_all_single_autograd(
        routed_input,
        output_splits_list,
        input_splits_list,
        device_mesh.get_group(),
    )
    routed_output = torch.ops._c10d_functional.wait_tensor(routed_output)

    # Step 4: Exchange expert split information using all_to_all_single (like TorchAO)
    # Flatten input_expert_splits to match TorchAO's num_tokens_per_expert format
    num_tokens_per_expert = (
        input_expert_splits.flatten()
    )  # [world_size * num_experts_per_rank]

    # Use all_to_all_single to exchange expert splits (no redundant all_gather needed!)
    tokens_per_expert_group = all_to_all_single(
        num_tokens_per_expert,
        None,
        None,
        group=device_mesh.get_group(),
    )
    tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
        tokens_per_expert_group
    )

    _, expert_major_output, _, _, _ = permute_and_pad(
        routed_output,
        tokens_per_expert_group,
        ep_degree=world_size,
        num_local_experts=num_experts_per_rank,
        alignment=32,
    )

    return expert_major_output


def mxfp8_a2a_fwd(
    routed_input: torch.Tensor,
    input_rank_splits: torch.Tensor,
    input_expert_splits: torch.Tensor,
    device_mesh: DeviceMesh,
    buffer_manager=None,
):
    (
        output_e4m3,
        output_scales_e8m0,
        output_rank_splits,
        output_expert_splits,
        expert_padded_offsets,
        all_expert_splits,
        _padded_tokens_per_expert,
    ) = mxfp8_token_dispatch(
        routed_input,
        input_rank_splits,
        input_expert_splits,
        device_mesh.get_group(),
        buffer_manager,
    )

    return output_e4m3, output_scales_e8m0, output_expert_splits


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    batch_size, seq_len, dim = config.input_shape
    x = torch.randn(
        (batch_size * seq_len, dim),
        dtype=torch.bfloat16,
        device=device,
    )
    ref_x = x.detach().clone()

    # Set up device mesh
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    def warmup(func_no_args):
        for _ in range(2):
            func_no_args()

    input_tokens_per_rank = batch_size * seq_len
    num_experts_per_rank = 2
    num_experts = dist.get_world_size() * num_experts_per_rank
    input_tokens_per_expert = input_tokens_per_rank // num_experts
    input_splits = torch.tensor(
        input_tokens_per_expert, dtype=torch.int32, device=device
    ).repeat(num_experts)
    input_splits_list, output_splits_list = get_split_lists(input_splits, mesh)

    # Create input_expert_splits for default implementation
    # For this benchmark, assume uniform distribution across experts
    tokens_per_expert_per_rank = [
        split // num_experts_per_rank for split in input_splits_list
    ]
    input_expert_splits = (
        torch.tensor(tokens_per_expert_per_rank, dtype=torch.int64, device=device)
        .unsqueeze(1)
        .repeat(1, num_experts_per_rank)
    )

    # Compute input_rank_splits from input_splits
    input_rank_splits = input_splits.view(dist.get_world_size(), -1).sum(dim=1)

    # Bench default a2a fwd (includes d2h sync overhead for NCCL API)
    warmup(
        lambda: default_a2a_fwd(
            ref_x, input_rank_splits, input_expert_splits, mesh, num_experts_per_rank
        )
    )

    # Run 10 iterations and take average
    NUM_BENCH_ITERS = 10
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    for _ in range(NUM_BENCH_ITERS):
        bf16_routed_input = default_a2a_fwd(
            ref_x, input_rank_splits, input_expert_splits, mesh, num_experts_per_rank
        )
    torch.cuda.synchronize()
    end_sec = time.perf_counter()

    fwd_bf16_ms = (end_sec - start_sec) * 1e3 / NUM_BENCH_ITERS
    if args.profile:

        def default_a2a_batch():
            for _ in range(10):
                default_a2a_fwd(
                    ref_x,
                    input_rank_splits,
                    input_expert_splits,
                    mesh,
                    num_experts_per_rank,
                )

        profile_fn(
            default_a2a_batch,
            distributed=True,
            profile_name="default_a2a_fwd",
            active_steps=1,
        )

    # Preallocate buffer manager (not timed - reused across model)
    buffer_manager = get_buffer_manager()

    # Calculate worst-case buffer size and preallocate buffers
    world_size = dist.get_world_size()
    total_tokens_across_all_ranks = x.shape[0] * world_size
    max_output_rows_per_rank = total_tokens_across_all_ranks

    # Preallocate symmetric memory buffers (simulates what happens during model init)
    buffer_manager.preallocate_buffers(
        max_output_rows_per_rank=max_output_rows_per_rank,
        data_shape=x.shape[1:],  # (dim,)
        scales_shape=(x.shape[1] // 32,),  # Assuming 32-element blocks for MXFP8 scales
        data_dtype=torch.float8_e4m3fn,
        scales_dtype=torch.uint8,
        device=device,
    )

    # Bench mxfp8 sync a2a fwd (zero device-to-host syncs!)
    warmup(
        lambda: mxfp8_a2a_fwd(
            x, input_rank_splits, input_expert_splits, mesh, buffer_manager
        )[0]  # Only use the output_e4m3 for warmup
    )

    # Run 10 iterations and take average
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    for _ in range(NUM_BENCH_ITERS):
        mxfp8_routed_input_e4m3, mxfp8_routed_input_scales, output_expert_splits = (
            mxfp8_a2a_fwd(
                x, input_rank_splits, input_expert_splits, mesh, buffer_manager
            )
        )
    torch.cuda.synchronize()
    end_sec = time.perf_counter()

    fwd_mxfp8_ms = (end_sec - start_sec) * 1e3 / NUM_BENCH_ITERS
    if args.profile:

        def mxfp8_a2a_batch():
            for _ in range(10):
                mxfp8_a2a_fwd(
                    x, input_rank_splits, input_expert_splits, mesh, buffer_manager
                )

        profile_fn(
            mxfp8_a2a_batch,
            distributed=True,
            profile_name="mxfp8_a2a_fwd",
            active_steps=1,
        )

    # Correctness check: quantize reference and compare directly
    from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

    # Use reference size for comparison - it allocates exactly what's needed
    # while mxfp8_token_dispatch may overallocate for worst-case scenario
    compare_size = bf16_routed_input.shape[0]

    # Quantize reference output to MXFP8 for comparison
    block_size = 32
    ref_output_e4m3, ref_output_scales_e8m0 = triton_to_mxfp8_dim0(
        bf16_routed_input,
        inner_block_size=block_size,
        scaling_mode="rceil",
    )

    if dist.get_rank() == 0:
        try:
            # Compare quantized tensors directly
            torch.testing.assert_close(
                mxfp8_routed_input_e4m3[:compare_size].view(torch.float32),
                ref_output_e4m3.view(torch.float32),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                mxfp8_routed_input_scales[:compare_size].view(torch.uint8),
                ref_output_scales_e8m0.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            print(f"Correctness check passed for shape {config.input_shape}")
        except AssertionError as e:
            print(f"Correctness check failed for shape {config.input_shape}: {e}")

    # Calculate bandwidth utilization
    world_size = dist.get_world_size()

    # Data volume calculation
    # Each rank has input data of size: batch_size * seq_len * dim * 2 bytes (bfloat16)
    input_data_bytes = batch_size * seq_len * dim * 2

    # In all-to-all, each rank sends its data to all ranks (including itself)
    # Network traffic excludes the local copy:
    #   sent_bytes = input_data_bytes * (world_size - 1) / world_size
    # Each rank both sends and receives, so:
    #   total network bytes = 2 * sent_bytes
    network_fraction = (world_size - 1) / world_size if world_size > 1 else 0
    total_network_bytes_per_rank = 2 * input_data_bytes * network_fraction

    # Convert to GB and calculate bandwidth
    total_network_gb = total_network_bytes_per_rank / (1024**3)

    # Calculate bandwidth in GB/s (only for MXFP8)
    mxfp8_bandwidth_gbps = (
        (total_network_gb / (fwd_mxfp8_ms / 1000)) if fwd_mxfp8_ms > 0 else 0
    )

    return ExperimentResult(
        fwd_bf16_ms=fwd_bf16_ms,
        fwd_mxfp8_ms=fwd_mxfp8_ms,
        mxfp8_bandwidth_gbps=mxfp8_bandwidth_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "num_ranks",
        "bf16_ms",
        "mxfp8_ms",
        "speedup",
        "mxfp8_bw_gbps",
    ]
    rows = []
    num_ranks = dist.get_world_size()

    for experiment in experiments:
        speedup = (
            experiment.result.fwd_bf16_ms / experiment.result.fwd_mxfp8_ms
            if experiment.result.fwd_mxfp8_ms > 0
            else float("inf")
        )

        rows.append(
            [
                str(experiment.config.input_shape),
                num_ranks,
                f"{experiment.result.fwd_bf16_ms:.2f}",
                f"{experiment.result.fwd_mxfp8_ms:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.mxfp8_bandwidth_gbps:.1f}",
            ]
        )

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    if len(experiments) > 0:
        avg_mxfp8_bw = sum(e.result.mxfp8_bandwidth_gbps for e in experiments) / len(
            experiments
        )
        print("\nBandwidth Summary:")
        print(f"  MXFP8 average NVLink bandwidth: {avg_mxfp8_bw:.1f} GB/s")
        print("=" * 80)


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
