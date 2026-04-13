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
from torchao.prototype.moe_training.ep.syncless.buffer_manager import (
    SymmetricMemoryBufferManager,
)
from torchao.prototype.moe_training.ep.syncless.token_combine import token_combine
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
    fwd_syncless_ms: float
    syncless_bandwidth_gbps: float


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


def _unpermute(out, input_shape, permuted_indices):
    """Scatter from expert-major layout back to rank-major order.

    Mirrors torchtitan's ExpertParallel._token_combine unpermute step.
    """
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    # Remove the padding row (last row added by permute_and_pad)
    return out_unpermuted[:-1]


def default_a2a_combine(
    combine_input: torch.Tensor,
    input_shape: torch.Size,
    permuted_indices: torch.Tensor,
    input_splits: list[int],
    output_splits: list[int],
    device_mesh: DeviceMesh,
):
    """Reference bf16 combine matching torchtitan's ExpertParallel._token_combine.

    Steps:
        1. _unpermute: scatter from expert-major back to rank-major layout
        2. all_to_all_single_autograd: send tokens back to source ranks
    """
    # Step 1: Unpermute from expert-major to rank-major order
    routed_output = _unpermute(combine_input, input_shape, permuted_indices)

    # Step 2: All-to-all to send tokens back to source ranks
    routed_output = all_to_all_single_autograd(
        routed_output,
        input_splits,
        output_splits,
        device_mesh.get_group(),
    )
    return routed_output


def syncless_a2a_combine(
    combine_input: torch.Tensor,
    all_expert_splits: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    total_local_tokens: int,
    group: dist.ProcessGroup,
    buffer_manager=None,
):
    """Syncless combine via symmetric memory push writes (zero D2H syncs)."""
    return token_combine(
        combine_input,
        all_expert_splits,
        expert_padded_offsets,
        total_local_tokens,
        group,
        buffer_manager,
    )


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

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    world_size = dist.get_world_size()

    def warmup(func_no_args):
        for _ in range(2):
            func_no_args()

    # --- Set up uniform dispatch splits ---
    input_tokens_per_rank = batch_size * seq_len
    num_experts_per_rank = 2
    num_experts = world_size * num_experts_per_rank
    input_tokens_per_expert = input_tokens_per_rank // num_experts
    num_tokens_per_expert = torch.tensor(
        input_tokens_per_expert, dtype=torch.int32, device=device
    ).repeat(num_experts)

    input_expert_splits = num_tokens_per_expert.view(world_size, -1).to(torch.int64)
    input_rank_splits = input_expert_splits.sum(dim=1)

    # ================================================================
    # Setup for DEFAULT combine: run default dispatch pipeline
    # (exchange splits, D2H sync, all_to_all, permute_and_pad)
    # ================================================================
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert,
            None,
            None,
            group=mesh.get_group(),
        )
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )
        default_input_splits = (
            num_tokens_per_expert.view(world_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        default_output_splits = (
            num_tokens_per_expert_group.view(world_size, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        default_input_splits_list = default_input_splits.tolist()
        default_output_splits_list = default_output_splits.tolist()

    # All-to-all data exchange
    routed_output = all_to_all_single_autograd(
        ref_x,
        default_output_splits_list,
        default_input_splits_list,
        mesh.get_group(),
    )

    # Permute to expert-major layout (produces permuted_indices and input_shape)
    default_input_shape, default_combine_input, permuted_indices, _, _ = (
        permute_and_pad(
            routed_output,
            num_tokens_per_expert_group,
            ep_degree=world_size,
            num_local_experts=num_experts_per_rank,
            alignment=32,
        )
    )

    # ================================================================
    # Setup for SYNCLESS combine: run syncless dispatch pipeline
    # ================================================================
    # Create a fresh buffer manager per experiment so that grad_input
    # is allocated with the correct size (the singleton from
    # get_buffer_manager() would keep the stale buffer from a prior
    # experiment with a smaller token count).
    buffer_manager = SymmetricMemoryBufferManager()
    total_tokens_across_all_ranks = x.shape[0] * world_size
    max_output_rows_per_rank = total_tokens_across_all_ranks

    buffer_manager.preallocate_buffers(
        max_output_rows_per_rank=max_output_rows_per_rank,
        data_shape=x.shape[1:],
        scales_shape=(x.shape[1] // 32,),
        data_dtype=torch.float8_e4m3fn,
        scales_dtype=torch.uint8,
        device=device,
    )

    (
        output_e4m3,
        output_scales_e8m0,
        output_rank_splits,
        output_expert_splits,
        expert_padded_offsets,
        all_expert_splits,
        _padded_tokens_per_expert,
    ) = mxfp8_token_dispatch(
        x,
        input_rank_splits,
        input_expert_splits,
        mesh.get_group(),
        buffer_manager,
    )

    syncless_combine_input = output_e4m3.to(torch.bfloat16)
    total_local_tokens = x.shape[0]

    # ================================================================
    # Benchmark DEFAULT combine (_unpermute + all_to_all_single)
    # ================================================================
    warmup(
        lambda: default_a2a_combine(
            default_combine_input,
            default_input_shape,
            permuted_indices,
            default_input_splits_list,
            default_output_splits_list,
            mesh,
        )
    )

    NUM_BENCH_ITERS = 10
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    for _ in range(NUM_BENCH_ITERS):
        _ = default_a2a_combine(
            default_combine_input,
            default_input_shape,
            permuted_indices,
            default_input_splits_list,
            default_output_splits_list,
            mesh,
        )
    torch.cuda.synchronize()
    end_sec = time.perf_counter()

    fwd_bf16_ms = (end_sec - start_sec) * 1e3 / NUM_BENCH_ITERS
    if args.profile:

        def default_combine_batch():
            for _ in range(10):
                default_a2a_combine(
                    default_combine_input,
                    default_input_shape,
                    permuted_indices,
                    default_input_splits_list,
                    default_output_splits_list,
                    mesh,
                )

        profile_fn(
            default_combine_batch,
            distributed=True,
            profile_name="default_a2a_combine",
            active_steps=1,
        )

    # ================================================================
    # Benchmark SYNCLESS combine (token_combine)
    # ================================================================
    warmup(
        lambda: syncless_a2a_combine(
            syncless_combine_input,
            all_expert_splits,
            expert_padded_offsets,
            total_local_tokens,
            mesh.get_group(),
            buffer_manager,
        )
    )

    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    for _ in range(NUM_BENCH_ITERS):
        _ = syncless_a2a_combine(
            syncless_combine_input,
            all_expert_splits,
            expert_padded_offsets,
            total_local_tokens,
            mesh.get_group(),
            buffer_manager,
        )
    torch.cuda.synchronize()
    end_sec = time.perf_counter()

    fwd_syncless_ms = (end_sec - start_sec) * 1e3 / NUM_BENCH_ITERS
    if args.profile:

        def syncless_combine_batch():
            for _ in range(10):
                syncless_a2a_combine(
                    syncless_combine_input,
                    all_expert_splits,
                    expert_padded_offsets,
                    total_local_tokens,
                    mesh.get_group(),
                    buffer_manager,
                )

        profile_fn(
            syncless_combine_batch,
            distributed=True,
            profile_name="syncless_a2a_combine",
            active_steps=1,
        )

    # --- Bandwidth calculation ---
    # Combine sends bf16 data (2 bytes per element) back to source ranks.
    input_data_bytes = batch_size * seq_len * dim * 2
    network_fraction = (world_size - 1) / world_size if world_size > 1 else 0
    total_network_bytes_per_rank = 2 * input_data_bytes * network_fraction
    total_network_gb = total_network_bytes_per_rank / (1024**3)
    syncless_bandwidth_gbps = (
        (total_network_gb / (fwd_syncless_ms / 1000)) if fwd_syncless_ms > 0 else 0
    )

    return ExperimentResult(
        fwd_bf16_ms=fwd_bf16_ms,
        fwd_syncless_ms=fwd_syncless_ms,
        syncless_bandwidth_gbps=syncless_bandwidth_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "num_ranks",
        "bf16_ms",
        "syncless_ms",
        "speedup",
        "syncless_bw_gbps",
    ]
    rows = []
    num_ranks = dist.get_world_size()

    for experiment in experiments:
        speedup = (
            experiment.result.fwd_bf16_ms / experiment.result.fwd_syncless_ms
            if experiment.result.fwd_syncless_ms > 0
            else float("inf")
        )

        rows.append(
            [
                str(experiment.config.input_shape),
                num_ranks,
                f"{experiment.result.fwd_bf16_ms:.2f}",
                f"{experiment.result.fwd_syncless_ms:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.syncless_bandwidth_gbps:.1f}",
            ]
        )

    print("\n" + "=" * 100)
    print("TOKEN COMBINE BENCHMARK RESULTS")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    if len(experiments) > 0:
        avg_syncless_bw = sum(
            e.result.syncless_bandwidth_gbps for e in experiments
        ) / len(experiments)
        print("\nBandwidth Summary:")
        print(f"  Syncless average NVLink bandwidth: {avg_syncless_bw:.1f} GB/s")
        print("=" * 80)


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)

    setup_distributed()

    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    print_results(results)

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
