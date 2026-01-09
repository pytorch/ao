# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run this benchmark, use the following command:
#
# torchrun --nproc-per-node=2 --local-ranks-filter=0 benchmarks/prototype/moe_training/mxfp8/bench_ep_pipeline.py
#
#######################################################################
import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from tabulate import tabulate
from torch import distributed as dist
from torch.distributed._functional_collectives import all_to_all_single
from torch.nn import functional as F
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from benchmarks.utils import profile_fn
from torchao.prototype.moe_training.ep import (
    a2a_combine_hp_fwd_mxfp8_bwd,
    a2a_dispatch_mxfp8_fwd_hp_bwd,
    permute_mxfp8_fwd_hp_bwd,
    unpermute_hp_fwd_mxfp8_bwd,
)
from torchao.prototype.moe_training.ep.permute import _permute_bf16
from torchao.prototype.moe_training.ep.unpermute import _unpermute_bf16
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    dim: int
    hidden_dim: int
    num_experts: int


@dataclass(frozen=True)
class ExperimentResult:
    # Forward times
    fwd_bf16_ms: float
    fwd_mxfp8_ms: float
    # Backward times
    bwd_bf16_ms: float
    bwd_mxfp8_ms: float
    # Speedup metrics
    fwd_speedup: float
    bwd_speedup: float
    total_speedup: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    """Generate experiment configurations."""
    configs = [
        ExperimentConfig(num_tokens=131072, dim=8192, hidden_dim=5120, num_experts=8),
        ExperimentConfig(num_tokens=131072, dim=7168, hidden_dim=2048, num_experts=8),
    ]
    return configs


def generate_split_sizes(K: int, N: int, device: str = "cuda") -> torch.Tensor:
    """
    Generates a tensor of K random non-negative integers that sum to N.
    """
    if K <= 0:
        raise ValueError("K must be a positive integer.")
    if N < 0:
        raise ValueError("N must be a non-negative integer.")

    if K == 1:
        return torch.tensor([N], dtype=torch.int32, device=device)

    # Generate K-1 random "dividers" in the range [0, N].
    dividers = torch.randint(0, N + 1, (K - 1,), device=device)

    # Add 0 and N to the set of dividers to form the boundaries.
    boundaries = torch.cat(
        [
            torch.tensor([0], device=device),
            dividers,
            torch.tensor([N], device=device),
        ]
    )

    # Sort the boundaries to ensure they are in order
    sorted_boundaries = torch.sort(boundaries).values

    # The K integers are the differences between consecutive boundaries
    result = sorted_boundaries[1:] - sorted_boundaries[:-1]

    return result.to(dtype=torch.int32)


def standard_pipeline(
    input_tensor: torch.Tensor,
    expert_weights_t: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_per_expert_group: torch.Tensor,
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_degree: int,
    num_experts: int,
    group,
) -> torch.Tensor:
    """
    Standard BF16 pipeline:
    bf16 a2a -> bf16 permute -> _to_mxfp8_then_scaled_grouped_mm -> bf16 unpermute -> bf16 a2a combine
    """
    block_size = 32

    # Step 1: All-to-all dispatch (BF16)
    dispatched = all_to_all_single(
        input_tensor,
        output_splits_list,
        input_splits_list,
        group=group,
    )
    dispatched = torch.ops._c10d_functional.wait_tensor(dispatched)

    # Step 2: Permute (BF16)
    input_shape, permuted, permuted_indices, num_tokens_per_expert_padded, offsets = (
        _permute_bf16(
            dispatched,
            num_tokens_per_expert_group,
            ep_degree,
            num_experts,
            block_size,
        )
    )

    # Step 3: BF16 Grouped MM
    gemm_output = _to_mxfp8_then_scaled_grouped_mm(
        permuted,
        expert_weights_t,
        offs=offsets,
        out_dtype=torch.bfloat16,
        use_cuda_kernel_for_blocked_layout=True,
        wgrad_with_hp=True,
    )

    # Step 4: Unpermute (BF16)
    # Create output shape with same number of rows as input_shape, but output dimension from gemm_output
    output_shape = (input_shape[0], gemm_output.shape[-1])
    unpermuted = _unpermute_bf16(gemm_output, permuted_indices, output_shape)

    # Step 5: All-to-all combine (BF16)
    final_output = all_to_all_single(
        unpermuted,
        input_splits_list,
        output_splits_list,
        group=group,
    )
    final_output = torch.ops._c10d_functional.wait_tensor(final_output)

    return final_output


def mxfp8_pipeline(
    input_tensor: torch.Tensor,
    expert_weights_t: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_tokens_per_expert_group: torch.Tensor,
    input_splits_list: List[int],
    output_splits_list: List[int],
    ep_degree: int,
    num_experts: int,
    group,
) -> torch.Tensor:
    """
    MXFP8 optimized pipeline with chained autograd functions:
    bf16 -> a2a_dispatch (MXTensor) -> permute (MXTensor) ->
    mxfp8_grouped_mm -> unpermute -> a2a_combine -> bf16
    """
    block_size = 32

    # Step 1: A2A dispatch - outputs MXTensor
    mx_dispatched = a2a_dispatch_mxfp8_fwd_hp_bwd(
        input_tensor,
        output_splits_list,
        input_splits_list,
        group=group,
    )

    # Step 2: Permute - maintains MXTensor
    (
        padded_mx_shape,
        mx_permuted,
        permuted_indices,
        num_tokens_per_expert_padded,
        mx_group_offsets,
    ) = permute_mxfp8_fwd_hp_bwd(
        mx_dispatched,
        num_tokens_per_expert_group,
        ep_degree,
        num_experts,
        block_size,
        use_triton_for_bwd=True,
    )

    # Step 3: MXFP8 Grouped MM - outputs BF16
    gemm_output = _to_mxfp8_then_scaled_grouped_mm(
        mx_permuted,
        expert_weights_t,
        offs=mx_group_offsets,
        block_size=block_size,
        use_cuda_kernel_for_blocked_layout=True,
        wgrad_with_hp=True,
    )

    # Step 4: Unpermute - maintains BF16
    # Update padded_shape to have output dimension instead of input dimension
    padded_output_shape = torch.Size([padded_mx_shape[0], gemm_output.shape[-1]])
    unpermuted = unpermute_hp_fwd_mxfp8_bwd(
        gemm_output,
        permuted_indices,
        padded_output_shape,
    )

    # Step 5: A2A combine - maintains BF16
    final_output = a2a_combine_hp_fwd_mxfp8_bwd(
        unpermuted,
        output_splits=input_splits_list,
        input_splits=output_splits_list,
        group=group,
        mxfp8_bwd=True,
    )

    return final_output


def mse_loss_and_bwd(output: torch.Tensor, labels: torch.Tensor):
    """Compute MSE loss and run backward pass."""
    loss = F.mse_loss(output, labels)
    loss.backward()


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    """Run a single experiment comparing both pipelines."""
    num_tokens = config.num_tokens
    dim = config.dim
    hidden_dim = config.hidden_dim
    num_experts = config.num_experts

    # Create input tensors
    input_tensor = torch.randn(
        num_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    ref_input_tensor = input_tensor.detach().clone().requires_grad_(True)

    expert_weights = torch.randn(
        num_experts,
        hidden_dim,
        dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    ref_expert_weights = expert_weights.detach().clone().requires_grad_(True)

    # Generate token distribution
    ep_degree = dist.get_world_size()
    total_experts = ep_degree * num_experts
    assert num_tokens % total_experts == 0
    uniform_group_size = num_tokens // total_experts
    num_tokens_per_expert = torch.full(
        (total_experts,), uniform_group_size, dtype=torch.int32, device="cuda"
    )

    # Compute splits for all-to-all
    group = dist.group.WORLD
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert,
            None,
            None,
            group=group,
        )
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )
        input_splits = (
            num_tokens_per_expert.view(ep_degree, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        output_splits = (
            num_tokens_per_expert_group.view(ep_degree, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )

    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()

    # Warmup function
    def warmup(func_no_args, n=2):
        for _ in range(n):
            func_no_args()

    # Set seed for deterministic execution across both pipelines
    torch.manual_seed(42)

    # === Benchmark Standard BF16 Pipeline ===

    # BF16 Forward
    def bf16_fwd(input_t, weight_t):
        return standard_pipeline(
            input_t,
            weight_t,
            num_tokens_per_expert,
            num_tokens_per_expert_group,
            input_splits_list,
            output_splits_list,
            ep_degree,
            num_experts,
            group,
        )

    warmup(lambda: bf16_fwd(ref_input_tensor, ref_expert_weights.transpose(-2, -1)))
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    _ = bf16_fwd(ref_input_tensor, ref_expert_weights.transpose(-2, -1))
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    fwd_bf16_ms = (end_sec - start_sec) * 1e3

    # BF16 Backward
    # Warmup backward pass
    def bf16_bwd_warmup():
        ref_input_tensor.grad = None
        ref_expert_weights.grad = None
        output = bf16_fwd(ref_input_tensor, ref_expert_weights.transpose(-2, -1))
        labels = torch.ones_like(output)
        mse_loss_and_bwd(output, labels)

    warmup(bf16_bwd_warmup)

    # Do a fresh forward pass right before timing backward
    ref_input_tensor.grad = None
    ref_expert_weights.grad = None
    bf16_output_for_bwd = bf16_fwd(
        ref_input_tensor, ref_expert_weights.transpose(-2, -1)
    )
    bf16_labels = torch.ones_like(bf16_output_for_bwd)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    mse_loss_and_bwd(bf16_output_for_bwd, bf16_labels)
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    bwd_bf16_ms = (end_sec - start_sec) * 1e3

    # BF16 Forward + Backward
    def bf16_fwd_bwd(input_t, weight_t, labels):
        output = bf16_fwd(input_t, weight_t)
        mse_loss_and_bwd(output, labels)

    if args.profile:
        # Create fresh tensors for profiling to avoid autograd graph conflicts
        ref_input_tensor_profile = torch.randn(
            num_tokens,
            dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        ref_expert_weights_profile = torch.randn(
            num_experts,
            hidden_dim,
            dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        # Profile backward using fresh tensors
        profile_fn(
            bf16_fwd_bwd,
            ref_input_tensor_profile,
            ref_expert_weights_profile.transpose(-2, -1),
            bf16_labels,
            distributed=True,
            profile_name="bf16_pipeline",
        )

    # === Benchmark MXFP8 Pipeline ===

    # Reset seed to ensure same random state as BF16 pipeline
    torch.manual_seed(42)

    # MXFP8 Forward
    def mxfp8_fwd(input_t, weight_t):
        return mxfp8_pipeline(
            input_t,
            weight_t,
            num_tokens_per_expert,
            num_tokens_per_expert_group,
            input_splits_list,
            output_splits_list,
            ep_degree,
            num_experts,
            group,
        )

    warmup(lambda: mxfp8_fwd(input_tensor, expert_weights.transpose(-2, -1)))
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    _ = mxfp8_fwd(input_tensor, expert_weights.transpose(-2, -1))
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    fwd_mxfp8_ms = (end_sec - start_sec) * 1e3

    # MXFP8 Backward
    # Warmup backward pass to compile Triton kernels
    def mxfp8_bwd_warmup():
        input_tensor.grad = None
        expert_weights.grad = None
        output = mxfp8_fwd(input_tensor, expert_weights.transpose(-2, -1))
        labels = torch.ones_like(output)
        mse_loss_and_bwd(output, labels)

    warmup(mxfp8_bwd_warmup)

    # Do a fresh forward pass right before timing backward
    input_tensor.grad = None
    expert_weights.grad = None
    mxfp8_output_for_bwd = mxfp8_fwd(input_tensor, expert_weights.transpose(-2, -1))
    mxfp8_labels = torch.ones_like(mxfp8_output_for_bwd)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    mse_loss_and_bwd(mxfp8_output_for_bwd, mxfp8_labels)
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    bwd_mxfp8_ms = (end_sec - start_sec) * 1e3

    # MXFP8 Forward + Backward
    def mxfp8_fwd_bwd(input_t, weight_t, labels):
        output = mxfp8_fwd(input_t, weight_t)
        mse_loss_and_bwd(output, labels)

    if args.profile:
        # Create fresh tensors for profiling to avoid autograd graph conflicts
        input_tensor_profile = torch.randn(
            num_tokens,
            dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        expert_weights_profile = torch.randn(
            num_experts,
            hidden_dim,
            dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        # Profile backward using fresh tensors
        profile_fn(
            mxfp8_fwd_bwd,
            input_tensor_profile,
            expert_weights_profile.transpose(-2, -1),
            mxfp8_labels,
            distributed=True,
            profile_name="mxfp8_pipeline",
        )

    # Calculate speedups
    fwd_speedup = fwd_bf16_ms / fwd_mxfp8_ms
    bwd_speedup = bwd_bf16_ms / bwd_mxfp8_ms
    total_bf16_ms = fwd_bf16_ms + bwd_bf16_ms
    total_mxfp8_ms = fwd_mxfp8_ms + bwd_mxfp8_ms
    total_speedup = total_bf16_ms / total_mxfp8_ms

    return ExperimentResult(
        fwd_bf16_ms=fwd_bf16_ms,
        fwd_mxfp8_ms=fwd_mxfp8_ms,
        bwd_bf16_ms=bwd_bf16_ms,
        bwd_mxfp8_ms=bwd_mxfp8_ms,
        fwd_speedup=fwd_speedup,
        bwd_speedup=bwd_speedup,
        total_speedup=total_speedup,
    )


def print_results(experiments: List[Experiment]):
    """Print benchmark results in a formatted table."""
    headers = [
        "tokens",
        "dim",
        "hidden_dim",
        "num_experts",
        "fwd_bf16_ms",
        "fwd_mxfp8_ms",
        "fwd_speedup",
        "bwd_bf16_ms",
        "bwd_mxfp8_ms",
        "bwd_speedup",
        "total_speedup",
    ]
    rows = []
    for experiment in experiments:
        cfg = experiment.config
        res = experiment.result
        rows.append(
            [
                cfg.num_tokens,
                cfg.dim,
                cfg.hidden_dim,
                cfg.num_experts,
                f"{res.fwd_bf16_ms:.3f}",
                f"{res.fwd_mxfp8_ms:.3f}",
                f"{res.fwd_speedup:.2f}x",
                f"{res.bwd_bf16_ms:.3f}",
                f"{res.bwd_mxfp8_ms:.3f}",
                f"{res.bwd_speedup:.2f}x",
                f"{res.total_speedup:.2f}x",
            ]
        )
    print("\n" + "=" * 120)
    print("Expert Parallelism Pipeline Benchmark Results")
    print(f"World Size: {dist.get_world_size()}")
    print("=" * 120)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 120 + "\n")


def main(args: argparse.Namespace):
    """Main benchmark entry point."""
    torch.random.manual_seed(123)

    # Set up process group
    setup_distributed()

    # Generate experiment configs
    configs = get_configs()
    results = []
    for config in tqdm(
        configs, desc="Running experiments", disable=dist.get_rank() != 0
    ):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    # Print results (only on rank 0)
    if dist.get_rank() == 0:
        print_results(results)

    # Clean up process group
    dist.destroy_process_group()


def setup_distributed():
    """Initialize distributed process group."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MoE Expert Parallelism pipelines (BF16 vs MXFP8)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for detailed performance analysis",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )
    args = parser.parse_args()
    main(args)
