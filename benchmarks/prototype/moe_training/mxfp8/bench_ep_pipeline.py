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
    fwd_mxfp8_no_ep_ms: float
    fwd_mxfp8_with_ep_ms: float
    # Backward times
    bwd_bf16_ms: float
    bwd_mxfp8_no_ep_ms: float
    bwd_mxfp8_with_ep_ms: float
    # Speedup metrics (vs BF16 baseline)
    fwd_mxfp8_no_ep_speedup: float
    bwd_mxfp8_no_ep_speedup: float
    total_mxfp8_no_ep_speedup: float
    fwd_mxfp8_with_ep_speedup: float
    bwd_mxfp8_with_ep_speedup: float
    total_mxfp8_with_ep_speedup: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    """Generate experiment configurations."""
    configs = [
        ExperimentConfig(num_tokens=131072, dim=8192, hidden_dim=5120, num_experts=8),
        ExperimentConfig(num_tokens=131072, dim=7168, hidden_dim=2048, num_experts=8),
        ExperimentConfig(num_tokens=131072, dim=5120, hidden_dim=1536, num_experts=8),
        ExperimentConfig(num_tokens=131072, dim=2048, hidden_dim=1408, num_experts=8),
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
    wgrad_with_hp: bool = True,
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
    scale_cols = expert_weights_t.shape[-1] // 32 # after down proj to N dim, grad_out scale cols may not be compatible with cuda kernel
    can_use_cuda_for_blocked_layout = scale_cols >= 64 and scale_cols % 16 == 0
    gemm_output = _to_mxfp8_then_scaled_grouped_mm(
        permuted,
        expert_weights_t,
        offs=offsets,
        out_dtype=torch.bfloat16,
        use_cuda_kernel_for_blocked_layout=can_use_cuda_for_blocked_layout,
        wgrad_with_hp=wgrad_with_hp,
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
    wgrad_with_hp: bool = True,
    use_mxfp8_ep_primitives: bool = True,
) -> torch.Tensor:
    """
    MXFP8 optimized pipeline with chained autograd functions:
    bf16 -> a2a_dispatch (MXTensor) -> permute (MXTensor) ->
    mxfp8_grouped_mm -> unpermute -> a2a_combine -> bf16

    If use_mxfp8_ep_primitives=False, uses BF16 all2alls and permute/unpermute
    instead of the mxfp8 EP primitives.
    """
    block_size = 32

    if use_mxfp8_ep_primitives:
        # Step 1: A2A dispatch - outputs MXTensor
        mx_dispatched = a2a_dispatch_mxfp8_fwd_hp_bwd(
            input_tensor,
            output_splits_list,
            input_splits_list,
            group_name=group.group_name,
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
        scale_cols = expert_weights_t.shape[-1] // 32
        can_use_cuda_for_blocked_layout = scale_cols >= 64 and scale_cols % 16 == 0
        gemm_output = _to_mxfp8_then_scaled_grouped_mm(
            mx_permuted,
            expert_weights_t,
            offs=mx_group_offsets,
            block_size=block_size,
            use_cuda_kernel_for_blocked_layout=can_use_cuda_for_blocked_layout,
            wgrad_with_hp=wgrad_with_hp,
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
            group_name=group.group_name,
            mxfp8_bwd=True,
        )
    else:
        # Use BF16 all2alls and permute/unpermute instead of mxfp8 EP primitives
        # Step 1: All-to-all dispatch (BF16)
        dispatched = all_to_all_single(
            input_tensor,
            output_splits_list,
            input_splits_list,
            group=group,
        )
        dispatched = torch.ops._c10d_functional.wait_tensor(dispatched)

        # Step 2: Permute (BF16)
        (
            input_shape,
            permuted,
            permuted_indices,
            num_tokens_per_expert_padded,
            offsets,
        ) = _permute_bf16(
            dispatched,
            num_tokens_per_expert_group,
            ep_degree,
            num_experts,
            block_size,
        )

        # Step 3: MXFP8 Grouped MM
        scale_cols = expert_weights_t.shape[-1] // 32
        can_use_cuda_for_blocked_layout = scale_cols >= 64 and scale_cols % 16 == 0
        gemm_output = _to_mxfp8_then_scaled_grouped_mm(
            permuted,
            expert_weights_t,
            offs=offsets,
            out_dtype=torch.bfloat16,
            use_cuda_kernel_for_blocked_layout=can_use_cuda_for_blocked_layout,
            wgrad_with_hp=wgrad_with_hp,
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


def mse_loss_and_bwd(output: torch.Tensor, labels: torch.Tensor):
    """Compute MSE loss and run backward pass."""
    loss = F.mse_loss(output, labels)
    loss.backward()


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    """Run a single experiment comparing three pipelines:
    1. Standard BF16 pipeline with torch._grouped_mm
    2. MXFP8 pipeline without EP primitives (wgrad_with_hp=False)
    3. MXFP8 pipeline with EP primitives (wgrad_with_hp=True)
    """
    num_tokens = config.num_tokens
    dim = config.dim
    hidden_dim = config.hidden_dim
    num_experts = config.num_experts

    # Create input tensors for variant 1 (standard BF16)
    input_tensor_v1 = torch.randn(
        num_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    expert_weights_v1 = torch.randn(
        num_experts,
        hidden_dim,
        dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )

    # Create input tensors for variant 2 (mxfp8 no EP)
    input_tensor_v2 = input_tensor_v1.detach().clone().requires_grad_(True)
    expert_weights_v2 = expert_weights_v1.detach().clone().requires_grad_(True)

    # Create input tensors for variant 3 (mxfp8 with EP)
    input_tensor_v3 = input_tensor_v1.detach().clone().requires_grad_(True)
    expert_weights_v3 = expert_weights_v1.detach().clone().requires_grad_(True)

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

    # Helper function to create pipeline functions
    def create_pipeline_fn(pipeline_func, input_t, weight_t, **pipeline_kwargs):
        fn = lambda: pipeline_func(
            input_t,
            weight_t,
            num_tokens_per_expert,
            num_tokens_per_expert_group,
            input_splits_list,
            output_splits_list,
            ep_degree,
            num_experts,
            group,
            **pipeline_kwargs,
        )
        # Apply torch.compile if requested
        if args.compile:
            fn = torch.compile(fn)
        return fn

    # Set seed for deterministic execution
    torch.manual_seed(42)

    # === Variant 1: Standard BF16 Pipeline (torch._grouped_mm) ===
    v1_fwd_fn = create_pipeline_fn(
        standard_pipeline,
        input_tensor_v1,
        expert_weights_v1.transpose(-2, -1),
        wgrad_with_hp=True,
    )

    # Forward warmup and timing
    warmup(v1_fwd_fn)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    _ = v1_fwd_fn()
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    fwd_bf16_ms = (end_sec - start_sec) * 1e3

    # Backward warmup
    def v1_bwd_warmup():
        input_tensor_v1.grad = None
        expert_weights_v1.grad = None
        output = v1_fwd_fn()
        labels = torch.ones_like(output)
        mse_loss_and_bwd(output, labels)

    warmup(v1_bwd_warmup)

    # Backward timing
    input_tensor_v1.grad = None
    expert_weights_v1.grad = None
    v1_output_for_bwd = v1_fwd_fn()
    v1_labels = torch.ones_like(v1_output_for_bwd)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    mse_loss_and_bwd(v1_output_for_bwd, v1_labels)
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    bwd_bf16_ms = (end_sec - start_sec) * 1e3

    # === Variant 2: MXFP8 without EP primitives (wgrad_with_hp=False) ===
    torch.manual_seed(42)
    v2_fwd_fn = create_pipeline_fn(
        mxfp8_pipeline,
        input_tensor_v2,
        expert_weights_v2.transpose(-2, -1),
        wgrad_with_hp=False,
        use_mxfp8_ep_primitives=False,
    )

    # Forward warmup and timing
    warmup(v2_fwd_fn)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    _ = v2_fwd_fn()
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    fwd_mxfp8_no_ep_ms = (end_sec - start_sec) * 1e3

    # Backward warmup
    def v2_bwd_warmup():
        input_tensor_v2.grad = None
        expert_weights_v2.grad = None
        output = v2_fwd_fn()
        labels = torch.ones_like(output)
        mse_loss_and_bwd(output, labels)

    warmup(v2_bwd_warmup)

    # Backward timing
    input_tensor_v2.grad = None
    expert_weights_v2.grad = None
    v2_output_for_bwd = v2_fwd_fn()
    v2_labels = torch.ones_like(v2_output_for_bwd)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    mse_loss_and_bwd(v2_output_for_bwd, v2_labels)
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    bwd_mxfp8_no_ep_ms = (end_sec - start_sec) * 1e3

    # === Variant 3: MXFP8 with EP primitives (wgrad_with_hp=True) ===
    torch.manual_seed(42)
    v3_fwd_fn = create_pipeline_fn(
        mxfp8_pipeline,
        input_tensor_v3,
        expert_weights_v3.transpose(-2, -1),
        wgrad_with_hp=True,
        use_mxfp8_ep_primitives=True,
    )

    # Forward warmup and timing
    warmup(v3_fwd_fn)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    _ = v3_fwd_fn()
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    fwd_mxfp8_with_ep_ms = (end_sec - start_sec) * 1e3

    # Backward warmup
    def v3_bwd_warmup():
        input_tensor_v3.grad = None
        expert_weights_v3.grad = None
        output = v3_fwd_fn()
        labels = torch.ones_like(output)
        mse_loss_and_bwd(output, labels)

    warmup(v3_bwd_warmup)

    # Backward timing
    input_tensor_v3.grad = None
    expert_weights_v3.grad = None
    v3_output_for_bwd = v3_fwd_fn()
    v3_labels = torch.ones_like(v3_output_for_bwd)
    torch.cuda.synchronize()
    start_sec = time.perf_counter()
    mse_loss_and_bwd(v3_output_for_bwd, v3_labels)
    torch.cuda.synchronize()
    end_sec = time.perf_counter()
    bwd_mxfp8_with_ep_ms = (end_sec - start_sec) * 1e3

    # Calculate speedups (vs BF16 baseline)
    fwd_mxfp8_no_ep_speedup = fwd_bf16_ms / fwd_mxfp8_no_ep_ms
    bwd_mxfp8_no_ep_speedup = bwd_bf16_ms / bwd_mxfp8_no_ep_ms
    total_bf16_ms = fwd_bf16_ms + bwd_bf16_ms
    total_mxfp8_no_ep_ms = fwd_mxfp8_no_ep_ms + bwd_mxfp8_no_ep_ms
    total_mxfp8_no_ep_speedup = total_bf16_ms / total_mxfp8_no_ep_ms

    fwd_mxfp8_with_ep_speedup = fwd_bf16_ms / fwd_mxfp8_with_ep_ms
    bwd_mxfp8_with_ep_speedup = bwd_bf16_ms / bwd_mxfp8_with_ep_ms
    total_mxfp8_with_ep_ms = fwd_mxfp8_with_ep_ms + bwd_mxfp8_with_ep_ms
    total_mxfp8_with_ep_speedup = total_bf16_ms / total_mxfp8_with_ep_ms

    return ExperimentResult(
        fwd_bf16_ms=fwd_bf16_ms,
        fwd_mxfp8_no_ep_ms=fwd_mxfp8_no_ep_ms,
        fwd_mxfp8_with_ep_ms=fwd_mxfp8_with_ep_ms,
        bwd_bf16_ms=bwd_bf16_ms,
        bwd_mxfp8_no_ep_ms=bwd_mxfp8_no_ep_ms,
        bwd_mxfp8_with_ep_ms=bwd_mxfp8_with_ep_ms,
        fwd_mxfp8_no_ep_speedup=fwd_mxfp8_no_ep_speedup,
        bwd_mxfp8_no_ep_speedup=bwd_mxfp8_no_ep_speedup,
        total_mxfp8_no_ep_speedup=total_mxfp8_no_ep_speedup,
        fwd_mxfp8_with_ep_speedup=fwd_mxfp8_with_ep_speedup,
        bwd_mxfp8_with_ep_speedup=bwd_mxfp8_with_ep_speedup,
        total_mxfp8_with_ep_speedup=total_mxfp8_with_ep_speedup,
    )


def print_results(experiments: List[Experiment]):
    """Print benchmark results in a formatted table."""
    headers = [
        "tokens",
        "dim",
        "hidden",
        "experts",
        "fwd_bf16",
        "fwd_mx_noep",
        "fwd_mx_ep",
        "bwd_bf16",
        "bwd_mx_noep",
        "bwd_mx_ep",
        "mx_ep_speedup_vs_bf16",
        "mx_ep_speedup_vs_noep",
    ]
    rows = []
    for experiment in experiments:
        cfg = experiment.config
        res = experiment.result
        # Calculate speedup of MX_EP vs MX_NoEP
        total_mxfp8_no_ep_ms = res.fwd_mxfp8_no_ep_ms + res.bwd_mxfp8_no_ep_ms
        total_mxfp8_with_ep_ms = res.fwd_mxfp8_with_ep_ms + res.bwd_mxfp8_with_ep_ms
        speedup_vs_noep = total_mxfp8_no_ep_ms / total_mxfp8_with_ep_ms

        rows.append(
            [
                cfg.num_tokens,
                cfg.dim,
                cfg.hidden_dim,
                cfg.num_experts,
                f"{res.fwd_bf16_ms:.2f}",
                f"{res.fwd_mxfp8_no_ep_ms:.2f}",
                f"{res.fwd_mxfp8_with_ep_ms:.2f}",
                f"{res.bwd_bf16_ms:.2f}",
                f"{res.bwd_mxfp8_no_ep_ms:.2f}",
                f"{res.bwd_mxfp8_with_ep_ms:.2f}",
                f"{res.total_mxfp8_with_ep_speedup:.2f}x",
                f"{speedup_vs_noep:.2f}x",
            ]
        )
    print("\n" + "=" * 140)
    print("Expert Parallelism Pipeline Benchmark Results")
    print(f"World Size: {dist.get_world_size()}")
    print("Variants:")
    print("  1. bf16: Standard BF16 pipeline with torch._grouped_mm")
    print(
        "  2. mxfp8_noep: MXFP8 without EP primitives (wgrad_with_hp=False, BF16 all2alls)"
    )
    print(
        "  3. mxfp8_ep: MXFP8 with EP primitives (wgrad_with_hp=True, MXFP8 all2alls)"
    )
    print("Speedups measured for MX_EP variant:")
    print("  - speedup_vs_bf16: MX_EP total time vs BF16 total time")
    print("  - speedup_vs_noep: MX_EP total time vs MX_NoEP total time")
    print("=" * 140)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 140 + "\n")


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
