# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# Benchmark for the Triton-based GPU-resident CUDAAllocator.
#
# Usage:
#   python benchmarks/prototype/moe_training/mxfp8/bench_cuda_allocator.py
#
######################################################################

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.ep.syncless.cuda_allocator import CUDAAllocator

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    pool_size: int  # total pool capacity (in "rows")
    alloc_size: int  # elements per alloc() call
    num_pools: int  # number of pools (1 = GPU only, 2 = GPU + CPU)


@dataclass(frozen=True)
class ExperimentResult:
    alloc_us: float  # median alloc() time in microseconds
    free_us: float  # median free() time in microseconds
    alloc_free_cycle_us: float  # median alloc+free round-trip
    multi_alloc_us: float  # median time for 10 sequential allocs
    multi_free_us: float  # median time for freeing all 10


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    """Generate experiment configurations."""
    pool_sizes = [10000, 100000, 1000000]
    alloc_sizes = [100, 1000, 10000]
    num_pools_list = [1, 2]

    configs = []
    for pool_size, alloc_size, num_pools in itertools.product(
        pool_sizes, alloc_sizes, num_pools_list
    ):
        # Skip configs where alloc_size > pool_size
        if alloc_size > pool_size:
            continue
        configs.append(
            ExperimentConfig(
                pool_size=pool_size,
                alloc_size=alloc_size,
                num_pools=num_pools,
            )
        )
    return configs


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    """Run a single benchmark experiment."""
    pool_sizes = [config.pool_size] * config.num_pools
    sz_tensor = torch.tensor(config.alloc_size, dtype=torch.int64, device=device)

    # Pre-create the allocator ONCE — benchmarks measure only kernel launches.
    allocator = CUDAAllocator(torch.empty([], device=device), pool_sizes)

    # Pre-allocate output tensor to avoid torch.empty overhead in the hot loop.
    out = torch.empty([], dtype=torch.int64, device=device)

    # --- Single alloc benchmark ---
    def bench_alloc():
        allocator.free_all()
        return allocator.alloc(sz_tensor)

    # Warmup (triggers Triton JIT compilation)
    for _ in range(5):
        bench_alloc()
    torch.cuda.synchronize()

    alloc_us = benchmark_cuda_function_in_microseconds(bench_alloc)

    # --- Single free benchmark ---
    # Alloc once, then measure repeated free of the same addr.
    # free of an unknown addr is a no-op scan, which is representative
    # of the kernel cost.
    allocator.free_all()
    addr_for_free = allocator.alloc(sz_tensor)

    def bench_free():
        allocator.free(addr_for_free)

    for _ in range(3):
        bench_free()
    torch.cuda.synchronize()

    free_us = benchmark_cuda_function_in_microseconds(bench_free)

    # --- Alloc + free cycle benchmark ---
    def bench_alloc_free_cycle():
        allocator.free_all()
        addr = allocator.alloc(sz_tensor)
        allocator.free(addr)

    alloc_free_cycle_us = benchmark_cuda_function_in_microseconds(
        bench_alloc_free_cycle
    )

    # --- Multi-alloc benchmark (10 sequential allocs) ---
    num_multi = 10
    multi_alloc_sz = min(config.alloc_size, config.pool_size // num_multi)
    multi_sz_tensor = torch.tensor(multi_alloc_sz, dtype=torch.int64, device=device)

    def bench_multi_alloc():
        allocator.free_all()
        addrs = []
        for _ in range(num_multi):
            addrs.append(allocator.alloc(multi_sz_tensor))
        return addrs

    multi_alloc_us = benchmark_cuda_function_in_microseconds(bench_multi_alloc)

    # --- Multi-free benchmark (alloc 10 then free all 10) ---
    def bench_multi_alloc_free():
        allocator.free_all()
        addrs = []
        for _ in range(num_multi):
            addrs.append(allocator.alloc(multi_sz_tensor))
        for addr in reversed(addrs):
            allocator.free(addr)

    multi_free_us = benchmark_cuda_function_in_microseconds(bench_multi_alloc_free)

    return ExperimentResult(
        alloc_us=alloc_us,
        free_us=free_us,
        alloc_free_cycle_us=alloc_free_cycle_us,
        multi_alloc_us=multi_alloc_us,
        multi_free_us=multi_free_us,
    )


def print_results(experiments: List[Experiment]):
    """Print benchmark results in a formatted table."""
    headers = [
        "pool_size",
        "alloc_size",
        "pools",
        "alloc_μs",
        "free_μs",
        "cycle_μs",
        "10x_alloc_μs",
        "10x_free_μs",
    ]
    rows = []
    for experiment in experiments:
        cfg = experiment.config
        res = experiment.result
        rows.append(
            [
                f"{cfg.pool_size:,}",
                f"{cfg.alloc_size:,}",
                cfg.num_pools,
                f"{res.alloc_us:.1f}",
                f"{res.free_us:.1f}",
                f"{res.alloc_free_cycle_us:.1f}",
                f"{res.multi_alloc_us:.1f}",
                f"{res.multi_free_us:.1f}",
            ]
        )
    print("\n" + "=" * 100)
    print("CUDAAllocator (Triton) Benchmark Results")
    print("All times are median wall-clock μs (lower is better).")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 100 + "\n")


def main(args: argparse.Namespace):
    """Main benchmark entry point."""
    torch.random.manual_seed(123)

    configs = get_configs()
    results = []
    for config in tqdm(configs, desc="Running experiments"):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the Triton-based GPU-resident CUDAAllocator"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling with PyTorch profiler",
    )
    args = parser.parse_args()
    main(args)
