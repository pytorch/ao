# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Benchmarks triton_fp8_per_group_colwise_scales_dual vs two sequential
# triton_fp8_per_group_colwise_scales calls, mirroring the MoE backward pass
# where both padded_grad_output and padded_A are quantized colwise.

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    triton_fp8_per_group_colwise_scales,
    triton_fp8_per_group_colwise_scales_dual,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    M: int  # total tokens (rows shared by both tensors)
    N1: int  # cols of tensor 1 (grad_output hidden dim)
    N2: int  # cols of tensor 2 (A hidden dim)
    n_groups: int


@dataclass(frozen=True)
class ExperimentResult:
    two_calls_time_us: float
    dual_time_us: float
    speedup: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Representative MoE backward shapes:
    #   M  = total padded tokens across all experts
    #   N1 = grad_output hidden dim (output of expert)
    #   N2 = A hidden dim (input to expert)
    #   n_groups = num experts
    shapes = [
        # (M,   N1,   N2,  n_groups)
        (16640, 2048, 2048, 64),
        (16640, 5120, 2048, 64),
        (16640, 5120, 5120, 64),
        (16640, 8192, 2048, 64),
        (16640, 2048, 2048, 128),
        (16640, 5120, 2048, 128),
        (16640, 5120, 5120, 128),
        (32768, 5120, 5120, 128),
    ]
    configs = []
    for (M, N1, N2, n_groups), dtype in itertools.product(shapes, [torch.bfloat16]):
        configs.append(
            ExperimentConfig(
                high_precision_dtype=dtype,
                M=M,
                N1=N1,
                N2=N2,
                n_groups=n_groups,
            )
        )
    return configs


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    dtype = config.high_precision_dtype
    offs = generate_jagged_offs(config.n_groups, config.M, multiple_of=16)

    # tensor 1: padded_grad_output shape (M, N1), row-major
    t1 = torch.randn(config.M, config.N1, dtype=dtype, device=device)
    # tensor 2: padded_A shape (M, N2), row-major
    t2 = torch.randn(config.M, config.N2, dtype=dtype, device=device)

    fp8_dtype = torch.float8_e4m3fn

    # --- Baseline: two sequential calls (before optimization) ---
    def run_two_calls():
        triton_fp8_per_group_colwise_scales(
            t1, offs, output_dtype=fp8_dtype, round_scales_to_power_of_2=True
        )
        triton_fp8_per_group_colwise_scales(
            t2, offs, output_dtype=fp8_dtype, round_scales_to_power_of_2=True
        )

    # --- Optimized: single dual call ---
    def run_dual():
        triton_fp8_per_group_colwise_scales_dual(
            t1, t2, offs, output_dtype=fp8_dtype, round_scales_to_power_of_2=True
        )

    # Warmup
    for _ in range(10):
        run_two_calls()
    for _ in range(10):
        run_dual()

    two_calls_time_us = benchmark_cuda_function_in_microseconds(run_two_calls)
    dual_time_us = benchmark_cuda_function_in_microseconds(run_dual)

    return ExperimentResult(
        two_calls_time_us=two_calls_time_us,
        dual_time_us=dual_time_us,
        speedup=two_calls_time_us / dual_time_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N1",
        "N2",
        "n_groups",
        "dtype",
        "two calls (us)",
        "dual (us)",
        "speedup",
    ]
    rows = []
    for e in experiments:
        c, r = e.config, e.result
        rows.append(
            [
                c.M,
                c.N1,
                c.N2,
                c.n_groups,
                str(c.high_precision_dtype).split(".")[-1],
                f"{r.two_calls_time_us:.1f}",
                f"{r.dual_time_us:.1f}",
                f"{r.speedup:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers))
    print()
    speedups = [e.result.speedup for e in experiments]
    print(
        f"dual vs two calls — avg: {sum(speedups) / len(speedups):.2f}x  "
        f"min: {min(speedups):.2f}x  max: {max(speedups):.2f}x"
    )


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print()
    print("=" * 70)
    print("Dual Colwise FP8 Scales Kernel Benchmark")
    print()
    print("  two calls : triton_fp8_per_group_colwise_scales called twice")
    print("              (baseline — backward pass before optimization)")
    print("  dual      : triton_fp8_per_group_colwise_scales_dual — single launch")
    print("              merges row loops for both tensors")
    print("=" * 70)
    print()
    print_results(results)


if __name__ == "__main__":
    main()
