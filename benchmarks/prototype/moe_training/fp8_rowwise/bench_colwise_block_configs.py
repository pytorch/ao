# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Sweeps (BLOCK_SIZE, BLOCK_SIZE_ITER, num_warps) for the colwise FP8 kernel
# on representative DeepSeek-MoE-16B training shapes (MI300X).
#
# Usage:
#   cd ~/ao
#   python benchmarks/prototype/moe_training/fp8_rowwise/bench_colwise_block_configs.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
import triton
from tabulate import tabulate
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    _triton_fp8_per_group_colwise_scales_kernel,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

BLOCK_SIZES = [32, 64, 128, 256]
BLOCK_SIZE_ITERS = [32, 64, 128, 256]
NUM_WARPS_LIST = [4, 8]

SHAPES = [
    dict(M=16640, K=2048, n_groups=64, label="grad_out  M=16640 K=2048  E=64"),
    dict(M=16640, K=5120, n_groups=64, label="grad_out  M=16640 K=5120  E=64"),
    dict(M=16640, K=2048, n_groups=128, label="A         M=16640 K=2048  E=128"),
    dict(M=16640, K=5120, n_groups=128, label="A         M=16640 K=5120  E=128"),
]


@dataclass(frozen=True)
class ExperimentConfig:
    M: int
    K: int
    n_groups: int
    block_size: int
    block_size_iter: int
    num_warps: int
    label: str


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    is_hip = torch.version.hip is not None
    fp8_dtype = torch.float8_e4m3fnuz if is_hip else torch.float8_e4m3fn

    import triton.language as tl

    tl_fp8_dtype = tl.float8e4b8 if is_hip else tl.float8e4nv
    tl_input_dtype = tl.bfloat16

    M, K, n_groups = config.M, config.K, config.n_groups
    bs, bsi, nw = config.block_size, config.block_size_iter, config.num_warps

    inp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(n_groups, M, multiple_of=16)
    fp8_min = torch.finfo(fp8_dtype).min
    fp8_max = torch.finfo(fp8_dtype).max

    def run():
        out = torch.empty_like(inp, dtype=fp8_dtype).as_strided(inp.size(), (1, M))
        sc = torch.empty(K * n_groups, dtype=torch.float32, device=device)
        grid = (triton.cdiv(K, bs), n_groups)
        _triton_fp8_per_group_colwise_scales_kernel[grid](
            inp,
            offs,
            out,
            sc,
            M,
            K,
            n_groups,
            inp.stride(0),
            inp.stride(1),
            out.stride(0),
            out.stride(1),
            fp8_min,
            fp8_max,
            tl_input_dtype,
            tl_fp8_dtype,
            round_scales_to_power_of_2=True,
            BLOCK_SIZE=bs,
            BLOCK_SIZE_ITER=bsi,
            EPS=1e-12,
            num_warps=nw,
            num_stages=2,
        )
        return out, sc

    time_us = benchmark_cuda_function_in_microseconds(run)
    return ExperimentResult(time_us=time_us)


def get_configs() -> List[ExperimentConfig]:
    configs = []
    for shape, (bs, bsi, nw) in itertools.product(
        SHAPES, itertools.product(BLOCK_SIZES, BLOCK_SIZE_ITERS, NUM_WARPS_LIST)
    ):
        configs.append(
            ExperimentConfig(
                M=shape["M"],
                K=shape["K"],
                n_groups=shape["n_groups"],
                block_size=bs,
                block_size_iter=bsi,
                num_warps=nw,
                label=shape["label"],
            )
        )
    return configs


def print_results(experiments: List[Experiment]):
    # Group by shape label, print best config per shape
    from collections import defaultdict

    by_label = defaultdict(list)
    for exp in experiments:
        by_label[exp.config.label].append(exp)

    print("\n=== Best config per shape ===")
    summary_rows = []
    for label, exps in by_label.items():
        best = min(exps, key=lambda e: e.result.time_us)
        summary_rows.append(
            [
                label,
                best.config.block_size,
                best.config.block_size_iter,
                best.config.num_warps,
                f"{best.result.time_us:.1f}",
            ]
        )
    print(
        tabulate(
            summary_rows,
            headers=["shape", "BLOCK_SIZE", "BLOCK_SIZE_ITER", "num_warps", "time_us"],
        )
    )

    print("\n=== All results ===")
    all_rows = []
    for exp in experiments:
        all_rows.append(
            [
                exp.config.label,
                exp.config.block_size,
                exp.config.block_size_iter,
                exp.config.num_warps,
                f"{exp.result.time_us:.1f}",
            ]
        )
    print(
        tabulate(
            all_rows,
            headers=["shape", "BLOCK_SIZE", "BLOCK_SIZE_ITER", "num_warps", "time_us"],
        )
    )


def main():
    print(f"GPU : {torch.cuda.get_device_name()}")
    print(f"ROCm: {torch.version.hip}")
    print()

    configs = get_configs()
    results = []
    for config in configs:
        result = run_experiment(config)
        print(
            f"  {config.label}  BS={config.block_size:3d} BSI={config.block_size_iter:3d} "
            f"warps={config.num_warps}  {result.time_us:.1f} us",
            flush=True,
        )
        results.append(Experiment(config=config, result=result))

    print_results(results)


if __name__ == "__main__":
    main()
