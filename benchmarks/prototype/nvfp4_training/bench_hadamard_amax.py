# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
N_SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n)
        for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    x = torch.randn(config.m, config.n, dtype=torch.bfloat16, device=device)

    time_us = benchmark_cuda_function_in_microseconds(triton_rht_amax, x)

    read_bytes = x.numel() * (torch.finfo(torch.bfloat16).bits // 8)
    gbps = (read_bytes / 1e9) / (time_us / 1e6)

    return ExperimentResult(time_us=time_us, gbps=gbps)


def print_results(experiments: List[Experiment]):
    headers = ["M", "N", "time_us", "gbps"]
    rows = [
        [e.config.m, e.config.n, round(e.result.time_us, 3), round(e.result.gbps, 3)]
        for e in experiments
    ]
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
