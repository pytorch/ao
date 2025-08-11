# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.blockwise_fp8_training.kernels import (
    blockwise_fp8_gemm_1x128_128x128,
    fp8_blockwise_act_quant_lhs,
    fp8_blockwise_weight_quant_transposed_rhs,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    out_dtype: torch.dtype
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ExperimentResult:
    triton_time_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    mnk_list = [
        (16640, 8192, 5120),
        (16640, 5120, 8192),
    ]
    out_dtypes = [torch.float32, torch.bfloat16]
    configs = []
    for mnk, out_dtype in itertools.product(mnk_list, out_dtypes):
        m, n, k = mnk
        configs.append(
            ExperimentConfig(
                out_dtype=out_dtype,
                m=m,
                n=n,
                k=k,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # define test inputs
    # Simulate output = input @ weight.T
    M, N, K = config.m, config.n, config.k
    A = torch.randn(M, K, dtype=config.out_dtype, device="cuda")
    B = torch.randn(N, K, dtype=config.out_dtype, device="cuda")
    A_q, A_s = fp8_blockwise_act_quant_lhs(A, dtype=torch.float8_e4m3fn)
    B_t_q, B_t_s = fp8_blockwise_weight_quant_transposed_rhs(
        B, dtype=torch.float8_e4m3fn
    )

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    # Warm up then run triton bench
    warmup(
        blockwise_fp8_gemm_1x128_128x128,
        A_q,
        1.0 / A_s,
        B_t_q,
        1.0 / B_t_s,
    )

    triton_time_us = benchmark_cuda_function_in_microseconds(
        blockwise_fp8_gemm_1x128_128x128,
        A_q,
        1.0 / A_s,
        B_t_q,
        1.0 / B_t_s,
    )

    return ExperimentResult(
        triton_time_us=triton_time_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N",
        "K",
        "out_dtype",
        "triton_time_us",
        "tflops/sec",
    ]
    rows = []
    for experiment in experiments:
        m, n, k = experiment.config.m, experiment.config.n, experiment.config.k
        flops = 2 * m * n * k
        seconds = experiment.result.triton_time_us / 1e6
        tflops_per_sec = (flops / seconds) / 1e12
        rows.append(
            [
                m,
                n,
                k,
                experiment.config.out_dtype,
                experiment.result.triton_time_us,
                tflops_per_sec,
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
