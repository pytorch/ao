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
    fp8_blockwise_act_quant_lhs,
    fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x128,
)

device = torch.device("cuda")

# This benchmark requires CUDA 12.9+
assert torch.version.cuda is not None, "CUDA is not available"
cuda_major, cuda_minor = map(int, torch.version.cuda.split("."))
assert cuda_major >= 12 and cuda_minor >= 9, "CUDA 12.9+ is required"

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
    bf16_mm_us: float
    fp8_triton_us: float
    fp8_scaled_mm_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    mnk_list = [
        # Llama4 shapes
        (16640, 5120, 8192),
        (16640, 8192, 5120),
    ]
    out_dtypes = [torch.bfloat16]
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
    # Simulate `grad_input = grad_output @ weight`
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

    # Warmup then run bf16 torch.mm
    warmup(torch.mm, A, B.t())

    bf16_mm_us = benchmark_cuda_function_in_microseconds(torch.mm, A, B.t())

    # Warm up then run triton bench
    warmup(
        triton_fp8_gemm_1x128_128x128,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    fp8_triton_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_gemm_1x128_128x128,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    # Warm up then run torch bench
    # scaled_mm requires A_s and B_t_s be in column-major format
    A_s = A_s.t().contiguous().t()

    warmup(
        torch._scaled_mm,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    fp8_scaled_mm_us = benchmark_cuda_function_in_microseconds(
        torch._scaled_mm,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    return ExperimentResult(
        bf16_mm_us=bf16_mm_us,
        fp8_triton_us=fp8_triton_us,
        fp8_scaled_mm_us=fp8_scaled_mm_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N",
        "K",
        "out_dtype",
        "bf16_mm_us",
        "fp8_triton_us",
        "fp8_scaled_mm_us",
        "bf16 tflops/sec",
        "triton tflops/sec",
        "scaled_mm tflops/sec",
    ]
    rows = []
    for experiment in experiments:
        m, n, k = experiment.config.m, experiment.config.n, experiment.config.k
        flops = 2 * m * n * k
        bf16_mm_tflops_per_sec = (flops / 1e12) / (experiment.result.bf16_mm_us / 1e6)
        triton_tflops_per_sec = (flops / 1e12) / (experiment.result.fp8_triton_us / 1e6)
        scaled_mm_tflops_per_sec = (flops / 1e12) / (
            experiment.result.fp8_scaled_mm_us / 1e6
        )
        rows.append(
            [
                m,
                n,
                k,
                experiment.config.out_dtype,
                experiment.result.bf16_mm_us,
                experiment.result.fp8_triton_us,
                experiment.result.fp8_scaled_mm_us,
                bf16_mm_tflops_per_sec,
                triton_tflops_per_sec,
                scaled_mm_tflops_per_sec,
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


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
