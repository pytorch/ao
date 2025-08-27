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
from torch.nn import functional as F
from tqdm import tqdm
from triton.testing import do_bench

from benchmarks.utils import bench_fwd_bwd_microseconds
from torchao.prototype.blockwise_fp8_training.linear import Float8BlockwiseLinear

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
    bf16_linear_us: float
    fp8_triton_linear_us: float
    fp8_scaled_mm_linear_us: float


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
    M, N, K = config.m, config.n, config.k
    inputs = torch.randn(M, K, dtype=config.out_dtype, device="cuda")
    bf16_linear = torch.nn.Linear(K, N, dtype=config.out_dtype, device="cuda")
    fp8_triton_linear = Float8BlockwiseLinear(
        K, N, dtype=config.out_dtype, device="cuda", use_triton=True
    )
    fp8_scaled_mm_linear = Float8BlockwiseLinear(
        K, N, dtype=config.out_dtype, device="cuda", use_triton=False
    )

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    def fwd_bwd(func, inputs, labels, *args, **kwargs):
        out = func(inputs, *args, **kwargs)
        loss = F.mse_loss(out, labels)
        loss.backward()
        torch.cuda.synchronize()

    # Warmup then run bf16 torch.mm
    labels = inputs.new_empty(M, N).fill_(1.0)
    warmup(fwd_bwd, bf16_linear, inputs, labels)

    bf16_linear_us = benchmark_cuda_function_in_microseconds(
        fwd_bwd, bf16_linear, inputs, labels
    )

    # Warm up then run triton bench
    warmup(
        fwd_bwd,
        fp8_triton_linear,
        inputs,
        labels,
    )

    fp8_triton_linear_us = bench_fwd_bwd_microseconds(
        fp8_triton_linear,
        inputs,
        labels=labels,
    )

    warmup(
        fwd_bwd,
        fp8_scaled_mm_linear,
        inputs,
        labels,
    )

    fp8_scaled_mm_linear_us = bench_fwd_bwd_microseconds(
        fp8_scaled_mm_linear,
        inputs,
        labels=labels,
    )

    return ExperimentResult(
        bf16_linear_us=bf16_linear_us,
        fp8_triton_linear_us=fp8_triton_linear_us,
        fp8_scaled_mm_linear_us=fp8_scaled_mm_linear_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N",
        "K",
        "out_dtype",
        "bf16_mm_linear_us",
        "fp8_triton_linear_us",
        "fp8_scaled_mm_linear_us",
    ]
    rows = []
    for experiment in experiments:
        m, n, k = experiment.config.m, experiment.config.n, experiment.config.k
        rows.append(
            [
                m,
                n,
                k,
                experiment.config.out_dtype,
                experiment.result.bf16_linear_us,
                experiment.result.fp8_triton_linear_us,
                experiment.result.fp8_scaled_mm_linear_us,
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
