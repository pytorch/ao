# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark cutedsl_rht_amax vs triton_rht_amax.

Both compute the same two global amaxes in one pass over A:
  col_amax = max|RHT(A.t())|  (post-Hadamard)   row_amax = max|A|  (plain)

Reports device kernel time, since NVFP4 training runs under CUDA graphs / torch.compile.

    python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax_cutedsl
    python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax_cutedsl --shape-set representative-models
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch
from tabulate import tabulate
from tqdm import tqdm
from torch.profiler import ProfilerActivity, profile
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
    cutedsl_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)

device = torch.device("cuda")

# Kernel requires M % 256 == 0, N % 128 == 0.
M_SHAPES = [256, 512, 1024, 8192]
N_SHAPES = [256, 512, 1024, 2048, 4096, 8192, 16384]

LLAMA_BATCH_SIZE = 1
LLAMA_SEQ_LEN = 2048
RHT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int
    model: str = ""
    shape: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    cutedsl_us: float
    triton_us: Optional[float]
    cutedsl_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)]


def get_representative_model_configs() -> List[ExperimentConfig]:
    llama_m = LLAMA_BATCH_SIZE * LLAMA_SEQ_LEN
    shapes = [
        (llama_m, 4096, "Llama 3 8B", "hidden-state input"),
        (llama_m, 14336, "Llama 3 8B", "mlp.down input"),
        (llama_m, 8192, "Llama 3 70B", "hidden-state input"),
        (llama_m, 28672, "Llama 3 70B", "mlp.down input"),
    ]
    return [
        ExperimentConfig(m=m, n=n, model=model, shape=shape)
        for m, n, model, shape in shapes
    ]


def _kernel_us(fn, warmup: int = 15, iters: int = 50) -> float:
    """Device kernel time per call (us): CUDA kernel self-time, averaged over `iters`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = sum(
        (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0))
        for e in prof.key_averages()
        if "memcpy" not in e.key.lower() and "memset" not in e.key.lower()
    )
    return total / iters


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    if not cutedsl_nvfp4_kernels_available():
        return None
    m, n = config.m, config.n
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    sign_vector = list(RHT_SIGN_VECTOR)

    cutedsl_us = _kernel_us(lambda: cutedsl_rht_amax(x, sign_vector))

    triton_us = None
    if has_triton():
        from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
            triton_rht_amax,
        )

        triton_us = _kernel_us(lambda: triton_rht_amax(x, sign_vector=sign_vector))

    # amax reads the full input (bfloat16); the two scalar outputs are negligible.
    cutedsl_gbps = (m * n * 2 / 1e9) / (cutedsl_us / 1e6)
    return ExperimentResult(cutedsl_us, triton_us, cutedsl_gbps)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = ["M", "N", "cutedsl_kernel_us", "triton_kernel_us", "speedup", "cutedsl_gbps"]
    rows = []
    for e in experiments:
        r = e.result
        speedup = f"{r.triton_us / r.cutedsl_us:.2f}x" if r.triton_us else "n/a"
        row = [
            e.config.m,
            e.config.n,
            round(r.cutedsl_us, 2),
            round(r.triton_us, 2) if r.triton_us else "n/a",
            speedup,
            round(r.cutedsl_gbps, 1),
        ]
        if has_labels:
            row = [e.config.model, e.config.shape] + row
        rows.append(row)
    if has_labels:
        headers = ["model", "shape"] + headers
    print(tabulate(rows, headers=headers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=("sweep", "representative-models"),
        default="representative-models",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    configs = (
        get_representative_model_configs()
        if args.shape_set == "representative-models"
        else get_configs()
    )
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
