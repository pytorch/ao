# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the RHT global-amax kernel across backends (triton, cutedsl).

Both backends compute the same two global amaxes in one pass over A:
  col_amax = max|RHT(A.t())|  (post-Hadamard)   row_amax = max|A|  (plain)

Reports device kernel time (see bench_utils.kernel_time_us) for each available backend
on a shared set of shapes, with the cutedsl-vs-triton speedup.

    python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax
    python -m benchmarks.prototype.nvfp4_training.bench_hadamard_amax --shape-set representative-models
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.utils._triton import has_triton
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)

device = torch.device("cuda")

BACKENDS = ("triton", "cutedsl")

# Shared shape set (satisfies both backends: M % 256 == 0, N % 128 == 0).
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
    us: Dict[str, float]  # backend -> device kernel time (us)
    read_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def get_representative_model_configs() -> List[ExperimentConfig]:
    llama_m = LLAMA_BATCH_SIZE * LLAMA_SEQ_LEN
    shapes = [
        (llama_m, 4096, "Llama 3 8B", "hidden-state input"),
        (llama_m, 14336, "Llama 3 8B", "mlp.down input"),
        (llama_m, 8192, "Llama 3 70B", "hidden-state input"),
        (llama_m, 28672, "Llama 3 70B", "mlp.down input"),
    ]
    return [ExperimentConfig(m, n, model, shape) for m, n, model, shape in shapes]


def make_runner(
    backend: str, x: torch.Tensor, sign_vector
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s amax op on ``x``, or None if unavailable."""
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
            triton_rht_amax,
        )

        return lambda: triton_rht_amax(x, sign_vector=list(sign_vector))
    if backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_cutedsl import (
            cutedsl_rht_amax,
        )

        return lambda: cutedsl_rht_amax(x, list(sign_vector))
    raise ValueError(f"unknown backend {backend}")


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    x = torch.randn(config.m, config.n, dtype=torch.bfloat16, device=device)
    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, x, RHT_SIGN_VECTOR)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None
    # amax reads the full bfloat16 input; the two scalar outputs are negligible.
    return ExperimentResult(us=us, read_bytes=config.m * config.n * 2)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = ["M", "N", "cutedsl_us", "triton_us", "speedup", "cutedsl_gbps"]
    rows = []
    for e in experiments:
        us = e.result.us
        c, t = us.get("cutedsl"), us.get("triton")
        speedup = f"{t / c:.2f}x" if (c and t) else "n/a"
        ref = c or t
        gbps = (e.result.read_bytes / 1e9) / (ref / 1e6)
        row = [
            e.config.m,
            e.config.n,
            round(c, 2) if c else "n/a",
            round(t, 2) if t else "n/a",
            speedup,
            round(gbps, 1),
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
        help="Benchmark the shape sweep or the model-derived shapes.",
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
