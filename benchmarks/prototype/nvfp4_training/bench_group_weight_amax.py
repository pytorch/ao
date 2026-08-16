# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped NVFP4 per-expert weight amax against the PyTorch reduction.

Both compute the same ``(E,)`` float32 amax over a dense ``(E, M, N)`` bf16 weight stack;
the kernel wins on memory-level parallelism, not on doing less work. Reports device kernel
time (see bench_utils.kernel_time_us).

Caveat on the reported bandwidth: ``kernel_time_us`` profiles a hot loop over one buffer
and does not flush L2, so shapes below L2 capacity read partly from cache and the absolute
TB/s is optimistic. The speedup survives it: both backends are pure-read reductions and
lose the cache in the same proportion, 1.52x hot against 1.54x flushed at 671B E=4. Read
the speedup column, and treat the bandwidth column as an upper bound.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.utils._triton import has_triton
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)

device = torch.device("cuda")

BACKENDS = ("triton", "vector_norm")

# The target deployment is high expert parallelism, so the small-E shapes are the
# representative ones; the ranking inverts at large E and misleads.
LOCAL_EXPERTS = 4


@dataclass(frozen=True)
class ExperimentConfig:
    experts: int
    m: int
    n: int
    model: str = ""
    projection: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    us: Dict[str, float]  # backend -> device kernel time (us)
    moved_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def make_runner(
    backend: str,
    weights: torch.Tensor,
    num_tensors: int,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s per-expert amax, or None if unavailable."""
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
            triton_group_weight_amax as op,
        )

        return lambda: op(weights, num_tensors)
    elif backend == "vector_norm":
        return lambda: torch.linalg.vector_norm(
            weights, ord=float("inf"), dim=(1, 2), dtype=torch.float
        )
    raise ValueError(f"unknown backend {backend}")


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, M, N = config.experts, config.m, config.n
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device=device)

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, weights, E)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None

    # Read-only: bf16 in, (E,) float32 out.
    return ExperimentResult(us=us, moved_bytes=E * M * N * 2)


def print_results(experiments: List[Experiment]) -> None:
    headers = [
        "model",
        "projection",
        "E",
        "M",
        "N",
        "vector_norm_us",
        "triton_us",
        "speedup",
        "triton_gbps",
    ]
    rows = []
    for e in experiments:
        us = e.result.us
        t, v = us.get("triton"), us.get("vector_norm")
        speedup = f"{v / t:.2f}x" if (t and v) else "n/a"
        ref = t or v
        gbps = (e.result.moved_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.m,
                e.config.n,
                round(v, 3) if v else "n/a",
                round(t, 3) if t else "n/a",
                speedup,
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Grouped NVFP4 weight amax requires CUDA")

    configs = [
        ExperimentConfig(
            shape.experts,
            shape.m,
            shape.n,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=LOCAL_EXPERTS)
    ]
    experiments = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            experiments.append(Experiment(config=config, result=result))
    print_results(experiments)


if __name__ == "__main__":
    main()
