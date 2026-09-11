# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped RHT global-amax kernel across backends (triton, cutedsl).

Both backends compute the same two per-group amaxes in one pass over the packed A:
  col_amax[g] = max|RHT(A_g.t())|  (post-Hadamard)   row_amax[g] = max|A_g|  (plain)

Reports device kernel time (see bench_utils.kernel_time_us) for each available backend
on the DeepSeek-V3 expert-weight shapes, with the cutedsl-vs-triton speedup.

    python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax
    python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax --experts 64
"""

import argparse
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
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    SAME_BOTH_DIMS,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

BACKENDS = ("triton", "cutedsl")

RHT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)


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
    read_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def make_runner(
    backend: str,
    A: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    logical_packed_length: torch.Tensor,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped amax op, or None if unavailable."""
    psl, hidden = A.shape
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
            triton_group_rht_amax as op,
        )
    elif backend == "cutedsl":
        from torchao.prototype.moe_training.nvfp4_training._cutedsl_group_kernels_impl import (
            MAX_GROUPS,
        )

        if not cutedsl_nvfp4_kernels_available() or num_tensors > MAX_GROUPS:
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
            cutedsl_group_rht_amax as op,
        )
    else:
        raise ValueError(f"unknown backend {backend}")

    return lambda: op(
        A,
        list(RHT_SIGN_VECTOR),
        offsets,
        num_tensors,
        psl,
        hidden,
        SAME_BOTH_DIMS,
        logical_packed_length=logical_packed_length,
    )


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, M, N = config.experts, config.m, config.n
    A = torch.randn((E * M, N), dtype=torch.bfloat16, device=device)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=device) * M
    logical_packed_length = offsets[-1:]

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, A, offsets, E, logical_packed_length)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None
    # amax reads the full bfloat16 input; the 2E scalar outputs are negligible.
    return ExperimentResult(us=us, read_bytes=A.numel() * 2)


def print_results(experiments: List[Experiment]):
    headers = [
        "model",
        "projection",
        "E",
        "M",
        "N",
        "cutedsl_us",
        "triton_us",
        "speedup",
        "cutedsl_gbps",
    ]
    rows = []
    for e in experiments:
        us = e.result.us
        c, t = us.get("cutedsl"), us.get("triton")
        speedup = f"{t / c:.2f}x" if (c and t) else "n/a"
        ref = c or t
        gbps = (e.result.read_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.m,
                e.config.n,
                round(c, 2) if c else "n/a",
                round(t, 2) if t else "n/a",
                speedup,
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 amax requires SM100+")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experts",
        type=int,
        default=4,
        help="Local experts per rank. Defaults to 4: at the expert-parallel "
        "degrees these models are trained at, a rank holds a handful of experts, "
        "so the per-model M/N with a small E is the representative shape. Pass "
        "None-like large values only to probe scaling (the CuteDSL backend caps "
        "at MAX_GROUPS=64 and reports n/a above it).",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    configs = [
        ExperimentConfig(
            shape.experts,
            shape.m,
            shape.n,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=args.experts)
    ]

    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
