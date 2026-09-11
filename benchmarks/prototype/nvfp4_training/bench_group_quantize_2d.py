# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the dense-expert grouped NVFP4 2D weight quantize across backends.

Both backends produce the same four outputs per expert in one launch over the whole (E, M, N)
stack: rowwise FP4 codes + swizzled e4m3 block scales for the forward GEMM, and the same for
W.T for the dgrad GEMM. Reports device kernel time (see bench_utils.kernel_time_us) with the
cutedsl-vs-triton speedup.
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
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

BACKENDS = ("triton", "cutedsl")

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
    global_amax: torch.Tensor,
    num_tensors: int,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped 2D quantize, or None if unavailable."""
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_triton import (
            triton_group_weight_quantize_2d as op,
        )
    elif backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_cutedsl import (
            cutedsl_group_weight_quantize_2d as op,
        )
    else:
        raise ValueError(f"unknown backend {backend}")

    return lambda: op(weights, global_amax, num_tensors)


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, M, N = config.experts, config.m, config.n
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device=device)
    global_amax = weights.float().abs().amax(dim=(1, 2))

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, weights, global_amax, E)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None

    elements = E * M * N
    # bf16 in; rowwise + colwise FP4 codes (elements/2 bytes each) and both scale sets out.
    moved_bytes = elements * 2 + elements + 2 * elements // 16
    return ExperimentResult(us=us, moved_bytes=moved_bytes)


def print_results(experiments: List[Experiment]) -> None:
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
        gbps = (e.result.moved_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.m,
                e.config.n,
                round(c, 3) if c else "n/a",
                round(t, 3) if t else "n/a",
                speedup,
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 2D quantization requires SM100+")

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
