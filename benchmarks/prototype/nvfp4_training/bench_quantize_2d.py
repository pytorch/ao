# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the 2D NVFP4 weight quantize across backends (triton, cutedsl).

Both produce the rowwise (forward GEMM) and transposed colwise (dgrad GEMM) NVFP4 weight
layouts, without RHT, using canonical 1x16 block scales. Reports device kernel time.

    python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d
    python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d --shape-set representative-models
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

# Weight A = (out_features, in_features): out (M) % 256 == 0, in (N) % 128 == 0.
M_SHAPES = [256, 512, 1024, 4096]
N_SHAPES = [256, 512, 1024, 4096]


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int
    model: str = ""
    shape: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    us: Dict[str, float]  # backend -> device kernel time (us)
    write_read_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def get_representative_model_configs() -> List[ExperimentConfig]:
    # Weight shapes (out_features, in_features) for the MLP projections.
    shapes = [
        (14336, 4096, "Llama 3 8B", "mlp.gate/up weight"),
        (4096, 14336, "Llama 3 8B", "mlp.down weight"),
        (28672, 8192, "Llama 3 70B", "mlp.gate/up weight"),
        (8192, 28672, "Llama 3 70B", "mlp.down weight"),
    ]
    return [ExperimentConfig(m, n, model, shape) for m, n, model, shape in shapes]


def get_peak_mem_bw_gbps() -> Optional[float]:
    props = torch.cuda.get_device_properties(device)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None
    return ((memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0) / 1e9


def _weight_bytes(m: int, n: int) -> int:
    read_bytes = m * n * 2  # bfloat16 weight
    row_write = m * (n // 2) + (m // 128) * (n // 64) * 32 * 16  # W codes + swizzled SF
    col_write = (
        n * (m // 2) + (n // 128) * (m // 64) * 32 * 16
    )  # W.T codes + swizzled SF
    return read_bytes + row_write + col_write


def make_runner(
    backend: str, w: torch.Tensor, global_amax: torch.Tensor
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s weight quantize on ``w``, or None if unavailable."""
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
            triton_weight_quantize_2d,
        )

        return lambda: triton_weight_quantize_2d(w, global_amax)
    if backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
            cutedsl_weight_quantize_2d,
        )

        return lambda: cutedsl_weight_quantize_2d(w, global_amax)
    raise ValueError(f"unknown backend {backend}")


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    m, n = config.m, config.n
    w = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    global_amax = w.float().abs().max()
    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, w, global_amax)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None
    return ExperimentResult(us=us, write_read_bytes=_weight_bytes(m, n))


def print_results(experiments: List[Experiment], peak_mem_bw_gbps: Optional[float]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = [
        "M",
        "N",
        "cutedsl_us",
        "triton_us",
        "speedup",
        "cutedsl_gbps",
        "pct_peak_bw",
    ]
    rows = []
    for e in experiments:
        us = e.result.us
        c, t = us.get("cutedsl"), us.get("triton")
        speedup = f"{t / c:.2f}x" if (c and t) else "n/a"
        ref = c or t
        gbps = (e.result.write_read_bytes / 1e9) / (ref / 1e6)
        pct = gbps / peak_mem_bw_gbps * 100.0 if peak_mem_bw_gbps else None
        row = [
            e.config.m,
            e.config.n,
            round(c, 2) if c else "n/a",
            round(t, 2) if t else "n/a",
            speedup,
            round(gbps, 1),
            round(pct, 1) if pct is not None else "n/a",
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
    peak = get_peak_mem_bw_gbps()
    print(
        f"Peak memory bandwidth: {peak:.1f} GB/s"
        if peak
        else "Peak memory bandwidth: n/a"
    )

    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results, peak)


if __name__ == "__main__":
    main()
