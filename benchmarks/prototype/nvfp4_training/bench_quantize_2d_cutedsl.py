# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark cutedsl_weight_quantize_2d vs triton_weight_quantize_2d.

Both produce the rowwise (forward GEMM) and transposed colwise (dgrad GEMM) NVFP4 weight layouts,
without RHT. The CuteDSL op reuses the fused RHT kernel with an identity Hadamard operand (the
columnwise UMMA collapses to a plain transpose-quantize). Reports device kernel time.

    python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d_cutedsl
    python -m benchmarks.prototype.nvfp4_training.bench_quantize_2d_cutedsl --shape-set representative-models
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch
from tabulate import tabulate
from torch.profiler import ProfilerActivity, profile
from torch.utils._triton import has_triton
from tqdm import tqdm

from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)

device = torch.device("cuda")

# Weight A = (out_features, in_features): out (M) % 256 == 0, in (N) % 128 == 0.
M_SHAPES = [256, 512, 1024, 4096]
N_SHAPES = [256, 512, 1024, 4096]

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
    pct_peak_mem_bw: Optional[float]


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
    return [
        ExperimentConfig(m=m, n=n, model=model, shape=shape)
        for m, n, model, shape in shapes
    ]


def get_peak_mem_bw_gbps() -> Optional[float]:
    props = torch.cuda.get_device_properties(device)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None
    return ((memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0) / 1e9


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
        (
            getattr(e, "self_device_time_total", 0)
            or getattr(e, "self_cuda_time_total", 0)
        )
        for e in prof.key_averages()
        if "memcpy" not in e.key.lower() and "memset" not in e.key.lower()
    )
    return total / iters


def _weight_bytes(m: int, n: int) -> int:
    read_bytes = m * n * 2  # bfloat16 weight
    row_write = m * (n // 2) + (m // 128) * (n // 64) * 32 * 16  # W codes + swizzled SF
    col_write = (
        n * (m // 2) + (n // 128) * (m // 64) * 32 * 16
    )  # W.T codes + swizzled SF
    return read_bytes + row_write + col_write


def run_experiment(
    config: ExperimentConfig, peak_mem_bw_gbps: Optional[float]
) -> Optional[ExperimentResult]:
    if not cutedsl_nvfp4_kernels_available():
        return None
    m, n = config.m, config.n
    w = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    global_amax = w.float().abs().max()

    cutedsl_us = _kernel_us(lambda: cutedsl_weight_quantize_2d(w, global_amax))

    triton_us = None
    if has_triton():
        from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
            triton_weight_quantize_2d,
        )

        triton_us = _kernel_us(lambda: triton_weight_quantize_2d(w, global_amax))

    cutedsl_gbps = (_weight_bytes(m, n) / 1e9) / (cutedsl_us / 1e6)
    pct = (
        cutedsl_gbps / peak_mem_bw_gbps * 100.0
        if peak_mem_bw_gbps is not None
        else None
    )
    return ExperimentResult(cutedsl_us, triton_us, cutedsl_gbps, pct)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = [
        "M",
        "N",
        "cutedsl_kernel_us",
        "triton_kernel_us",
        "speedup",
        "cutedsl_gbps",
        "pct_peak_bw",
    ]
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
            round(r.pct_peak_mem_bw, 1) if r.pct_peak_mem_bw is not None else "n/a",
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
        result = run_experiment(config, peak)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
