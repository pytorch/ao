# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

"""MXFP8 grouped GEMM on AMD gfx950: FlyDSL vs bf16 ``torch._grouped_mm``.

Same shape grid and the same jagged group offsets as
``bench_2d_3d_grouped_gemm.py``. GEMM only: both operands are quantized once,
outside the timed region, so this measures the grouped GEMM and nothing else.

    python benchmarks/prototype/moe_training/mxfp8/bench_flydsl_grouped_mm.py
"""

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import mxfp8_grouped_mm_flydsl
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx

device = torch.device("cuda")

BLOCK_SIZE = 32

# MI350X / MI355X dense MXFP8 peak: 256 CU x 2.2 GHz x 8192 FLOP/clk/CU.
PEAK_TFLOPS = 4614.0


@dataclass(frozen=True)
class ExperimentConfig:
    e: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ExperimentResult:
    bf16_us: float
    mxfp8_us: float
    bf16_tflops: float
    mxfp8_tflops: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes, matching bench_2d_3d_grouped_gemm.py.
    M = [16640]
    K = [2048, 5120, 8192]
    N = [2048, 5120, 8192]
    E = [1, 2, 4, 8]
    return [
        ExperimentConfig(e=e, m=m, n=n, k=k)
        for e, m, n, k in itertools.product(E, M, N, K)
    ]


def _to_mxfp8(t: torch.Tensor):
    """Quantize along the last dim; returns (fp8 data, E8M0 scales as uint8)."""
    scale, data = to_mx(
        t.contiguous(),
        torch.float8_e4m3fn,
        BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.RCEIL,
    )
    return data, scale.view(torch.uint8).reshape(
        *t.shape[:-1], t.shape[-1] // BLOCK_SIZE
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    e, m, n, k = config.e, config.m, config.n, config.k

    A = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    # (E, N, K) row-major; B_t is the (E, K, N) view torch._grouped_mm wants.
    B = torch.randn((e, n, k), dtype=torch.bfloat16, device=device)
    B_t = B.transpose(-2, -1)

    offs = generate_jagged_offs(e, m, multiple_of=BLOCK_SIZE, device=device)

    A_data, A_scale = _to_mxfp8(A)
    B_data, B_scale = _to_mxfp8(B)

    bf16_us = benchmark_cuda_function_in_microseconds(
        torch._grouped_mm,
        A,
        B_t,
        offs,
        out_dtype=torch.bfloat16,
    )
    mxfp8_us = benchmark_cuda_function_in_microseconds(
        mxfp8_grouped_mm_flydsl,
        A_data,
        A_scale,
        B_data,
        B_scale,
        offs,
        out_dtype=torch.bfloat16,
    )

    # Useful FLOPs only: a tile that overhangs N is charged for the overhang
    # rather than credited with it.
    flops = 2.0 * m * n * k
    return ExperimentResult(
        bf16_us=bf16_us,
        mxfp8_us=mxfp8_us,
        bf16_tflops=flops / bf16_us / 1e6,
        mxfp8_tflops=flops / mxfp8_us / 1e6,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "E",
        "M",
        "N",
        "K",
        "bf16_us",
        "mxfp8_flydsl_us",
        "speedup",
        "bf16_tflops",
        "mxfp8_flydsl_tflops",
        "pct_of_mxfp8_peak",
    ]
    rows = []
    speedups = []
    for exp in experiments:
        speedup = exp.result.bf16_us / exp.result.mxfp8_us
        speedups.append(speedup)
        rows.append(
            [
                exp.config.e,
                exp.config.m,
                exp.config.n,
                exp.config.k,
                f"{exp.result.bf16_us:.1f}",
                f"{exp.result.mxfp8_us:.1f}",
                f"{speedup:.2f}x",
                f"{exp.result.bf16_tflops:.0f}",
                f"{exp.result.mxfp8_tflops:.0f}",
                f"{100 * exp.result.mxfp8_tflops / PEAK_TFLOPS:.1f}%",
            ]
        )
    print(tabulate(rows, headers=headers))
    if speedups:
        geomean = torch.tensor(speedups).log().mean().exp().item()
        print(
            f"\nspeedup vs bf16 over {len(speedups)} shapes: "
            f"geomean {geomean:.2f}x, min {min(speedups):.2f}x, max {max(speedups):.2f}x"
        )


def main():
    torch.random.manual_seed(123)
    experiments = []
    for config in tqdm(get_configs()):
        experiments.append(Experiment(config=config, result=run_experiment(config)))
    print_results(experiments)


if __name__ == "__main__":
    main()
