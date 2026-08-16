# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped fused RHT + NVFP4 quantize across backends (triton, cutedsl).

Both backends consume the per-group amaxes from the grouped amax kernel and write
rowwise flat buffers plus columnwise per-group views over one flat columnwise buffer.

Reports device kernel time (see bench_utils.kernel_time_us) for each available backend
on the DeepSeek-V3 expert-weight shapes, with the cutedsl-vs-triton speedup.

    python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col
    python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col \
        --experts 64 --rounding rs
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
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

BACKENDS = ("triton", "cutedsl")

ROUNDING_MODES = ("rtne", "rs")
ROUNDING_CHOICES = (*ROUNDING_MODES, "all")

RHT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)


@dataclass(frozen=True)
class ExperimentConfig:
    experts: int
    m: int
    n: int
    rounding: str = "rtne"
    model: str = ""
    projection: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    us: Dict[str, float]  # backend -> device kernel time (us)
    total_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_roundings(rounding: str) -> List[str]:
    return list(ROUNDING_MODES if rounding == "all" else (rounding,))


def get_peak_mem_bw_gbps() -> Optional[float]:
    props = torch.cuda.get_device_properties(device)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None
    peak = (memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0
    return peak / 1e9


def make_runner(
    backend: str,
    A: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    row_amax: torch.Tensor,
    col_amax: torch.Tensor,
    rng_state: Optional[torch.Tensor],
    stochastic_rounding: bool,
    logical_packed_length: torch.Tensor,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped quantize op, or None if unavailable."""
    psl, hidden = A.shape
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
            triton_group_rht_quantize_row_col as op,
        )
    elif backend == "cutedsl":
        from torchao.prototype.moe_training.nvfp4_training._cutedsl_group_kernels_impl import (
            MAX_GROUPS,
        )

        if not cutedsl_nvfp4_kernels_available() or num_tensors > MAX_GROUPS:
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
            cutedsl_group_rht_quantize_row_col as op,
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
        VARYING_FIRST_DIM,
        row_amax,
        col_amax,
        rng_state,
        stochastic_rounding,
        logical_packed_length=logical_packed_length,
    )


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, M, N = config.experts, config.m, config.n
    m = E * M  # total packed tokens
    A = torch.randn((m, N), dtype=torch.bfloat16, device=device)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=device) * M
    logical_packed_length = offsets[-1:]

    # Per-group amaxes (values do not affect timing); compute cheaply from A.
    row_amax = A.view(E, M, N).float().abs().amax(dim=(1, 2)).contiguous()
    col_amax = row_amax.clone()

    stochastic_rounding = config.rounding == "rs"
    rng_state = (
        torch.randint(-(2**63), 2**63 - 1, (4,), dtype=torch.int64, device=device)
        if stochastic_rounding
        else None
    )

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(
            backend,
            A,
            offsets,
            E,
            row_amax,
            col_amax,
            rng_state,
            stochastic_rounding,
            logical_packed_length,
        )
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None

    read_bytes = m * N * 2  # bfloat16 input
    col_write = N * (m // 2) + N * (m // 16)  # fp4 codes + fp8 scales
    row_write = m * (N // 2) + m * (N // 16)
    return ExperimentResult(us=us, total_bytes=read_bytes + col_write + row_write)


def print_results(experiments: List[Experiment], peak_mem_bw_gbps: Optional[float]):
    headers = [
        "model",
        "projection",
        "E",
        "M",
        "N",
        "rounding",
        "cutedsl_us",
        "triton_us",
        "speedup",
        "cutedsl_gbps",
        "pct_peak",
    ]
    rows = []
    for e in experiments:
        us = e.result.us
        c, t = us.get("cutedsl"), us.get("triton")
        speedup = f"{t / c:.2f}x" if (c and t) else "n/a"
        ref = c or t
        gbps = (e.result.total_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.m,
                e.config.n,
                e.config.rounding,
                round(c, 2) if c else "n/a",
                round(t, 2) if t else "n/a",
                speedup,
                round(gbps, 1),
                (
                    round(gbps / peak_mem_bw_gbps * 100.0, 2)
                    if peak_mem_bw_gbps
                    else "n/a"
                ),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 quantization requires SM100+")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounding",
        choices=ROUNDING_CHOICES,
        default="all",
        help="Quantization rounding mode to benchmark.",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=4,
        help="Local experts per rank. Defaults to 4: at the expert-parallel "
        "degrees these models are trained at, a rank holds a handful of experts, "
        "so the per-model M/N with a small E is the representative shape. Pass "
        "larger values only to probe scaling (the CuteDSL backend caps at "
        "MAX_GROUPS=64 and reports n/a above it).",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    configs = [
        ExperimentConfig(
            shape.experts,
            shape.m,
            shape.n,
            rounding=rounding,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=args.experts)
        for rounding in get_roundings(args.rounding)
    ]

    peak_mem_bw_gbps = get_peak_mem_bw_gbps()
    print(
        f"Peak memory bandwidth: {peak_mem_bw_gbps:.1f} GB/s"
        if peak_mem_bw_gbps
        else "Peak memory bandwidth: n/a"
    )

    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results, peak_mem_bw_gbps)


if __name__ == "__main__":
    main()
