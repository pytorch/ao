# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
from typing import List

import torch
import triton
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    BLOCK_M,
    BLOCK_N,
    VARYING_FIRST_DIM,
    _group_row_col_rht_gemm_triton_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_rht_matrix,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

ROUNDING_MODES = ("rtne", "rs")
ROUNDING_CHOICES = (*ROUNDING_MODES, "all")
RHT_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)


@dataclass(frozen=True)
class ExperimentConfig:
    num_groups: int
    m_per_group: int
    n: int
    rounding: str = "rtne"
    model: str = ""
    projection: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float
    pct_peak_mem_bw: float | None


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_roundings(rounding: str) -> List[str]:
    return list(ROUNDING_MODES if rounding == "all" else (rounding,))


def get_deepseek_v3_configs(roundings: List[str]) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            num_groups=shape.experts,
            m_per_group=shape.m,
            n=shape.n,
            rounding=rounding,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_weight_shapes()
        for rounding in roundings
    ]


def get_peak_mem_bw_gbps() -> float | None:
    props = torch.cuda.get_device_properties(device)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None
    peak = (memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0
    return peak / 1e9


def run_experiment(
    config: ExperimentConfig, peak_mem_bw_gbps: float | None
) -> ExperimentResult | None:
    if torch.cuda.is_available() and not is_sm_at_least_100():
        return None

    g, mpg, n = config.num_groups, config.m_per_group, config.n
    m = g * mpg  # total packed tokens
    A = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    B = get_rht_matrix(RHT_SIGN_VECTOR, device, torch.bfloat16, 16)

    first_dims = torch.full((g,), mpg, dtype=torch.int64, device=device)
    offsets = torch.empty((g + 1,), dtype=torch.int64, device=device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(first_dims * n, dim=0)

    # Per-group amaxes (values do not affect timing); compute cheaply from A.
    A_grouped = A.view(g, mpg, n).float().abs()
    amax_row = A_grouped.amax(dim=(1, 2)).contiguous()
    amax_col = amax_row.clone()

    qa = torch.empty((m, n // 2), dtype=torch.uint8, device=device)
    qd = torch.empty((n, m // 2), dtype=torch.uint8, device=device)
    sfa = torch.empty(
        (m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=device
    )
    sfd = torch.empty(
        (n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=device
    )

    stochastic_rounding = config.rounding == "rs"
    if stochastic_rounding:
        rng = torch.randint(-(2**63), 2**63 - 1, (4,), dtype=torch.int64, device=device)
        col_seed, col_off, row_seed, row_off = (
            rng[0:1],
            rng[1:2],
            rng[2:3],
            rng[3:4],
        )
    else:
        col_seed = col_off = row_seed = row_off = 0

    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)

    def run_kernel():
        _group_row_col_rht_gemm_triton_kernel[grid](
            A,
            B,
            qa,
            sfa,
            offsets,
            amax_row,
            amax_col,
            qd,
            sfd,
            col_seed,
            col_off,
            row_seed,
            row_off,
            m,
            n,
            num_tensors=g,
            STOCHASTIC_ROUNDING=stochastic_rounding,
            SHAPE_REP=VARYING_FIRST_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=3,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)

    read_bytes = m * n * 2  # bfloat16 input
    col_write = n * (m // 2) + n * (m // 16)  # fp4 codes + fp8 scales
    row_write = m * (n // 2) + m * (n // 16)
    total_bytes = read_bytes + col_write + row_write
    gbps = (total_bytes / 1e9) / (time_us / 1e6)
    pct_peak_mem_bw = (
        gbps / peak_mem_bw_gbps * 100.0 if peak_mem_bw_gbps is not None else None
    )
    return ExperimentResult(time_us=time_us, gbps=gbps, pct_peak_mem_bw=pct_peak_mem_bw)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model for e in experiments)
    headers = [
        "groups",
        "m/group",
        "N",
        "rounding",
        "time_us",
        "gbps",
        "pct_peak_mem_bw",
    ]
    rows = []
    for e in experiments:
        row = [
            e.config.num_groups,
            e.config.m_per_group,
            e.config.n,
            e.config.rounding,
            round(e.result.time_us, 3),
            round(e.result.gbps, 3),
            (
                round(e.result.pct_peak_mem_bw, 2)
                if e.result.pct_peak_mem_bw is not None
                else "n/a"
            ),
        ]
        if has_labels:
            row = [e.config.model, e.config.projection] + row
        rows.append(row)
    if has_labels:
        headers = ["model", "projection"] + headers
    print(tabulate(rows, headers=headers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounding",
        choices=ROUNDING_CHOICES,
        default="all",
        help="Quantization rounding mode to benchmark.",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    roundings = get_roundings(args.rounding)
    configs = get_deepseek_v3_configs(roundings)
    peak_mem_bw_gbps = get_peak_mem_bw_gbps()
    print(
        f"Peak memory bandwidth: {peak_mem_bw_gbps:.1f} GB/s"
        if peak_mem_bw_gbps
        else "Peak memory bandwidth: n/a"
    )

    results = []
    for config in tqdm(configs):
        result = run_experiment(config, peak_mem_bw_gbps)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
