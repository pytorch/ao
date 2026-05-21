# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
from dataclasses import dataclass
from typing import List

import torch
import triton
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
    triton_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_quantize_row_col_triton import (
    _hadamard_quantize_row_col_kernel,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_rht_matrix,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
N_SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

LLAMA_BATCH_SIZE = 1
LLAMA_SEQ_LEN = 2048
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
    m: int
    n: int
    model: str = ""
    shape: str = ""
    rounding: str = "rtne"


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


def get_configs(roundings: List[str]) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n, rounding=rounding)
        for m, n in itertools.product(M_SHAPES, N_SHAPES)
        for rounding in roundings
    ]


def get_representative_model_configs(roundings: List[str]) -> List[ExperimentConfig]:
    llama_m = LLAMA_BATCH_SIZE * LLAMA_SEQ_LEN

    shapes = [
        (llama_m, 4096, "Llama 3 8B", "hidden-state input"),
        (llama_m, 14336, "Llama 3 8B", "mlp.down input"),
        (llama_m, 8192, "Llama 3 70B", "hidden-state input"),
        (llama_m, 28672, "Llama 3 70B", "mlp.down input"),
    ]
    return [
        ExperimentConfig(m=m, n=n, model=model, shape=shape, rounding=rounding)
        for m, n, model, shape in shapes
        for rounding in roundings
    ]


def get_peak_mem_bw_gbps() -> float | None:
    props = torch.cuda.get_device_properties(device)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None

    peak_mem_bw_bytes_per_second = (
        (memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0
    )
    return peak_mem_bw_bytes_per_second / 1e9


def run_experiment(
    config: ExperimentConfig, peak_mem_bw_gbps: float | None
) -> ExperimentResult | None:
    m, n = config.m, config.n
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)

    if torch.cuda.is_available() and not is_sm_at_least_100():
        return None

    try:
        col_amax, row_amax = triton_rht_amax(x, sign_vector=list(RHT_SIGN_VECTOR))
    except NotImplementedError:
        return None

    # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch
    # space. Set it outside the timed region so the benchmark measures the
    # kernel body instead of wrapper setup.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=x.device
            )
        )

    rht_matrix = get_rht_matrix(RHT_SIGN_VECTOR, x.device, torch.bfloat16, 16)
    colwise_C = torch.empty((n, m // 2), dtype=torch.uint8, device=x.device)
    colwise_sf = torch.empty(
        (n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    rowwise_C = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    rowwise_sf = torch.empty(
        (m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    stochastic_rounding = config.rounding == "rs"

    if stochastic_rounding:
        col_seed_base = torch.randint(
            low=-(2**63),
            high=2**63 - 1,
            size=(1,),
            dtype=torch.int64,
            device=x.device,
        )
        col_offset_base = torch.randint(
            low=-(2**63),
            high=2**63 - 1,
            size=(1,),
            dtype=torch.int64,
            device=x.device,
        )
        row_seed_base = torch.randint(
            low=-(2**63),
            high=2**63 - 1,
            size=(1,),
            dtype=torch.int64,
            device=x.device,
        )
        row_offset_base = torch.randint(
            low=-(2**63),
            high=2**63 - 1,
            size=(1,),
            dtype=torch.int64,
            device=x.device,
        )
    else:
        col_seed_base = 0
        col_offset_base = 0
        row_seed_base = 0
        row_offset_base = 0

    def run_kernel():
        _hadamard_quantize_row_col_kernel[(num_sms,)](
            x,
            rht_matrix,
            col_amax,
            row_amax,
            colwise_C,
            colwise_sf,
            rowwise_C,
            rowwise_sf,
            col_seed_base,
            col_offset_base,
            row_seed_base,
            row_offset_base,
            m,
            n,
            GROUP_SIZE_N=8,
            NUM_SMS=num_sms,
            STOCHASTIC_ROUNDING=stochastic_rounding,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)

    read_bytes = m * n * 2  # bfloat16 input
    col_write = n * (m // 2) + (n // 128) * (m // 64) * 32 * 16
    row_write = m * (n // 2) + (m // 128) * (n // 64) * 32 * 16
    total_bytes = read_bytes + col_write + row_write
    gbps = (total_bytes / 1e9) / (time_us / 1e6)
    pct_peak_mem_bw = (
        gbps / peak_mem_bw_gbps * 100.0 if peak_mem_bw_gbps is not None else None
    )

    return ExperimentResult(time_us=time_us, gbps=gbps, pct_peak_mem_bw=pct_peak_mem_bw)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = ["M", "N", "rounding", "time_us", "gbps", "pct_peak_mem_bw"]
    rows = []
    for e in experiments:
        row = [
            e.config.m,
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
        default="sweep",
        help="Benchmark the original sweep or selected model-derived shapes.",
    )
    parser.add_argument(
        "--rounding",
        choices=ROUNDING_CHOICES,
        default="all",
        help="Quantization rounding mode to benchmark.",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    roundings = get_roundings(args.rounding)
    configs = (
        get_representative_model_configs(roundings)
        if args.shape_set == "representative-models"
        else get_configs(roundings)
    )
    peak_mem_bw_gbps = get_peak_mem_bw_gbps()
    if peak_mem_bw_gbps is not None:
        print(f"Peak memory bandwidth: {peak_mem_bw_gbps:.1f} GB/s")
    else:
        print("Peak memory bandwidth: n/a")

    results = []
    for config in tqdm(configs):
        result = run_experiment(config, peak_mem_bw_gbps)
        if result is not None:
            results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
