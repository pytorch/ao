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
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_triton import (
    triton_quantize_2d_weight,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
# N must be a multiple of BLOCK_N=256
N_SHAPES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

LLAMA_BATCH_SIZE = 1
LLAMA_SEQ_LEN = 2048


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int
    model: str = ""
    shape: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float


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

    return [
        ExperimentConfig(llama_m, 4096, "Llama 3 8B", "hidden-state input"),
        ExperimentConfig(llama_m, 14336, "Llama 3 8B", "mlp.down input"),
        ExperimentConfig(llama_m, 8192, "Llama 3 70B", "hidden-state input"),
        ExperimentConfig(llama_m, 28672, "Llama 3 70B", "mlp.down input"),
    ]


def run_experiment(config: ExperimentConfig) -> ExperimentResult | None:
    m, n = config.m, config.n
    x = torch.randn(m, n, dtype=torch.bfloat16, device=device)

    if torch.cuda.is_available() and not is_sm_at_least_100():
        return None

    global_amax = x.float().abs().max()

    # tl.make_tensor_descriptor requires a Triton allocator for per-CTA scratch
    # space. Set it outside the timed region so the benchmark measures the
    # kernel body instead of wrapper setup.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=x.device
            )
        )

    a_fp4 = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    a_sf = torch.empty(
        (m // 128, n // 64, 32, 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    a_t_fp4 = torch.empty((n, m // 2), dtype=torch.uint8, device=x.device)
    a_t_sf = torch.empty(
        (n // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn, device=x.device
    )
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count

    def run_kernel():
        triton_quantize_2d_weight[(num_sms,)](
            x,
            a_fp4,
            a_sf,
            a_t_fp4,
            a_t_sf,
            global_amax,
            m,
            n,
            GROUP_SIZE_N=8,
            NUM_SMS=num_sms,
        )

    time_us = benchmark_cuda_function_in_microseconds(run_kernel)

    read_bytes = m * n * 2  # bf16 input
    write_fp4 = 2 * m * (n // 2)  # rowwise and colwise packed FP4 outputs
    write_scales = 2 * m * (n // 16)  # rowwise and colwise FP8 scale factors
    gbps = ((read_bytes + write_fp4 + write_scales) / 1e9) / (time_us / 1e6)

    return ExperimentResult(time_us=time_us, gbps=gbps)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = ["M", "N", "time_us", "gbps"]
    rows = []
    for e in experiments:
        row = [
            e.config.m,
            e.config.n,
            round(e.result.time_us, 3),
            round(e.result.gbps, 3),
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
