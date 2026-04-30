# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

"""FlyDSL counterpart of ``bench_cutedsl_quantize_2d_32x1.py``.

FlyDSL FLOOR-only baseline — RCEIL, ``blocked_scale_output``, and ``offs``
are not yet implemented and would raise :class:`NotImplementedError`. The
shape grid mirrors the cutedsl bench; the comparison baseline is
``triton_to_mxfp8_dim1`` (no scale rearrange, since FlyDSL does not emit
the blocked layout).
"""

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_2d_32x1 import (
    mxfp8_quantize_flydsl_2d_32x1,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim1

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    scaling_mode: str


@dataclass(frozen=True)
class ExperimentResult:
    # time
    flydsl_us: float
    triton_us: float
    # mem bw
    flydsl_gbps: float
    triton_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        # DeepSeekV3 671b shapes
        (8192, 2048),
        (8192, 7168),
        (32768, 2048),
        (32768, 7168),
        (131072, 2048),
        (131072, 7168),
    ]
    scaling_modes = ["floor"]  # FlyDSL baseline is FLOOR-only.
    configs = []
    for shape, scaling_mode in itertools.product(input_shapes, scaling_modes):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                scaling_mode=scaling_mode,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    input_shape = config.input_shape
    scaling_mode = config.scaling_mode

    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    # Benchmark 1: FlyDSL kernel (32x1 — column-major output)
    data_flydsl, scales_flydsl = mxfp8_quantize_flydsl_2d_32x1(
        input_tensor,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )
    flydsl_time_us = benchmark_cuda_function_in_microseconds(
        mxfp8_quantize_flydsl_2d_32x1,
        input_tensor,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )

    # Benchmark 2: Triton dim1 quantization (raw, no scale rearrange).
    def triton_dim1(x):
        return triton_to_mxfp8_dim1(
            x,
            inner_block_size=block_size,
            scaling_mode=scaling_mode,
        )

    triton_dim1(input_tensor)
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_dim1,
        input_tensor,
    )

    # Memory bandwidth calculations
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = (
        data_flydsl.numel() * bytes_per_output_el
        + scales_flydsl.numel() * bytes_per_scale_el
    )

    flydsl_gbps = ((read_bytes + write_bytes) / 1e9) / (flydsl_time_us / 1e6)
    triton_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        flydsl_us=flydsl_time_us,
        triton_us=triton_time_us,
        flydsl_gbps=flydsl_gbps,
        triton_gbps=triton_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "flydsl_us",
        "triton_us",
        "speedup",
        "flydsl_gbps",
        "triton_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.triton_us / experiment.result.flydsl_us
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.scaling_mode,
                f"{experiment.result.flydsl_us:.2f}",
                f"{experiment.result.triton_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.flydsl_gbps:.1f}",
                f"{experiment.result.triton_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
