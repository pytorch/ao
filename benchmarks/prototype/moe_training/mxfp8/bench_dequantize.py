# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.mx_formats.kernels import triton_mxfp8_dequant_dim0
from torchao.prototype.mx_formats.mx_tensor import to_dtype, to_mx

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    # time
    torch_us: float
    triton_us: float
    torch_gbps: float
    triton_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        # (local_batch_size, seq_len, dim)
        (1, 8192, 7168),
        (2, 8192, 7168),
        (4, 8192, 7168),
        (8, 8192, 7168),
    ]
    configs = []
    for shape in input_shapes:
        configs.append(
            ExperimentConfig(
                input_shape=shape,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    input_shape = config.input_shape
    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    e8m0_scales, e4m3_data = to_mx(input_tensor, torch.float8_e4m3fn, block_size)

    # Bench torch dequant
    to_dtype_c = torch.compile(to_dtype)
    elem_dtype, target_dtype = torch.float8_e4m3fn, torch.bfloat16
    torch_output = to_dtype_c(
        e4m3_data,
        e8m0_scales,
        elem_dtype,
        block_size,
        target_dtype,
    )
    torch_us = benchmark_cuda_function_in_microseconds(
        to_dtype_c,
        e4m3_data,
        e8m0_scales,
        elem_dtype,
        block_size,
        target_dtype,
    )

    # Bench triton kernel
    _ = triton_mxfp8_dequant_dim0(
        e4m3_data,
        e8m0_scales,
        target_dtype,
        block_size,
    )
    triton_us = benchmark_cuda_function_in_microseconds(
        triton_mxfp8_dequant_dim0,
        e4m3_data,
        e8m0_scales,
        target_dtype,
        block_size,
    )

    # mem bw calculations
    bytes_per_input_el = torch.finfo(elem_dtype).bits / 8
    bytes_per_output_el = torch.finfo(target_dtype).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = (
        e4m3_data.numel() * bytes_per_input_el
        + e8m0_scales.numel() * bytes_per_scale_el
    )
    write_bytes = torch_output.numel() * bytes_per_output_el

    torch_gbps = ((read_bytes + write_bytes) / 1e9) / (torch_us / 1e6)
    triton_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_us / 1e6)

    return ExperimentResult(
        torch_us=torch_us,
        triton_us=triton_us,
        triton_gbps=triton_gbps,
        torch_gbps=torch_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "torch_us",
        "triton_us",
        "torch_gbps",
        "triton_gbps",
        "triton_speedup",
    ]
    rows = []
    for experiment in experiments:
        triton_speedup = round(
            experiment.result.torch_us / experiment.result.triton_us, 3
        )
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.result.torch_us,
                experiment.result.triton_us,
                round(experiment.result.torch_gbps, 3),
                round(experiment.result.triton_gbps, 3),
                f"{triton_speedup}x",
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
