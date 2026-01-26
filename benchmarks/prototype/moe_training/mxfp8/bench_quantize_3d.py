# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import mxfp8_quantize_cuda_3d
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_mxfp8_dim1_3d,
)
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]
    scaling_mode: ScaleCalculationMode


@dataclass(frozen=True)
class ExperimentResult:
    # time
    to_mx_us: float
    cuda_2d_us: float
    cuda_3d_us: float
    # mem bw
    to_mx_gbps: float
    cuda_2d_gbps: float
    cuda_3d_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        # Llama4 and DeepSeekV3 671b shapes
        (1, 8192, 5120),
        (1, 7168, 2048),
        (8, 8192, 5120),
        (8, 7168, 2048),
        (32, 7168, 2048),
        (32, 8192, 5120),
    ]
    round_modes = [ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL]
    configs = []
    for shape, scaling_mode in itertools.product(input_shapes, round_modes):
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
    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    def using_to_mx(x: torch.Tensor) -> torch.Tensor:
        # Reference implementation
        s_d1_ref, y_d1_ref = to_mx(
            # Transpose (E,N,K) to (E,K,N) so N is final dim,
            # since to_mx scales along that dim
            x.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
        )

        # Transpose tensors and scales back so we have effectively
        # quantized input shape (E, N, K) along N
        y_d1_ref = y_d1_ref.transpose(-2, -1)
        s_d1_ref = s_d1_ref.transpose(-2, -1)
        return y_d1_ref, s_d1_ref

    # bench to_mx
    using_to_mx_c = torch.compile(using_to_mx)
    scales_to_mx, data_to_mx = using_to_mx_c(input_tensor)
    to_mx_time_us = benchmark_cuda_function_in_microseconds(
        using_to_mx_c,
        input_tensor,
    )

    # bench 2d dim1 kernel then transforming to col major
    using_cuda_2d_c = torch.compile(_to_mxfp8_dim1_3d)
    scales_cuda_2d, data_cuda_2d = using_cuda_2d_c(input_tensor)
    time_cuda_2d_us = benchmark_cuda_function_in_microseconds(
        using_cuda_2d_c,
        input_tensor,
        block_size=block_size,
        scaling_mode=config.scaling_mode,
    )

    # bench 3d cuda kernel
    data_cuda_3d, scales_cuda_3d = mxfp8_quantize_cuda_3d(input_tensor)
    time_cuda_3d_us = benchmark_cuda_function_in_microseconds(
        mxfp8_quantize_cuda_3d,
        input_tensor,
        block_size=block_size,
        scaling_mode=str(config.scaling_mode.value),
    )

    # mem bw calculations
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = (
        data_cuda_3d.numel() * bytes_per_output_el
        + scales_cuda_3d.numel() * bytes_per_scale_el
    )

    to_mx_gbps = ((read_bytes + write_bytes) / 1e9) / (to_mx_time_us / 1e6)
    cuda_2d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_cuda_2d_us / 1e6)
    cuda_3d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_cuda_3d_us / 1e6)

    return ExperimentResult(
        # time
        to_mx_us=to_mx_time_us,
        cuda_2d_us=time_cuda_2d_us,
        cuda_3d_us=time_cuda_3d_us,
        # mem bw
        to_mx_gbps=to_mx_gbps,
        cuda_2d_gbps=cuda_2d_gbps,
        cuda_3d_gbps=cuda_3d_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "cuda_3d_us",
        "cuda_2d_us",
        "to_mx_us",
        "cuda_3d_gbps",
        "cuda_2d_gbps",
        "to_mx_gbps",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                str(experiment.config.input_shape),
                str(experiment.config.scaling_mode),
                experiment.result.cuda_3d_us,
                experiment.result.cuda_2d_us,
                experiment.result.to_mx_us,
                round(experiment.result.cuda_3d_gbps, 3),
                round(experiment.result.cuda_2d_gbps, 3),
                round(experiment.result.to_mx_gbps, 3),
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
