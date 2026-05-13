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
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
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
    variant: str


@dataclass(frozen=True)
class ExperimentResult:
    # time
    to_mx_us: float
    cuda_2d_us: float
    cutedsl_3d_us: float
    # mem bw
    to_mx_gbps: float
    cuda_2d_gbps: float
    cutedsl_3d_gbps: float


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
    variants = ["32x1_t", "32x1_n", "32x32_n"]
    configs = []
    for shape, scaling_mode, variant in itertools.product(
        input_shapes, round_modes, variants
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                scaling_mode=scaling_mode,
                variant=variant,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    variant = config.variant
    input_shape = config.input_shape
    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    def get_quant_input(x: torch.Tensor) -> torch.Tensor:
        # The "32x1_t" benchmark row is the reviewer-requested
        # contract: feed (E, K, N) K-major expert weights directly into the
        # existing 3D 32x1 kernel.
        if variant == "32x1_t":
            return x.transpose(-2, -1)
        return x

    def using_to_mx(x: torch.Tensor) -> torch.Tensor:
        if variant == "32x1_t":
            x_t = x.transpose(-2, -1)
            s_ref, y_ref = to_mx(
                x_t.transpose(-2, -1).contiguous(),
                elem_dtype=torch.float8_e4m3fn,
                block_size=block_size,
                scaling_mode=config.scaling_mode,
            )
            return y_ref.transpose(-2, -1), s_ref.transpose(-2, -1)

        if variant == "32x1_n":
            s_ref, y_ref = to_mx(
                x.transpose(-2, -1).contiguous(),
                elem_dtype=torch.float8_e4m3fn,
                block_size=block_size,
                scaling_mode=config.scaling_mode,
            )
            return y_ref.transpose(-2, -1), s_ref.transpose(-2, -1)

        assert variant == "32x32_n"
        E, N, K = x.shape
        x_tiles = (
            x.view(E, N // block_size, block_size, K // block_size, block_size)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, N // block_size, K // block_size, block_size * block_size)
        )
        s_ref, y_tiles_ref = to_mx(
            x_tiles,
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size * block_size,
            scaling_mode=config.scaling_mode,
        )
        y_ref = (
            y_tiles_ref.view(
                E, N // block_size, K // block_size, block_size, block_size
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, N, K)
        )
        y_ref = y_ref.transpose(-2, -1).contiguous().transpose(-2, -1)
        return y_ref, s_ref

    # bench to_mx
    using_to_mx_c = torch.compile(using_to_mx)
    scales_to_mx, data_to_mx = using_to_mx_c(input_tensor)
    to_mx_time_us = benchmark_cuda_function_in_microseconds(
        using_to_mx_c,
        input_tensor,
    )

    if variant == "32x1_n":
        # bench 2d dim1 kernel then transforming to col major
        using_cuda_2d_c = torch.compile(_to_mxfp8_dim1_3d)
        using_cuda_2d_c(input_tensor)
        time_cuda_2d_us = benchmark_cuda_function_in_microseconds(
            using_cuda_2d_c,
            input_tensor,
            block_size=block_size,
            scaling_mode=config.scaling_mode,
        )
    else:
        time_cuda_2d_us = float("nan")

    quant_input = get_quant_input(input_tensor)
    scale_block_dim1 = block_size
    scale_block_dim2 = 1 if variant in ("32x1_t", "32x1_n") else block_size

    # bench 3d CuTeDSL kernel
    data_cuda_3d, scales_cuda_3d = mxfp8_quantize_cuda_3d(
        quant_input,
        block_size=block_size,
        scale_block_dim1=scale_block_dim1,
        scale_block_dim2=scale_block_dim2,
        scaling_mode=str(config.scaling_mode.value),
    )
    time_cutedsl_3d_us = benchmark_cuda_function_in_microseconds(
        mxfp8_quantize_cuda_3d,
        quant_input,
        block_size=block_size,
        scale_block_dim1=scale_block_dim1,
        scale_block_dim2=scale_block_dim2,
        scaling_mode=str(config.scaling_mode.value),
    )

    # mem bw calculations
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = quant_input.numel() * bytes_per_input_el
    write_bytes = (
        data_cuda_3d.numel() * bytes_per_output_el
        + scales_cuda_3d.numel() * bytes_per_scale_el
    )

    cutedsl_3d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_cutedsl_3d_us / 1e6)
    to_mx_gbps = ((read_bytes + write_bytes) / 1e9) / (to_mx_time_us / 1e6)
    cuda_2d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_cuda_2d_us / 1e6)
    return ExperimentResult(
        # time
        to_mx_us=to_mx_time_us,
        cuda_2d_us=time_cuda_2d_us,
        cutedsl_3d_us=time_cutedsl_3d_us,
        # mem bw
        to_mx_gbps=to_mx_gbps,
        cuda_2d_gbps=cuda_2d_gbps,
        cutedsl_3d_gbps=cutedsl_3d_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "variant",
        "cuda_2d_us",
        "cutedsl_3d_us",
        "to_mx_us",
        "cuda_2d_gbps",
        "cutedsl_3d_gbps",
        "to_mx_gbps",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                str(experiment.config.input_shape),
                str(experiment.config.scaling_mode),
                experiment.config.variant,
                experiment.result.cuda_2d_us,
                experiment.result.cutedsl_3d_us,
                experiment.result.to_mx_us,
                round(experiment.result.cuda_2d_gbps, 3),
                round(experiment.result.cutedsl_3d_gbps, 3),
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
