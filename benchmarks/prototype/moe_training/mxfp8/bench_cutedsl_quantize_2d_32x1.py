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
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_quantize_2d_32x1 import (
    mxfp8_quantize_cutedsl_2d_32x1,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    scaling_mode: str
    num_groups: int


@dataclass(frozen=True)
class ExperimentResult:
    # time
    cutedsl_blocked_us: float
    cuda_plus_rearrange_us: float
    # mem bw
    cutedsl_blocked_gbps: float
    cuda_plus_rearrange_gbps: float


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
    scaling_modes = ["rceil"]
    num_groups_list = [4, 8]
    configs = []
    for shape, scaling_mode, num_groups in itertools.product(
        input_shapes, scaling_modes, num_groups_list
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                scaling_mode=scaling_mode,
                num_groups=num_groups,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    input_shape = config.input_shape
    scaling_mode = config.scaling_mode
    num_groups = config.num_groups

    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    M, K = input_shape

    # Generate jagged offsets with multiples of 128 for K dimension
    # TODO: we use multiple of 128 here to avoid per-group padding requirement in blocked scales layout, which cutedsl doesn't support yet.
    offs = generate_jagged_offs(num_groups, K, multiple_of=128, device=device)

    # Benchmark 1: CuTeDSL kernel with blocked scale output
    data_cutedsl, scales_cutedsl = mxfp8_quantize_cutedsl_2d_32x1(
        input_tensor,
        block_size=block_size,
        scaling_mode=scaling_mode,
        blocked_scale_output=True,
    )
    cutedsl_blocked_time_us = benchmark_cuda_function_in_microseconds(
        mxfp8_quantize_cutedsl_2d_32x1,
        input_tensor,
        block_size=block_size,
        scaling_mode=scaling_mode,
        blocked_scale_output=True,
    )

    # Benchmark 2: CUDA quantization + CUDA scale rearrangement
    def cuda_plus_rearrange(x, group_offs):
        # Quantize with 32x1 scaling (rowwise=True, colwise=False)
        _, output_colwise, _, scales_colwise = mxfp8_quantize_cuda(
            x,
            rowwise=False,
            colwise=True,
            scaling_mode=scaling_mode,
        )
        # Convert scales to blocked layout for K groups
        scales_blocked = triton_mx_block_rearrange_2d_K_groups(
            scales_colwise.view(torch.uint8), group_offs // 32
        )
        return output_colwise, scales_blocked

    data_cuda, scales_cuda = cuda_plus_rearrange(input_tensor, offs)
    cuda_plus_rearrange_time_us = benchmark_cuda_function_in_microseconds(
        cuda_plus_rearrange,
        input_tensor,
        offs,
    )

    # Memory bandwidth calculations
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = (
        data_cutedsl.numel() * bytes_per_output_el
        + scales_cutedsl.numel() * bytes_per_scale_el
    )

    cutedsl_blocked_gbps = ((read_bytes + write_bytes) / 1e9) / (
        cutedsl_blocked_time_us / 1e6
    )
    cuda_plus_rearrange_gbps = ((read_bytes + write_bytes) / 1e9) / (
        cuda_plus_rearrange_time_us / 1e6
    )

    return ExperimentResult(
        cutedsl_blocked_us=cutedsl_blocked_time_us,
        cuda_plus_rearrange_us=cuda_plus_rearrange_time_us,
        cutedsl_blocked_gbps=cutedsl_blocked_gbps,
        cuda_plus_rearrange_gbps=cuda_plus_rearrange_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "num_groups",
        "cutedsl_blocked_us",
        "cuda+rearrange_us",
        "speedup",
        "cutedsl_gbps",
        "cuda+rearrange_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = (
            experiment.result.cuda_plus_rearrange_us
            / experiment.result.cutedsl_blocked_us
        )
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.scaling_mode,
                experiment.config.num_groups,
                f"{experiment.result.cutedsl_blocked_us:.2f}",
                f"{experiment.result.cuda_plus_rearrange_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.cutedsl_blocked_gbps:.1f}",
                f"{experiment.result.cuda_plus_rearrange_gbps:.1f}",
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
