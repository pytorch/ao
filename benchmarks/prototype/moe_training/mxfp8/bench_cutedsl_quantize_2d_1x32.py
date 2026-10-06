# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
import os
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mx_block_rearrange_2d_M_groups_cuda,
)
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_quantize_2d_1x32 import (
    mxfp8_quantize_cutedsl_2d_1x32,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import (
    triton_mx_block_rearrange,
    triton_to_mxfp8_dim0,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx

device = torch.device("cuda")
VALIDATE = os.environ.get("MXFP8_BENCH_VALIDATE", "0") == "1"
INPUT_MODE = os.environ.get("MXFP8_BENCH_INPUT_MODE", "randn")
EXTRA_NON_NICE = os.environ.get("MXFP8_BENCH_EXTRA_NON_NICE", "0") == "1"

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
    triton_plus_rearrange_us: float
    # mem bw
    cutedsl_blocked_gbps: float
    triton_plus_rearrange_gbps: float


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
    if EXTRA_NON_NICE:
        input_shapes += [
            (128, 128),
            (384, 1408),
            (1152, 3456),
        ]
    scaling_modes = ["floor", "rceil"]
    num_groups_list = [8]
    if EXTRA_NON_NICE:
        num_groups_list = [1, 4, 8]
    configs = []
    for shape, scaling_mode, num_groups in itertools.product(
        input_shapes, scaling_modes, num_groups_list
    ):
        if num_groups > shape[0] // 128:
            continue
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                scaling_mode=scaling_mode,
                num_groups=num_groups,
            )
        )
    return configs


def make_input(shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    if INPUT_MODE == "randn":
        return torch.randn(*shape, dtype=dtype, device=device)

    numel = shape[0] * shape[1]
    if INPUT_MODE == "zeros":
        x = torch.zeros(numel, dtype=torch.float32, device=device)
    elif INPUT_MODE == "arange_signed":
        x = (torch.arange(numel, dtype=torch.float32, device=device) % 257) - 128
    elif INPUT_MODE == "wide_dynamic":
        base = torch.arange(numel, dtype=torch.float32, device=device)
        signs = torch.where(base.to(torch.int64) % 2 == 0, 1.0, -1.0)
        x = signs * torch.pow(2.0, (base % 17) - 8)
    elif INPUT_MODE == "near_saturation":
        base = torch.arange(numel, dtype=torch.float32, device=device)
        vals = torch.tensor(
            [-448.0, -127.5, -1.0, 0.0, 1.0, 127.5, 448.0],
            dtype=torch.float32,
            device=device,
        )
        x = vals[base.to(torch.int64) % vals.numel()]
    else:
        raise ValueError(f"unknown MXFP8_BENCH_INPUT_MODE={INPUT_MODE}")
    return x.reshape(shape).to(dtype)


def validate_outputs(
    x: torch.Tensor,
    data_cutedsl: torch.Tensor,
    scales_cutedsl: torch.Tensor,
    scaling_mode: str,
):
    scale_mode = (
        ScaleCalculationMode.FLOOR
        if scaling_mode == "floor"
        else ScaleCalculationMode.RCEIL
    )
    scales_ref, data_ref = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=scale_mode,
    )
    scales_ref = triton_mx_block_rearrange(scales_ref)
    torch.testing.assert_close(data_cutedsl, data_ref, rtol=0, atol=0)
    torch.testing.assert_close(scales_cutedsl, scales_ref, rtol=0, atol=0)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    input_shape = config.input_shape
    scaling_mode = config.scaling_mode
    num_groups = config.num_groups

    input_tensor = make_input(input_shape, torch.bfloat16)

    M, K = input_shape

    # Generate jagged offsets with multiples of 128
    # TODO: we use multiple of 128 here to avoid per-group padding requirement in blocked scales layout, which cutedsl doesn't support yet.
    offs = generate_jagged_offs(num_groups, M, multiple_of=128, device=device)

    # Benchmark 1: CuTeDSL kernel with blocked scale output
    data_cutedsl, scales_cutedsl = mxfp8_quantize_cutedsl_2d_1x32(
        input_tensor,
        block_size=block_size,
        scaling_mode=scaling_mode,
        blocked_scale_output=True,
    )
    if VALIDATE:
        validate_outputs(input_tensor, data_cutedsl, scales_cutedsl, scaling_mode)
    cutedsl_blocked_time_us = benchmark_cuda_function_in_microseconds(
        mxfp8_quantize_cutedsl_2d_1x32,
        input_tensor,
        block_size=block_size,
        scaling_mode=scaling_mode,
        blocked_scale_output=True,
    )

    # Benchmark 2: Triton quantization + CUDA scale rearrangement
    def triton_plus_rearrange(x, group_offs):
        # Quantize along dim0 (rowwise)
        data, scales = triton_to_mxfp8_dim0(
            x,
            inner_block_size=block_size,
            scaling_mode=scaling_mode,
        )
        # Convert scales to blocked layout
        scales_blocked = mx_block_rearrange_2d_M_groups_cuda(
            scales.view(torch.uint8), group_offs
        )
        return data, scales_blocked

    data_triton, scales_triton = triton_plus_rearrange(input_tensor, offs)
    triton_plus_rearrange_time_us = benchmark_cuda_function_in_microseconds(
        triton_plus_rearrange,
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
    triton_plus_rearrange_gbps = ((read_bytes + write_bytes) / 1e9) / (
        triton_plus_rearrange_time_us / 1e6
    )

    return ExperimentResult(
        cutedsl_blocked_us=cutedsl_blocked_time_us,
        triton_plus_rearrange_us=triton_plus_rearrange_time_us,
        cutedsl_blocked_gbps=cutedsl_blocked_gbps,
        triton_plus_rearrange_gbps=triton_plus_rearrange_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "num_groups",
        "cutedsl_blocked_us",
        "triton+rearrange_us",
        "speedup",
        "cutedsl_gbps",
        "triton+rearrange_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = (
            experiment.result.triton_plus_rearrange_us
            / experiment.result.cutedsl_blocked_us
        )
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.scaling_mode,
                experiment.config.num_groups,
                f"{experiment.result.cutedsl_blocked_us:.2f}",
                f"{experiment.result.triton_plus_rearrange_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.cutedsl_blocked_gbps:.1f}",
                f"{experiment.result.triton_plus_rearrange_gbps:.1f}",
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
