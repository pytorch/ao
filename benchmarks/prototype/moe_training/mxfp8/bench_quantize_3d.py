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
    _mxfp8_cuda_kernels_available,
    _mxfp8_flydsl_kernels_available,
    mxfp8_quantize_cuda_3d,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_3d_flydsl,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_dim1_3d,
)
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import from_blocked

device = torch.device("cuda")
VALIDATE = os.environ.get("MXFP8_BENCH_VALIDATE", "0") == "1"
INPUT_MODE = os.environ.get("MXFP8_BENCH_INPUT_MODE", "randn")
EXTRA_NON_NICE = os.environ.get("MXFP8_BENCH_EXTRA_NON_NICE", "0") == "1"

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
    flydsl_3d_us: float
    # mem bw
    to_mx_gbps: float
    cuda_2d_gbps: float
    cutedsl_3d_gbps: float
    flydsl_3d_gbps: float


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
    if EXTRA_NON_NICE:
        input_shapes += [
            (3, 96, 416),
            (5, 160, 1408),
            (2, 384, 1152),
        ]
    round_modes = [ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL]
    variants = ["32x1_n", "32x1_t", "32x32_n", "32x32_t"]
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


def make_input(shape: tuple[int, int, int], dtype: torch.dtype) -> torch.Tensor:
    if INPUT_MODE == "randn":
        return torch.randn(*shape, dtype=dtype, device=device)

    numel = shape[0] * shape[1] * shape[2]
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


def logical_scales_from_blocked(
    scales_blocked: torch.Tensor,
    quant_input: torch.Tensor,
    scale_block_dim2: int,
) -> torch.Tensor:
    block_size = 32
    s_rows = quant_input.shape[-1]
    s_cols = quant_input.shape[-2] // block_size
    s_blocked_full = (
        torch.stack(
            [
                from_blocked(scales_blocked[e], s_rows, s_cols).view(torch.uint8)
                for e in range(quant_input.shape[0])
            ],
            dim=0,
        )
        .view(torch.float8_e8m0fnu)
        .to(torch.float32)
    )
    if scale_block_dim2 == 32:
        return s_blocked_full[:, ::block_size, :].transpose(-2, -1).contiguous()
    return s_blocked_full.transpose(-2, -1).contiguous()


def validate_cutedsl_3d(
    input_tensor: torch.Tensor,
    quant_input: torch.Tensor,
    data_cutedsl: torch.Tensor,
    scales_cutedsl: torch.Tensor,
    config: ExperimentConfig,
    scale_block_dim2: int,
):
    data_ref, scales_ref = using_to_mx_reference(input_tensor, config)
    scales_logical = logical_scales_from_blocked(
        scales_cutedsl,
        quant_input,
        scale_block_dim2,
    ).to(scales_ref.dtype)
    if (
        scale_block_dim2 == 32
        and scales_ref.ndim == scales_logical.ndim + 1
        and scales_ref.shape[-1] == 1
    ):
        scales_ref = scales_ref.squeeze(-1)
    if scale_block_dim2 == 32:
        scales_ref = scales_ref.to(scales_logical.dtype)
    torch.testing.assert_close(data_cutedsl, data_ref, rtol=0, atol=0)
    torch.testing.assert_close(scales_logical, scales_ref, rtol=0, atol=0)


def using_to_mx_reference(
    x: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = 32
    variant = config.variant
    if variant == "32x1_t":
        x_t = x.transpose(-2, -1)
        s_ref, y_ref = to_mx(
            x_t.transpose(-2, -1).contiguous(),
            elem_dtype=torch.float8_e4m3fn,
            block_size=block_size,
            scaling_mode=config.scaling_mode,
        )
        return y_ref.transpose(-2, -1), s_ref.transpose(-2, -1)

    if variant == "32x32_t":
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
            .transpose(-2, -1)
        )
        return y_ref, s_ref.squeeze(-1).transpose(-2, -1)

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
        y_tiles_ref.view(E, N // block_size, K // block_size, block_size, block_size)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(E, N, K)
    )
    y_ref = y_ref.transpose(-2, -1).contiguous().transpose(-2, -1)
    return y_ref, s_ref.squeeze(-1)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    variant = config.variant
    input_shape = config.input_shape
    input_tensor = make_input(input_shape, torch.bfloat16)

    def get_quant_input(x: torch.Tensor) -> torch.Tensor:
        # The "*_t" benchmark rows feed (E, K, N) K-major expert weights
        # directly into the transposed-input 3D kernel contracts.
        if variant in ("32x1_t", "32x32_t"):
            return x.transpose(-2, -1)
        return x

    # bench to_mx
    using_to_mx_c = torch.compile(lambda x: using_to_mx_reference(x, config))
    data_to_mx, scales_to_mx = using_to_mx_c(input_tensor)
    to_mx_time_us = benchmark_cuda_function_in_microseconds(
        using_to_mx_c,
        input_tensor,
    )

    if variant == "32x1_n" and _mxfp8_cuda_kernels_available:
        # bench 2d dim1 kernel then transforming to col major — CUDA cuTeDSL only.
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

    # bench 3d CuTeDSL kernel — CUDA only (SM 10.0+).
    if _mxfp8_cuda_kernels_available:
        data_cutedsl, scales_cutedsl = mxfp8_quantize_cuda_3d(
            quant_input,
            block_size=block_size,
            scale_block_dim1=scale_block_dim1,
            scale_block_dim2=scale_block_dim2,
            scaling_mode=str(config.scaling_mode.value),
        )
        if VALIDATE:
            validate_cutedsl_3d(
                input_tensor,
                quant_input,
                data_cutedsl,
                scales_cutedsl,
                config,
                scale_block_dim2,
            )
        time_cutedsl_3d_us = benchmark_cuda_function_in_microseconds(
            mxfp8_quantize_cuda_3d,
            quant_input,
            block_size=block_size,
            scale_block_dim1=scale_block_dim1,
            scale_block_dim2=scale_block_dim2,
            scaling_mode=str(config.scaling_mode.value),
        )
    else:
        time_cutedsl_3d_us = float("nan")

    # bench 3d FlyDSL kernel — AMD (CDNA3+) only; supports FLOOR and RCEIL.
    # Same interface as mxfp8_quantize_cuda_3d. FlyDSL requires a contiguous
    # row-major input, so the transposed "*_t" rows are made contiguous first.
    if _mxfp8_flydsl_kernels_available:
        flydsl_input = quant_input.contiguous()
        mxfp8_quantize_3d_flydsl(
            flydsl_input,
            block_size=block_size,
            scale_block_dim1=scale_block_dim1,
            scale_block_dim2=scale_block_dim2,
            scaling_mode=str(config.scaling_mode.value),
            blocked_scale_output=False,
        )
        time_flydsl_3d_us = benchmark_cuda_function_in_microseconds(
            mxfp8_quantize_3d_flydsl,
            flydsl_input,
            block_size=block_size,
            scale_block_dim1=scale_block_dim1,
            scale_block_dim2=scale_block_dim2,
            scaling_mode=str(config.scaling_mode.value),
            blocked_scale_output=False,
        )
    else:
        time_flydsl_3d_us = float("nan")

    # mem bw calculations
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = quant_input.numel() * bytes_per_input_el
    # Use to_mx outputs as the byte-count reference (always available); the
    # cuda_3d outputs may be None on AMD where the cuTeDSL kernel is unavailable.
    write_bytes = (
        data_to_mx.numel() * bytes_per_output_el
        + scales_to_mx.numel() * bytes_per_scale_el
    )

    cutedsl_3d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_cutedsl_3d_us / 1e6)
    to_mx_gbps = ((read_bytes + write_bytes) / 1e9) / (to_mx_time_us / 1e6)
    cuda_2d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_cuda_2d_us / 1e6)
    flydsl_3d_gbps = ((read_bytes + write_bytes) / 1e9) / (time_flydsl_3d_us / 1e6)
    return ExperimentResult(
        # time
        to_mx_us=to_mx_time_us,
        cuda_2d_us=time_cuda_2d_us,
        cutedsl_3d_us=time_cutedsl_3d_us,
        flydsl_3d_us=time_flydsl_3d_us,
        # mem bw
        to_mx_gbps=to_mx_gbps,
        cuda_2d_gbps=cuda_2d_gbps,
        cutedsl_3d_gbps=cutedsl_3d_gbps,
        flydsl_3d_gbps=flydsl_3d_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "variant",
        "cuda_2d_us",
        "cutedsl_3d_us",
        "flydsl_3d_us",
        "to_mx_us",
        "cuda_2d_gbps",
        "cutedsl_3d_gbps",
        "flydsl_3d_gbps",
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
                experiment.result.flydsl_3d_us,
                experiment.result.to_mx_us,
                round(experiment.result.cuda_2d_gbps, 3),
                round(experiment.result.cutedsl_3d_gbps, 3),
                round(experiment.result.flydsl_3d_gbps, 3),
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
