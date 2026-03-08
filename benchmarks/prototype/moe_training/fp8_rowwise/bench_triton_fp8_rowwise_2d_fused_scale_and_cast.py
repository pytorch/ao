# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels import (
    triton_fp8_rowwise_2d_scale_and_cast,
)

device = torch.device("cuda")

torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple  # (M, K)
    round_scales_to_power_of_2: bool


@dataclass(frozen=True)
class ExperimentResult:
    torch_compile_time_us: float
    triton_time_us: float
    torch_compile_mem_bw_gbps: float
    triton_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> list[ExperimentConfig]:
    # MoE-relevant 2D shapes: (M, K) where M = total tokens routed to experts.
    # In practice M depends on batch_size * seq_len * top_k / num_experts.
    # K is model hidden dim or intermediate dim.
    input_shapes = [
        (128, 5120),
        (128, 8192),
        (1024, 5120),
        (1024, 8192),
        (4096, 5120),
        (4096, 8192),
        (8192, 5120),
        (8192, 8192),
        (16384, 5120),
        (16384, 8192),
    ]
    high_precision_dtypes = [torch.bfloat16]
    power_of_2_scales = [True]
    configs = []
    for (
        input_shape,
        high_precision_dtype,
        round_scales_to_power_of_2,
    ) in itertools.product(input_shapes, high_precision_dtypes, power_of_2_scales):
        configs.append(
            ExperimentConfig(
                input_shape=input_shape,
                high_precision_dtype=high_precision_dtype,
                round_scales_to_power_of_2=round_scales_to_power_of_2,
            )
        )
    return configs


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    float8_dtype = torch.float8_e4m3fn

    input_tensor = torch.randn(
        *config.input_shape,
        dtype=config.high_precision_dtype,
        device=device,
    )

    # --- torch.compile reference: 3-kernel sequence ---
    def run_original(A: torch.Tensor):
        A_scales = tensor_to_scale(
            A,
            float8_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=config.round_scales_to_power_of_2,
        )
        A_scaled = A.to(torch.float32) * A_scales
        A_data = to_fp8_saturated(A_scaled, float8_dtype)
        return A_data, A_scales

    run_original_compiled = torch.compile(run_original)

    # --- Fused Triton kernel ---
    def run_fused(A: torch.Tensor):
        return triton_fp8_rowwise_2d_scale_and_cast(
            A,
            output_dtype=float8_dtype,
            round_scales_to_power_of_2=config.round_scales_to_power_of_2,
        )

    # Warmup
    for _ in range(10):
        run_original_compiled(input_tensor)
    for _ in range(10):
        run_fused(input_tensor)

    torch_compile_time_us = benchmark_cuda_function_in_microseconds(
        run_original_compiled, input_tensor
    )
    triton_time_us = benchmark_cuda_function_in_microseconds(run_fused, input_tensor)

    # Memory bandwidth calculation
    # Both approaches: read input 2x (pass 1: absmax, pass 2: scale+cast), write FP8 output 1x
    bytes_per_input_el = torch.finfo(config.high_precision_dtype).bits / 8
    num_elements = input_tensor.numel()
    read_bytes = num_elements * bytes_per_input_el * 2  # 2 passes over input
    write_bytes = num_elements * 1  # 1 byte per FP8 element
    total_bytes = read_bytes + write_bytes

    torch_compile_mem_bw_gbps = (total_bytes / 1e9) / (torch_compile_time_us / 1e6)
    triton_mem_bw_gbps = (total_bytes / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        torch_compile_time_us=torch_compile_time_us,
        triton_time_us=triton_time_us,
        torch_compile_mem_bw_gbps=torch_compile_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
    )


def print_results(experiments: list[Experiment]):
    headers = [
        "shape (M, K)",
        "dtype",
        "torch.compile (us)",
        "triton (us)",
        "speedup",
        "torch.compile BW (GB/s)",
        "triton BW (GB/s)",
    ]
    rows = []
    for experiment in experiments:
        m, k = experiment.config.input_shape
        speedup = (
            experiment.result.torch_compile_time_us / experiment.result.triton_time_us
        )
        rows.append(
            [
                f"({m}, {k})",
                str(experiment.config.high_precision_dtype).split(".")[-1],
                f"{experiment.result.torch_compile_time_us:.1f}",
                f"{experiment.result.triton_time_us:.1f}",
                f"{speedup:.2f}x",
                f"{experiment.result.torch_compile_mem_bw_gbps:.1f}",
                f"{experiment.result.triton_mem_bw_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))
    print()

    speedups = [
        e.result.torch_compile_time_us / e.result.triton_time_us for e in experiments
    ]
    print(f"Average speedup: {sum(speedups) / len(speedups):.2f}x")
    print(f"Min speedup:     {min(speedups):.2f}x")
    print(f"Max speedup:     {max(speedups):.2f}x")


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print()
    print("=" * 100)
    print("Fused 2D Scale+Cast Kernel vs torch.compile 3-Kernel Sequence")
    print(
        "Reference: torch.compile(tensor_to_scale() + A * scales + to_fp8_saturated())"
    )
    print("Fused:     triton_fp8_rowwise_2d_scale_and_cast()")
    print("=" * 100)
    print()
    print_results(results)


if __name__ == "__main__":
    main()
