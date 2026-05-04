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
from triton.testing import do_bench

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels import (
    triton_fp8_colwise_3d_scale_and_cast,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: (
        tuple  # (E, N, K), allocated row-major then transposed to (E, K, N) col-major
    )
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


def get_configs() -> List[ExperimentConfig]:
    # MoE expert weights are 3D (E, K, N) column-major in fp8_grouped_mm forward.
    # Specified here as (E, N, K) row-major; the bench transposes to col-major
    # (matches actual usage in _Float8GroupedMM.forward).
    input_shapes = [
        # Llama4 expert weight shapes (cross-reference with bench_triton_fp8_rowwise_3d_transpose_rhs.py)
        (1, 8192, 5120),  # w1, w3
        (1, 5120, 8192),  # w2
        (16, 8192, 5120),  # w1, w3
        (16, 5120, 8192),  # w2
        (128, 8192, 5120),  # w1, w3
        (128, 5120, 8192),  # w2
        # DeepSeek-V3 671B with EP=8 (E_local = 256/8 = 32 experts/rank)
        # hidden_size=7168, moe_inter_dim=2048
        (32, 4096, 7168),  # w1+w3 fused gate/up: 2*moe_inter_dim along N
        (32, 7168, 2048),  # w2 down proj
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

    # Allocate (E, N, K) row-major then transpose to (E, K, N) column-major
    # (matches B_t layout in _Float8GroupedMM.forward: strides (K*N, 1, K)).
    input_tensor = torch.randn(
        *config.input_shape,
        dtype=config.high_precision_dtype,
        device=device,
    ).transpose(-2, -1)

    # --- torch.compile of the native 3-op sequence (tensor_to_scale + multiply + cast).
    # This is the best-case for the unfused path: the compiler sees the full sequence
    # and can fuse ops freely.
    def run_original(B_t: torch.Tensor):
        B_t_scales = tensor_to_scale(
            B_t,
            float8_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=config.round_scales_to_power_of_2,
        )
        B_t_scaled = B_t.to(torch.float32) * B_t_scales
        B_t_data = to_fp8_saturated(B_t_scaled, float8_dtype)
        return B_t_data, B_t_scales

    run_original_compiled = torch.compile(run_original)

    # --- Fused Triton kernel.
    def run_fused(B_t: torch.Tensor):
        return triton_fp8_colwise_3d_scale_and_cast(
            B_t,
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
    # Both approaches: read input 2x (pass 1: absmax along K, pass 2: scale+cast),
    # write FP8 output 1x. Scales are negligible.
    bytes_per_input_el = torch.finfo(config.high_precision_dtype).bits / 8
    bytes_per_output_el = torch.finfo(float8_dtype).bits / 8
    num_elements = input_tensor.numel()
    read_bytes = num_elements * bytes_per_input_el * 2  # 2 passes over input
    write_bytes = num_elements * bytes_per_output_el
    total_bytes = read_bytes + write_bytes

    torch_compile_mem_bw_gbps = (total_bytes / 1e9) / (torch_compile_time_us / 1e6)
    triton_mem_bw_gbps = (total_bytes / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        torch_compile_time_us=torch_compile_time_us,
        triton_time_us=triton_time_us,
        torch_compile_mem_bw_gbps=torch_compile_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "shape (E, K, N)",
        "dtype",
        "torch.compile (us)",
        "triton (us)",
        "speedup",
        "torch.compile BW (GB/s)",
        "triton BW (GB/s)",
    ]
    rows = []
    for experiment in experiments:
        # input_shape was (E, N, K) row-major; tensor is (E, K, N) col-major after transpose
        e, n, k = experiment.config.input_shape
        rows.append(
            [
                f"({e}, {k}, {n})",
                experiment.config.high_precision_dtype,
                f"{experiment.result.torch_compile_time_us:.2f}",
                f"{experiment.result.triton_time_us:.2f}",
                f"{experiment.result.torch_compile_time_us / experiment.result.triton_time_us:.2f}x",
                f"{experiment.result.torch_compile_mem_bw_gbps:.1f}",
                f"{experiment.result.triton_mem_bw_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print_results(results)


if __name__ == "__main__":
    main()
