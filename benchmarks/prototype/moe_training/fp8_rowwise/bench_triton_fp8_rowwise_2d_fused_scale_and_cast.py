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

# Wrap tensor_to_scale so torch.compile treats it as an opaque call.
# This simulates what happens inside the MoE forward pass: the compiler cannot
# fuse across the tensor_to_scale boundary, leaving 3 separate kernel launches.
_tensor_to_scale_opaque = torch.compiler.disable(tensor_to_scale)


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple  # (M, K)
    round_scales_to_power_of_2: bool


@dataclass(frozen=True)
class ExperimentResult:
    torch_compile_time_us: float
    compiled_graph_unfused_time_us: float
    triton_time_us: float
    torch_compile_mem_bw_gbps: float
    compiled_graph_unfused_mem_bw_gbps: float
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

    # --- Column 1: torch.compile of the isolated 3-kernel sequence.
    # Best-case for unfused: compiler sees the whole sequence and can optimize freely.
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

    # --- Column 2: 3-kernel sequence inside a compiled graph with an opaque boundary.
    # Simulates the actual MoE forward pass: torch.compile cannot fuse across the
    # tensor_to_scale call because it's treated as an opaque custom op, leaving
    # 3 separate kernel launches — exactly what the fused kernel replaces in practice.
    def run_compiled_graph_unfused(A: torch.Tensor):
        A_scales = _tensor_to_scale_opaque(
            A,
            float8_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=config.round_scales_to_power_of_2,
        )
        A_scaled = A.to(torch.float32) * A_scales
        A_data = to_fp8_saturated(A_scaled, float8_dtype)
        return A_data, A_scales

    run_compiled_graph_unfused_compiled = torch.compile(run_compiled_graph_unfused)

    # --- Column 3: Fused Triton kernel (also opaque to torch.compile as a custom_op).
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
        run_compiled_graph_unfused_compiled(input_tensor)
    for _ in range(10):
        run_fused(input_tensor)

    torch_compile_time_us = benchmark_cuda_function_in_microseconds(
        run_original_compiled, input_tensor
    )
    compiled_graph_unfused_time_us = benchmark_cuda_function_in_microseconds(
        run_compiled_graph_unfused_compiled, input_tensor
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
    compiled_graph_unfused_mem_bw_gbps = (total_bytes / 1e9) / (
        compiled_graph_unfused_time_us / 1e6
    )
    triton_mem_bw_gbps = (total_bytes / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        torch_compile_time_us=torch_compile_time_us,
        compiled_graph_unfused_time_us=compiled_graph_unfused_time_us,
        triton_time_us=triton_time_us,
        torch_compile_mem_bw_gbps=torch_compile_mem_bw_gbps,
        compiled_graph_unfused_mem_bw_gbps=compiled_graph_unfused_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
    )


def print_results(experiments: list[Experiment]):
    headers = [
        "shape (M, K)",
        "dtype",
        "compile isolated (us)",
        "compile+opaque (us)",
        "triton (us)",
        "speedup vs opaque",
        "compile isolated BW (GB/s)",
        "compile+opaque BW (GB/s)",
        "triton BW (GB/s)",
    ]
    rows = []
    for experiment in experiments:
        m, k = experiment.config.input_shape
        speedup_vs_opaque = (
            experiment.result.compiled_graph_unfused_time_us
            / experiment.result.triton_time_us
        )
        rows.append(
            [
                f"({m}, {k})",
                str(experiment.config.high_precision_dtype).split(".")[-1],
                f"{experiment.result.torch_compile_time_us:.1f}",
                f"{experiment.result.compiled_graph_unfused_time_us:.1f}",
                f"{experiment.result.triton_time_us:.1f}",
                f"{speedup_vs_opaque:.2f}x",
                f"{experiment.result.torch_compile_mem_bw_gbps:.1f}",
                f"{experiment.result.compiled_graph_unfused_mem_bw_gbps:.1f}",
                f"{experiment.result.triton_mem_bw_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))
    print()

    speedups_vs_isolated = [
        e.result.torch_compile_time_us / e.result.triton_time_us for e in experiments
    ]
    speedups_vs_opaque = [
        e.result.compiled_graph_unfused_time_us / e.result.triton_time_us
        for e in experiments
    ]
    print(
        f"Triton vs compile-isolated  — avg: {sum(speedups_vs_isolated)/len(speedups_vs_isolated):.2f}x  "
        f"min: {min(speedups_vs_isolated):.2f}x  max: {max(speedups_vs_isolated):.2f}x"
    )
    print(
        f"Triton vs compile+opaque    — avg: {sum(speedups_vs_opaque)/len(speedups_vs_opaque):.2f}x  "
        f"min: {min(speedups_vs_opaque):.2f}x  max: {max(speedups_vs_opaque):.2f}x"
    )


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print()
    print("=" * 110)
    print("Fused 2D Scale+Cast Kernel Benchmark")
    print()
    print("  compile-isolated : torch.compile of the 3-kernel sequence in isolation.")
    print(
        "                     Best-case for unfused — compiler can optimize the full sequence freely."
    )
    print(
        "  compile+opaque   : torch.compile where tensor_to_scale is opaque (compiler.disable)."
    )
    print(
        "                     Simulates actual MoE training: 3 separate kernel launches inside"
    )
    print(
        "                     a larger compiled graph, which is what the fused kernel replaces."
    )
    print("  triton           : triton_fp8_rowwise_2d_scale_and_cast() fused kernel.")
    print("=" * 110)
    print()
    print_results(results)


if __name__ == "__main__":
    main()
