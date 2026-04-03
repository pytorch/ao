# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import List, Tuple

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels.fp8_tensorwise_per_group import (
    triton_fp8_tensorwise_per_group_quantize,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple  # (M, K)
    n_groups: int
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


def torch_to_float8_per_group_tensorwise(
    tensor: torch.Tensor,
    offs: torch.Tensor,
    target_dtype: torch.dtype = torch.float8_e4m3fn,
    round_scales_to_power_of_2: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference torch implementation of per-group tensorwise FP8 quantization.
    Each group gets a single scalar scale (one scale per group, not per-row or per-col).
    """
    num_groups = offs.numel()
    M, K = tensor.shape
    fp8_out = torch.empty_like(tensor, dtype=target_dtype)
    scales = torch.empty(num_groups, dtype=torch.float32, device=tensor.device)

    start_idx = 0
    for g, end_idx in enumerate(offs.tolist()):
        subtensor = tensor[start_idx:end_idx, :]

        # Compute a single tensorwise scale for this group.
        subtensor_scale = tensor_to_scale(
            subtensor,
            target_dtype,
            scaling_granularity=ScalingGranularity.TENSORWISE,
            round_scales_to_power_of_2=round_scales_to_power_of_2,
        )

        # Apply scale and cast to FP8.
        tensor_scaled = subtensor.to(torch.float32) * subtensor_scale
        fp8_out[start_idx:end_idx, :] = to_fp8_saturated(tensor_scaled, target_dtype)
        scales[g] = subtensor_scale.squeeze()

        start_idx = end_idx

    return fp8_out, scales


def get_configs() -> List[ExperimentConfig]:
    # MoE-relevant shapes: (M, K) where M = total tokens routed across experts.
    input_shapes = [
        (4096, 5120),
        (4096, 8192),
        (8192, 5120),
        (8192, 8192),
        (16384, 5120),
        (16384, 8192),
    ]
    n_groups_list = [1, 4, 8, 16]
    high_precision_dtypes = [torch.bfloat16]
    power_of_2_scales = [True]
    configs = []
    for (
        input_shape,
        n_groups,
        high_precision_dtype,
        round_scales_to_power_of_2,
    ) in itertools.product(
        input_shapes, n_groups_list, high_precision_dtypes, power_of_2_scales
    ):
        configs.append(
            ExperimentConfig(
                input_shape=input_shape,
                n_groups=n_groups,
                high_precision_dtype=high_precision_dtype,
                round_scales_to_power_of_2=round_scales_to_power_of_2,
            )
        )
    return configs


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    float8_dtype = torch.float8_e4m3fn
    M, K = config.input_shape

    input_tensor = torch.randn(
        M,
        K,
        dtype=config.high_precision_dtype,
        device=device,
    )

    offs = generate_jagged_offs(config.n_groups, M, multiple_of=16)

    # --- torch.compile of the per-group loop baseline.
    torch_baseline_compiled = torch.compile(torch_to_float8_per_group_tensorwise)

    # --- Fused Triton kernel (two-pass per-group: amax + quantize).
    def run_fused(A: torch.Tensor):
        return triton_fp8_tensorwise_per_group_quantize(
            A,
            offs,
            output_dtype=float8_dtype,
            output_scale_dim=1,
            round_scales_to_power_of_2=config.round_scales_to_power_of_2,
        )

    # Warmup
    for _ in range(10):
        torch_baseline_compiled(
            input_tensor,
            offs,
            target_dtype=float8_dtype,
            round_scales_to_power_of_2=config.round_scales_to_power_of_2,
        )
    for _ in range(10):
        run_fused(input_tensor)

    torch_compile_time_us = benchmark_cuda_function_in_microseconds(
        torch_baseline_compiled,
        input_tensor,
        offs,
        target_dtype=float8_dtype,
        round_scales_to_power_of_2=config.round_scales_to_power_of_2,
    )
    triton_time_us = benchmark_cuda_function_in_microseconds(run_fused, input_tensor)

    # Memory bandwidth calculation
    # Both approaches: read input 2x (pass 1: per-group amax, pass 2: scale+cast), write FP8 output 1x
    bytes_per_input_el = torch.finfo(config.high_precision_dtype).bits / 8
    num_elements = input_tensor.numel()
    n_groups = config.n_groups
    read_bytes = (
        num_elements * bytes_per_input_el * 2  # 2 passes over input
        + 4 * n_groups  # read per-group amax (fp32)
    )
    write_bytes = (
        num_elements * 1  # 1 byte per FP8 element
        + 4 * n_groups  # write per-group scales (fp32)
    )
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
        "shape (M, K)",
        "n_groups",
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
                experiment.config.n_groups,
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
    print(
        f"Triton vs torch.compile  -- avg: {sum(speedups) / len(speedups):.2f}x  "
        f"min: {min(speedups):.2f}x  max: {max(speedups):.2f}x"
    )


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print()
    print("=" * 90)
    print("Fused Per-Group Tensorwise Quantization Kernel Benchmark")
    print()
    print("  torch.compile : torch.compile of a per-group loop baseline")
    print("                  (loop over groups, tensor_to_scale(TENSORWISE) per group,")
    print("                  multiply + to_fp8_saturated).")
    print("  triton        : triton_fp8_tensorwise_per_group_quantize() fused two-pass")
    print("                  kernel. Pass 1: per-group amax via in-kernel offset scan +")
    print("                  atomic_max. Pass 2: per-group scale + clamp + FP8 cast.")
    print("=" * 90)
    print()
    print_results(results)


if __name__ == "__main__":
    main()
