# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Microbenchmark for fused mxfp8_quant_and_transpose kernel vs 2-step approach.

from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.ep.syncless.mxfp8_kernels import (
    mxfp8_quant_and_transpose,
    triton_mx_block_rearrange_input_sym_mem_buffer,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

SCALE_BLOCK_SIZE = 32


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    # Fused kernel
    fused_us: float
    fused_gbps: float
    # 2-stage: quant + rearrange for both non-transpose and transpose
    two_stage_us: float
    two_stage_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        (1 * 4096, 2048),
        (1 * 4096, 7168),
        (2 * 4096, 7168),
        (2 * 4096, 2048),
        (4 * 4096, 7168),
        (4 * 4096, 2048),
    ]
    configs = []
    for shape in input_shapes:
        configs.append(ExperimentConfig(input_shape=shape))
    return configs


def _two_stage_quant_and_transpose(input_bf16, num_tokens, num_tokens_t):
    """2-stage: quantize + rearrange for both non-transpose and transpose paths."""
    # Non-transpose: quantize (M, N) with 1x32 row-wise, then rearrange scales
    e4m3, scales = triton_to_mxfp8_dim0(
        input_bf16, inner_block_size=SCALE_BLOCK_SIZE, scaling_mode="rceil"
    )
    scales_blocked = triton_mx_block_rearrange_input_sym_mem_buffer(scales, num_tokens)

    # Transpose: quantize (N, M) with 1x32 row-wise, then rearrange scales
    input_t = input_bf16.T.contiguous()
    t_e4m3, t_scales = triton_to_mxfp8_dim0(
        input_t, inner_block_size=SCALE_BLOCK_SIZE, scaling_mode="rceil"
    )
    t_scales_blocked = triton_mx_block_rearrange_input_sym_mem_buffer(
        t_scales, num_tokens_t
    )

    return e4m3, scales_blocked, t_e4m3, t_scales_blocked


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M, N = config.input_shape

    input_bf16 = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    num_tokens = torch.tensor(M, dtype=torch.int64, device=device)
    num_tokens_t = torch.tensor(N, dtype=torch.int64, device=device)

    # --- Fused kernel ---
    # Warmup
    mxfp8_quant_and_transpose(input_bf16)

    fused_us = benchmark_cuda_function_in_microseconds(
        mxfp8_quant_and_transpose,
        input_bf16,
    )

    # --- 2-stage ---
    # Warmup
    _two_stage_quant_and_transpose(input_bf16, num_tokens, num_tokens_t)

    two_stage_us = benchmark_cuda_function_in_microseconds(
        _two_stage_quant_and_transpose,
        input_bf16,
        num_tokens,
        num_tokens_t,
    )

    # --- Memory bandwidth ---
    # Fused: reads input once (M*N*2B), writes 2x FP8 data + 2x scales
    # Non-transpose FP8: M*N*1B, scales: padded_M * padded_N_scale_cols * 1B
    # Transpose FP8: N*M*1B, scales: padded_N * padded_M_scale_cols * 1B
    read_bytes = M * N * 2  # bf16 input read once
    write_fp8_bytes = M * N * 1 + N * M * 1  # two FP8 outputs
    scale_cols = N // SCALE_BLOCK_SIZE
    scale_cols_t = M // SCALE_BLOCK_SIZE
    write_scale_bytes = M * scale_cols * 1 + N * scale_cols_t * 1  # approx
    total_bytes = read_bytes + write_fp8_bytes + write_scale_bytes

    fused_gbps = (total_bytes / 1e9) / (fused_us / 1e6)
    two_stage_gbps = (total_bytes / 1e9) / (two_stage_us / 1e6)

    return ExperimentResult(
        fused_us=fused_us,
        fused_gbps=fused_gbps,
        two_stage_us=two_stage_us,
        two_stage_gbps=two_stage_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "fused_us",
        "2stage_us",
        "fused_gbps",
        "2stage_gbps",
        "fused_speedup",
    ]
    rows = []
    for experiment in experiments:
        speedup = round(experiment.result.two_stage_us / experiment.result.fused_us, 3)
        rows.append(
            [
                str(experiment.config.input_shape),
                round(experiment.result.fused_us, 3),
                round(experiment.result.two_stage_us, 3),
                round(experiment.result.fused_gbps, 3),
                round(experiment.result.two_stage_gbps, 3),
                f"{speedup}x",
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

    print_results(results)


if __name__ == "__main__":
    main()
