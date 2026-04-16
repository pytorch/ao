# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.ep.syncless.mxfp8_requant_kernel import (
    mxfp8_dequant_buffer,
    mxfp8_dequant_requant_col_major,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx

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
    # 2-stage: dequant + requant
    two_stage_us: float
    two_stage_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        # (num_tokens, dim)
        (1 * 8192, 7168),
        (2 * 8192, 7168),
        (4 * 8192, 7168),
        (8 * 8192, 7168),
    ]
    configs = []
    for shape in input_shapes:
        configs.append(
            ExperimentConfig(
                input_shape=shape,
            )
        )
    return configs


def _two_stage_dequant_requant(
    e4m3_data,
    e8m0_scales,
    num_tokens,
    sym_mem_buffer_rows,
    out_buffer,
    out_offset,
):
    """2-stage: dequant to bf16 via mxfp8_dequant_buffer, then requant 32x1 via CuTeDSL."""
    mxfp8_dequant_buffer(
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        out_buffer,
        out_offset,
        torch.bfloat16,
        SCALE_BLOCK_SIZE,
    )
    return mxfp8_quantize_2d_32x1_cutedsl(out_buffer, SCALE_BLOCK_SIZE)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M, dim = config.input_shape

    # Create input: quantize a random bf16 tensor to MXFP8 (1x32 row-major scales)
    input_tensor = torch.randn(M, dim, dtype=torch.bfloat16, device=device)
    e8m0_scales, e4m3_data = to_mx(input_tensor, torch.float8_e4m3fn, SCALE_BLOCK_SIZE)

    # --- Fused kernel setup ---
    num_tokens = torch.tensor(M, dtype=torch.int64, device=device)
    out_offset = torch.tensor(0, dtype=torch.int64, device=device)
    sym_mem_buffer_rows = M
    assert sym_mem_buffer_rows % SCALE_BLOCK_SIZE == 0

    out_data = torch.empty(
        dim, sym_mem_buffer_rows, dtype=torch.float8_e4m3fn, device=device
    )
    out_scale_cols = sym_mem_buffer_rows // SCALE_BLOCK_SIZE
    out_scales = torch.empty(dim, out_scale_cols, dtype=torch.uint8, device=device)

    # Warmup fused
    mxfp8_dequant_requant_col_major(
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        out_data,
        out_scales,
        out_offset,
        SCALE_BLOCK_SIZE,
    )

    # Benchmark fused
    fused_us = benchmark_cuda_function_in_microseconds(
        mxfp8_dequant_requant_col_major,
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        out_data,
        out_scales,
        out_offset,
        SCALE_BLOCK_SIZE,
    )

    # --- 2-stage setup ---
    dequant_buffer = torch.empty(M, dim, dtype=torch.bfloat16, device=device)

    # Warmup 2-stage
    _two_stage_dequant_requant(
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        dequant_buffer,
        out_offset,
    )

    # Benchmark 2-stage
    two_stage_us = benchmark_cuda_function_in_microseconds(
        _two_stage_dequant_requant,
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        dequant_buffer,
        out_offset,
    )

    # --- Memory bandwidth calculation (same logical I/O for both) ---
    # Reads: e4m3_data (M x dim, 1B) + e8m0_scales (M x dim//32, 1B)
    # Writes: out_data (dim x M, 1B) + out_scales (dim x M//32, 1B)
    read_bytes = e4m3_data.numel() + e8m0_scales.numel()
    write_bytes = out_data.numel() + out_scales.numel()
    total_bytes = read_bytes + write_bytes
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
