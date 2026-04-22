# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Microbenchmark for mxfp8_dequant_buffer kernel vs eager dequant.

from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.ep.syncless.mxfp8_kernels import (
    mxfp8_dequant_buffer,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

SCALE_BLOCK_SIZE = 32


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]
    out_dtype: torch.dtype


@dataclass(frozen=True)
class ExperimentResult:
    triton_us: float
    triton_gbps: float
    eager_us: float
    eager_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        # (num_tokens, hidden_dim)
        (1 * 8192, 7168),
        (2 * 8192, 7168),
        (4 * 8192, 7168),
        (8 * 8192, 7168),
    ]
    configs = []
    for shape in input_shapes:
        for out_dtype in [torch.bfloat16]:
            configs.append(
                ExperimentConfig(input_shape=shape, out_dtype=out_dtype)
            )
    return configs


def _eager_dequant(
    e4m3_data: torch.Tensor,
    e8m0_scales: torch.Tensor,
    num_tokens_val: int,
    out_buffer: torch.Tensor,
    out_offset_val: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Eager dequant: cast FP8 to out_dtype, multiply by decoded E8M0 scales."""
    data_bf16 = e4m3_data[:num_tokens_val].to(out_dtype)
    # Decode E8M0 scales: reinterpret uint8 as biased exponent, broadcast over block
    scales_uint8 = e8m0_scales[:num_tokens_val]
    scales_f32 = torch.pow(2.0, scales_uint8.to(torch.float32) - 127.0)
    scales_expanded = scales_f32.repeat_interleave(SCALE_BLOCK_SIZE, dim=1)[
        :, : data_bf16.shape[1]
    ]
    out_buffer[out_offset_val : out_offset_val + num_tokens_val] = (
        data_bf16 * scales_expanded.to(out_dtype)
    )
    return out_buffer


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M, dim = config.input_shape
    out_dtype = config.out_dtype

    # Create MXFP8 quantized inputs
    input_tensor = torch.randn(M, dim, dtype=torch.bfloat16, device=device)
    e8m0_scales, e4m3_data = to_mx(input_tensor, torch.float8_e4m3fn, SCALE_BLOCK_SIZE)

    num_tokens = torch.tensor(M, dtype=torch.int64, device=device)
    out_offset = torch.tensor(0, dtype=torch.int64, device=device)
    sym_mem_buffer_rows = M

    # --- Triton kernel ---
    out_buffer = torch.empty(M, dim, dtype=out_dtype, device=device)

    # Warmup
    mxfp8_dequant_buffer(
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        out_buffer,
        out_offset,
        out_dtype,
        SCALE_BLOCK_SIZE,
    )

    triton_us = benchmark_cuda_function_in_microseconds(
        mxfp8_dequant_buffer,
        e4m3_data,
        e8m0_scales,
        num_tokens,
        sym_mem_buffer_rows,
        out_buffer,
        out_offset,
        out_dtype,
        SCALE_BLOCK_SIZE,
    )

    # --- Eager baseline ---
    eager_out_buffer = torch.empty(M, dim, dtype=out_dtype, device=device)

    # Warmup
    _eager_dequant(e4m3_data, e8m0_scales, M, eager_out_buffer, 0, out_dtype)

    eager_us = benchmark_cuda_function_in_microseconds(
        _eager_dequant,
        e4m3_data,
        e8m0_scales,
        M,
        eager_out_buffer,
        0,
        out_dtype,
    )

    # --- Memory bandwidth ---
    # Reads: e4m3_data (M x dim, 1B) + e8m0_scales (M x scale_dim, 1B)
    # Writes: out_buffer (M x dim, 2B for bf16 / 4B for fp32)
    out_elem_bytes = 2 if out_dtype == torch.bfloat16 else 4
    scale_dim = dim // SCALE_BLOCK_SIZE
    read_bytes = M * dim * 1 + M * scale_dim * 1
    write_bytes = M * dim * out_elem_bytes
    total_bytes = read_bytes + write_bytes

    triton_gbps = (total_bytes / 1e9) / (triton_us / 1e6)
    eager_gbps = (total_bytes / 1e9) / (eager_us / 1e6)

    return ExperimentResult(
        triton_us=triton_us,
        triton_gbps=triton_gbps,
        eager_us=eager_us,
        eager_gbps=eager_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "out_dtype",
        "triton_us",
        "eager_us",
        "triton_gbps",
        "eager_gbps",
        "speedup",
    ]
    rows = []
    for experiment in experiments:
        c, r = experiment.config, experiment.result
        speedup = round(r.eager_us / r.triton_us, 3) if r.triton_us > 0 else 0
        rows.append(
            [
                str(c.input_shape),
                str(c.out_dtype),
                round(r.triton_us, 3),
                round(r.eager_us, 3),
                round(r.triton_gbps, 1),
                round(r.eager_gbps, 1),
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
