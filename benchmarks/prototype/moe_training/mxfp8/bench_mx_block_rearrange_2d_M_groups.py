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
from torch.utils.cpp_extension import load
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    torch_to_blocked_2d_M_groups,
    triton_mx_block_rearrange_2d_M_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Load CUDA extension for M_groups pipelined kernel
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MX_KERNELS_DIR = os.path.join(
    SCRIPT_DIR, "..", "..", "..", "..", "torchao", "csrc", "cuda", "mx_kernels"
)

print("Compiling CUDA M_groups kernel...")
mx_block_rearrange_cuda = load(
    name="mx_block_rearrange_2d_M_groups",
    sources=[
        os.path.join(MX_KERNELS_DIR, "mxfp8_extension.cpp"),
        os.path.join(MX_KERNELS_DIR, "mx_block_rearrange_2d_M_groups.cu"),
        os.path.join(MX_KERNELS_DIR, "mx_block_rearrange_2d_K_groups.cu"),
        os.path.join(MX_KERNELS_DIR, "mxfp8_cuda.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    extra_ldflags=["-lcuda"],
    verbose=False,
)
print("CUDA M_groups kernel compiled successfully!")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]
    num_groups: int
    max_cols: int
    chunks_per_tb: int


@dataclass(frozen=True)
class ExperimentResult:
    torch_time_us: float
    triton_time_us: float
    cuda_time_us: float
    torch_mem_bw_gbps: float
    triton_mem_bw_gbps: float
    cuda_mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes. Input activations are scaled along K dim.
    block_size = 32
    input_shapes = [
        (16640, 5120 // block_size),
        (131072, 5120 // block_size),
    ]
    num_groups = [8]
    max_cols_list = [64, 128]
    chunks_per_tb_list = [4, 8]

    configs = []
    for shape, groups, max_cols, chunks_per_tb in itertools.product(
        input_shapes,
        num_groups,
        max_cols_list,
        chunks_per_tb_list,
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                num_groups=groups,
                max_cols=max_cols,
                chunks_per_tb=chunks_per_tb,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    input_shape, num_groups = config.input_shape, config.num_groups
    max_cols, chunks_per_tb = config.max_cols, config.chunks_per_tb

    input_tensor = torch.randint(
        low=0,
        high=256,
        size=input_shape,
        dtype=torch.uint8,
        device=device,
    )

    Mg, K = input_shape
    block_size = 32
    input_group_offsets = generate_jagged_offs(num_groups, Mg, multiple_of=block_size)

    # bench torch
    compiled_run_torch = torch.compile(torch_to_blocked_2d_M_groups)
    torch_out_scales, torch_group_offs = compiled_run_torch(
        input_tensor,
        input_group_offsets,
        block_size=block_size,
    )
    torch_time_us = benchmark_cuda_function_in_microseconds(
        compiled_run_torch,
        input_tensor,
        input_group_offsets,
        block_size=block_size,
    )

    # bench triton
    triton_out_scales = triton_mx_block_rearrange_2d_M_groups(
        input_tensor,
        input_group_offsets,
    )
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_mx_block_rearrange_2d_M_groups,
        input_tensor,
        input_group_offsets,
    )

    # bench CUDA pipelined kernel with configured max_cols and chunks_per_tb
    _ = mx_block_rearrange_cuda.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined(
        input_tensor.view(torch.uint8),
        input_group_offsets.to(torch.int32),
        max_cols,
        chunks_per_tb,
    )
    cuda_time_us = benchmark_cuda_function_in_microseconds(
        mx_block_rearrange_cuda.mx_block_rearrange_2d_M_groups_rowmajor_128x4_vec_pipelined,
        input_tensor.view(torch.uint8),
        input_group_offsets.to(torch.int32),
        max_cols,
        chunks_per_tb,
    )

    # mem bw calculations
    bytes_per_input_el = torch.finfo(torch.float8_e8m0fnu).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = triton_out_scales.numel() * bytes_per_output_el

    torch_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (torch_time_us / 1e6)
    triton_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_time_us / 1e6)
    cuda_mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (cuda_time_us / 1e6)

    return ExperimentResult(
        torch_time_us=torch_time_us,
        triton_time_us=triton_time_us,
        cuda_time_us=cuda_time_us,
        torch_mem_bw_gbps=torch_mem_bw_gbps,
        triton_mem_bw_gbps=triton_mem_bw_gbps,
        cuda_mem_bw_gbps=cuda_mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "max_cols",
        "chunks_per_tb",
        "torch_time_us",
        "triton_time_us",
        "cuda_time_us",
        "triton_speedup",
        "cuda_speedup",
    ]
    rows = []
    for experiment in experiments:
        input_shape = (
            f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]})"
        )
        rows.append(
            [
                input_shape,
                experiment.config.max_cols,
                experiment.config.chunks_per_tb,
                f"{experiment.result.torch_time_us:.2f}",
                f"{experiment.result.triton_time_us:.2f}",
                f"{experiment.result.cuda_time_us:.2f}",
                f"{experiment.result.torch_time_us / experiment.result.triton_time_us:.2f}x",
                f"{experiment.result.torch_time_us / experiment.result.cuda_time_us:.2f}x",
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
