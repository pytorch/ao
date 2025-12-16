# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mx_block_rearrange_2d_K_groups_cuda,
    torch_to_blocked_2d_K_groups,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int]
    num_groups: int
    version: str  # "naive" or "parallel"


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    mem_bw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 and DSV3 671b shapes. Input activations are scaled along the total_M dim, which contains all the token groups.
    block_size = 32
    input_shapes = [
        (8192, 32768 // block_size),
        (8192, 65536 // block_size),
        (8192, 131072 // block_size),
        (5120, 32768 // block_size),
        (5120, 65536 // block_size),
        (5120, 131072 // block_size),
        (7168, 32768 // block_size),
        (7168, 65536 // block_size),
        (7168, 131072 // block_size),
        (2048, 32768 // block_size),
        (2048, 65536 // block_size),
        (2048, 131072 // block_size),
    ]
    num_groups = [8]
    versions = [
        "torch",
        "triton",
        # CUDA kernel versions: cuda_{max_cols}_{chunks_per_tb}
        "cuda_64_4",
        "cuda_64_8",
        "cuda_64_16",
        "cuda_128_4",
        "cuda_128_8",
        "cuda_128_16",
    ]

    configs = []
    for shape, groups, version in itertools.product(
        input_shapes,
        num_groups,
        versions,
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                num_groups=groups,
                version=version,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    input_shape, num_groups, version = (
        config.input_shape,
        config.num_groups,
        config.version,
    )
    input_tensor = torch.randint(
        low=0,
        high=256,
        size=input_shape,
        dtype=torch.uint8,
        device=device,
    )

    M, Kg = input_shape
    block_size = 32
    input_group_offsets = generate_jagged_offs(num_groups, Kg, multiple_of=block_size)

    # Select which kernel to benchmark based on version
    if version == "torch":
        kernel_fn = torch_to_blocked_2d_K_groups
        kernel_input = input_tensor
    elif version == "triton":
        kernel_fn = triton_mx_block_rearrange_2d_K_groups
        # Triton uses row-major input
        kernel_input = input_tensor
    elif version.startswith("cuda_"):
        # Parse version string: cuda_{max_cols}_{chunks_per_tb}
        parts = version.split("_")
        max_cols = int(parts[1])
        chunks_per_tb = int(parts[2])
        kernel_fn = (
            lambda t,
            o,
            mc=max_cols,
            cptb=chunks_per_tb: mx_block_rearrange_2d_K_groups_cuda(
                t,
                o,
                max_cols=mc,
                chunks_per_tb=cptb,
            )
        )
        kernel_input = input_tensor.view(torch.float8_e8m0fnu)
    else:
        raise ValueError(f"Unknown version: {version}")

    # Run kernel to get output shape
    outputs = kernel_fn(
        kernel_input,
        input_group_offsets,
    )
    if isinstance(outputs, tuple):  # torch returns a tuple with extra metadata
        out_scales, _ = outputs
    else:
        out_scales = outputs

    # Benchmark the kernel
    time_us = benchmark_cuda_function_in_microseconds(
        kernel_fn,
        kernel_input,
        input_group_offsets,
    )

    # Calculate memory bandwidth
    bytes_per_input_el = torch.finfo(torch.float8_e8m0fnu).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = out_scales.numel() * bytes_per_output_el

    mem_bw_gbps = ((read_bytes + write_bytes) / 1e9) / (time_us / 1e6)

    return ExperimentResult(
        time_us=time_us,
        mem_bw_gbps=mem_bw_gbps,
    )


def print_results(experiments: List[Experiment]):
    # Group experiments by input shape
    shapes_dict = {}
    for exp in experiments:
        shape_key = exp.config.input_shape
        if shape_key not in shapes_dict:
            shapes_dict[shape_key] = {}
        shapes_dict[shape_key][exp.config.version] = exp.result

    headers = [
        "kernel_version",
        "scale_shape",
        "time_us",
        "mem_bw_gbps",
        "speedup_vs_torch",
        "speedup_vs_triton",
    ]

    rows = []
    for shape, versions in shapes_dict.items():
        # Get torch baseline time for speedup calculation
        torch_time_us = versions.get("torch").time_us if "torch" in versions else None

        # Get triton baseline time for speedup calculation
        triton_time_us = (
            versions.get("triton").time_us if "triton" in versions else None
        )

        # Add rows for each version
        for version, result in versions.items():
            # Calculate speedup vs torch
            speedup_vs_torch_str = ""
            if version != "torch" and torch_time_us is not None:
                speedup = torch_time_us / result.time_us
                speedup_vs_torch_str = f"{speedup:.2f}x"

            # Calculate speedup vs triton (only for CUDA kernels)
            speedup_vs_triton_str = ""
            if version.startswith("cuda_") and triton_time_us is not None:
                speedup = triton_time_us / result.time_us
                speedup_vs_triton_str = f"{speedup:.2f}x"

            rows.append(
                [
                    version,
                    f"({shape[0]}, {shape[1]})",
                    f"{result.time_us:.2f}",
                    round(result.mem_bw_gbps, 3),
                    speedup_vs_torch_str,
                    speedup_vs_triton_str,
                ]
            )

        # Find best CUDA kernel speedup vs triton for this shape
        best_cuda_speedup = 0.0
        best_cuda_version = None
        for version, result in versions.items():
            if version.startswith("cuda_") and triton_time_us is not None:
                speedup = triton_time_us / result.time_us
                if speedup > best_cuda_speedup:
                    best_cuda_speedup = speedup
                    best_cuda_version = version

        if best_cuda_version is not None:
            rows.append(
                [
                    f">>> BEST: {best_cuda_speedup:.2f}x vs triton with {best_cuda_version}",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        # Add empty row for visual separation between shapes
        rows.append([""] * len(headers))

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
