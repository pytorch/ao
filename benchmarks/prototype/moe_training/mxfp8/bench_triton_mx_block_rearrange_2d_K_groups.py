# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from torch.utils.cpp_extension import load
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

# Build CUDA kernel directly using torch.utils.cpp_extension.load
mxfp8_cuda = None
try:
    # Get the kernel source directory
    KERNEL_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "torchao",
        "csrc",
        "cuda",
        "mx_kernels",
    )
    KERNEL_DIR = os.path.normpath(KERNEL_DIR)

    print("Compiling CUDA kernel...")
    mxfp8_cuda = load(
        name="mxfp8_cuda",
        sources=[
            os.path.join(KERNEL_DIR, "mxfp8_extension.cpp"),
            os.path.join(KERNEL_DIR, "mxfp8_cuda.cu"),
            os.path.join(KERNEL_DIR, "mx_block_rearrange_2d_K_groups.cu"),
        ],
        extra_cuda_cflags=[
            "-lineinfo",
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-gencode=arch=compute_100,code=sm_100",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )
    print("✓ CUDA kernel compilation successful!")
except (ImportError, RuntimeError) as e:
    print(f"⚠ CUDA kernel not available: {e}")
    print("The benchmark will only run 'naive' and 'parallel' Triton versions.\n")

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
        (5120, 16384 // block_size),
        (5120, 131072 // block_size),
        (8192, 16384 // block_size),
        (8192, 131072 // block_size),
        (7168, 16384 // block_size),
        (7168, 131072 // block_size),
        (2048, 16384 // block_size),
        (2048, 131072 // block_size),
    ]
    num_groups = [8]
    versions = [
        "triton",
        "cuda_rowmajor",
        "cuda_colmajor",
        "cuda_colmajor_vec",
        "cuda_colmajor_vec_16B",
        "cuda_rowmajor_vec",
        "cuda_rowmajor_128x4_vec_64",  # max_cols=64: 512 threads, 8KB SMEM
        "cuda_rowmajor_128x4_vec_128",  # max_cols=128: 1024 threads, 16KB SMEM
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
    if version == "triton":
        kernel_fn = triton_mx_block_rearrange_2d_K_groups
        # Triton uses row-major input
        kernel_input = input_tensor
    elif version == "cuda_rowmajor":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = mxfp8_cuda.mx_block_rearrange_2d_K_groups_rowmajor
        # Row-major kernel expects contiguous row-major input
        kernel_input = input_tensor.contiguous()
    elif version == "cuda_colmajor":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = mxfp8_cuda.mx_block_rearrange_2d_K_groups_colmajor
        # Column-major kernel expects column-major input
        # Column-major: same shape (rows, cols) but stride(0)=1, stride(1)=rows
        kernel_input = input_tensor.T.contiguous().T
    elif version == "cuda_colmajor_vec":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = mxfp8_cuda.mx_block_rearrange_2d_K_groups_colmajor_vectorized
        # Vectorized column-major kernel also expects column-major input
        kernel_input = input_tensor.T.contiguous().T
    elif version == "cuda_colmajor_vec_16B":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = mxfp8_cuda.mx_block_rearrange_2d_K_groups_colmajor_vectorized_16B
        # 16B vectorized column-major kernel also expects column-major input
        kernel_input = input_tensor.T.contiguous().T
    elif version == "cuda_rowmajor_vec":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = mxfp8_cuda.mx_block_rearrange_2d_K_groups_rowmajor_vectorized
        # Row-major vectorized kernel expects contiguous row-major input
        kernel_input = input_tensor.contiguous()
    elif version == "cuda_rowmajor_128x4_vec":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = (
            lambda t, o: mxfp8_cuda.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec(
                t, o, 64
            )
        )
        # Row-major 128x4 vectorized kernel expects contiguous row-major input
        kernel_input = input_tensor.contiguous()
    elif version == "cuda_rowmajor_128x4_vec_64":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = (
            lambda t, o: mxfp8_cuda.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec(
                t, o, 64
            )
        )
        # Row-major 128x4 vectorized kernel with max_cols=64 (512 threads, 8KB SMEM)
        kernel_input = input_tensor.contiguous()
    elif version == "cuda_rowmajor_128x4_vec_128":
        if mxfp8_cuda is None:
            raise RuntimeError("CUDA kernel not available")
        kernel_fn = (
            lambda t, o: mxfp8_cuda.mx_block_rearrange_2d_K_groups_rowmajor_128x4_vec(
                t, o, 128
            )
        )
        # Row-major 128x4 vectorized kernel with max_cols=128 (1024 threads, 16KB SMEM)
        kernel_input = input_tensor.contiguous()
    else:
        raise ValueError(f"Unknown version: {version}")

    # Run kernel to get output shape
    out_scales = kernel_fn(
        kernel_input,
        input_group_offsets,
    )

    # Benchmark the kernel
    # Note: column-major tensors are not "contiguous" in PyTorch's row-major sense,
    # but they are valid and have the expected strides for the CUDA kernel
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
        "input_shape",
        "time_us",
        "mem_bw_gbps",
        "fastest_version",
        "speedup_vs_triton",
    ]

    rows = []
    for shape, versions in shapes_dict.items():
        # Find fastest version for this shape
        fastest_version = min(versions.items(), key=lambda x: x[1].time_us)[0]

        # Get triton baseline time for speedup calculation
        triton_time_us = (
            versions.get("triton").time_us if "triton" in versions else None
        )

        # Add rows for each version
        for version, result in versions.items():
            # Calculate speedup vs triton
            speedup_str = ""
            if version != "triton":
                speedup = triton_time_us / result.time_us
                speedup_str = f"{speedup:.2f}x"

            rows.append(
                [
                    version,
                    f"({shape[0]}, {shape[1]})",
                    f"{result.time_us:.2f}",
                    round(result.mem_bw_gbps, 3),
                    fastest_version,
                    speedup_str,
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
