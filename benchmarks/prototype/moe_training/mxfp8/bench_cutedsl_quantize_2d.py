# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import argparse
import itertools
from dataclasses import dataclass
from typing import List, Optional

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    fused_pad_token_groups_cuda,
    mx_block_rearrange_2d_M_groups_cuda,
)
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_quantize_2d import (
    mxfp8_quantize_cutedsl_2d,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    scaling_mode: str
    num_groups: int
    group_alignment: Optional[int] = None


@dataclass(frozen=True)
class ExperimentResult:
    # time
    cutedsl_blocked_us: float
    triton_plus_rearrange_us: float
    # mem bw
    cutedsl_blocked_gbps: float
    triton_plus_rearrange_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs(group_alignment: Optional[int] = None) -> List[ExperimentConfig]:
    input_shapes = [
        # DeepSeekV3 671b shapes
        (8192, 2048),
        (8192, 7168),
        (32768, 2048),
        (32768, 7168),
        (131072, 2048),
        (131072, 7168),
    ]
    scaling_modes = ["rceil"]
    num_groups_list = [8]
    configs = []
    for shape, scaling_mode, num_groups in itertools.product(
        input_shapes, scaling_modes, num_groups_list
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                scaling_mode=scaling_mode,
                num_groups=num_groups,
                group_alignment=group_alignment,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    block_size = 32
    input_shape = config.input_shape
    scaling_mode = config.scaling_mode
    num_groups = config.num_groups
    group_alignment = config.group_alignment

    input_tensor = torch.randn(
        *input_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    M, K = input_shape

    # Generate jagged offsets
    if group_alignment:
        # When using group alignment, allow variable group sizes
        group_end_offsets = generate_jagged_offs(
            num_groups, M, multiple_of=1, device=device
        )
    else:
        # Without group alignment, use multiples of 128 to avoid per-group padding
        group_end_offsets = generate_jagged_offs(
            num_groups, M, multiple_of=128, device=device
        )

    # Benchmark 1: CuTeDSL kernel with blocked scale output
    if group_alignment:
        data_cutedsl, scales_cutedsl = mxfp8_quantize_cutedsl_2d(
            input_tensor,
            block_size=block_size,
            scaling_mode=scaling_mode,
            blocked_scale_output=True,
            group_end_offsets=group_end_offsets,
            group_alignment_size=group_alignment,
        )
        cutedsl_blocked_time_us = benchmark_cuda_function_in_microseconds(
            mxfp8_quantize_cutedsl_2d,
            input_tensor,
            block_size=block_size,
            scaling_mode=scaling_mode,
            blocked_scale_output=True,
            group_end_offsets=group_end_offsets,
            group_alignment_size=group_alignment,
        )
    else:
        data_cutedsl, scales_cutedsl = mxfp8_quantize_cutedsl_2d(
            input_tensor,
            block_size=block_size,
            scaling_mode=scaling_mode,
            blocked_scale_output=True,
        )
        cutedsl_blocked_time_us = benchmark_cuda_function_in_microseconds(
            mxfp8_quantize_cutedsl_2d,
            input_tensor,
            block_size=block_size,
            scaling_mode=scaling_mode,
            blocked_scale_output=True,
        )

    # Benchmark 2: Triton quantization + CUDA scale rearrangement
    if group_alignment:
        # Pre-pad inputs for triton path when using group alignment
        def triton_plus_rearrange(x, group_offs, alignment_size):
            # Pad token groups to alignment
            padded_x, _, padded_group_end_offs = fused_pad_token_groups_cuda(
                x, group_offs, alignment_size
            )
            # Quantize along dim0 (rowwise)
            data, scales = triton_to_mxfp8_dim0(
                padded_x,
                inner_block_size=block_size,
                scaling_mode=scaling_mode,
            )
            # Convert scales to blocked layout
            scales_blocked = mx_block_rearrange_2d_M_groups_cuda(
                scales.view(torch.uint8), padded_group_end_offs
            )
            return data, scales_blocked

        data_triton, scales_triton = triton_plus_rearrange(
            input_tensor, group_end_offsets, group_alignment
        )
        triton_plus_rearrange_time_us = benchmark_cuda_function_in_microseconds(
            triton_plus_rearrange,
            input_tensor,
            group_end_offsets,
            group_alignment,
        )
    else:

        def triton_plus_rearrange(x, group_offs):
            # Quantize along dim0 (rowwise)
            data, scales = triton_to_mxfp8_dim0(
                x,
                inner_block_size=block_size,
                scaling_mode=scaling_mode,
            )
            # Convert scales to blocked layout
            scales_blocked = mx_block_rearrange_2d_M_groups_cuda(
                scales.view(torch.uint8), group_offs
            )
            return data, scales_blocked

        data_triton, scales_triton = triton_plus_rearrange(
            input_tensor, group_end_offsets
        )
        triton_plus_rearrange_time_us = benchmark_cuda_function_in_microseconds(
            triton_plus_rearrange,
            input_tensor,
            group_end_offsets,
        )

    # Memory bandwidth calculations
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = (
        data_cutedsl.numel() * bytes_per_output_el
        + scales_cutedsl.numel() * bytes_per_scale_el
    )

    cutedsl_blocked_gbps = ((read_bytes + write_bytes) / 1e9) / (
        cutedsl_blocked_time_us / 1e6
    )
    triton_plus_rearrange_gbps = ((read_bytes + write_bytes) / 1e9) / (
        triton_plus_rearrange_time_us / 1e6
    )

    return ExperimentResult(
        cutedsl_blocked_us=cutedsl_blocked_time_us,
        triton_plus_rearrange_us=triton_plus_rearrange_time_us,
        cutedsl_blocked_gbps=cutedsl_blocked_gbps,
        triton_plus_rearrange_gbps=triton_plus_rearrange_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "scaling_mode",
        "num_groups",
        "group_align",
        "cutedsl_blocked_us",
        "pad+triton+rearrange_us",
        "speedup",
        "cutedsl_gbps",
        "pad+triton+rearrange_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = (
            experiment.result.triton_plus_rearrange_us
            / experiment.result.cutedsl_blocked_us
        )
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.scaling_mode,
                experiment.config.num_groups,
                experiment.config.group_alignment or "None",
                f"{experiment.result.cutedsl_blocked_us:.2f}",
                f"{experiment.result.triton_plus_rearrange_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.cutedsl_blocked_gbps:.1f}",
                f"{experiment.result.triton_plus_rearrange_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MXFP8 quantization with CuTeDSL and Triton"
    )
    parser.add_argument(
        "--group-alignment",
        type=int,
        default=None,
        help="Alignment size for padding token groups along M dimension (e.g., 128). "
        "If not specified, groups are generated as multiples of 128 to avoid padding.",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    configs = get_configs(group_alignment=args.group_alignment)
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print("\nBenchmark Results:")
    if args.group_alignment:
        print(
            f"Group alignment enabled: padding to multiples of {args.group_alignment}\n"
        )
    else:
        print("Group alignment disabled: using 128-aligned groups\n")
    print_results(results)


if __name__ == "__main__":
    main()
