# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
import argparse
import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from utils import benchmark_cuda_function_in_microseconds

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import (
    to_blocked_per_group_2d,
    to_blocked_per_group_3d,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    e: int
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ExperimentResult:
    bf16_us: float
    fp8_rowwise_us: float
    mxfp8_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes
    M = [16640]
    K = [5120]
    N = [8192]
    E = [16]
    configs = []
    for e, m, n, k in itertools.product(
        E,
        M,
        N,
        K,
    ):
        configs.append(
            ExperimentConfig(
                e=e,
                m=m,
                n=n,
                k=k,
            )
        )
    return configs


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    e, m, n, k = config.e, config.m, config.n, config.k

    # define test inputs
    A = torch.randn(
        (m, k),
        dtype=torch.bfloat16,
        device=device,
    )
    B_t = torch.randn(
        (e, n, k),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    ).transpose(-2, -1)

    # Configure groups
    n_groups = e
    Mg = A.shape[0]
    alignment_size = 16
    offs = generate_jagged_offs(n_groups, Mg, multiple_of=alignment_size)

    # benchmark bf16 grouped mm
    bf16_us = benchmark_cuda_function_in_microseconds(
        torch._grouped_mm,
        A,
        B_t,
        offs,
        out_dtype=torch.bfloat16,
    )

    # bench fp8 rowwise grouped mm
    fp8_rowwise_us = bench_fp8_rowwise_grouped_mm(A, B_t, offs)

    # benchmark mxfp8 grouped mm
    mxfp8_us = bench_mxfp8_grouped_mm(A, B_t, offs)

    return ExperimentResult(
        bf16_us=round(bf16_us, 3),
        fp8_rowwise_us=round(fp8_rowwise_us, 3),
        mxfp8_us=round(mxfp8_us, 3),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "E",
        "M",
        "N",
        "K",
        "bf16_time_us",
        "fp8_rowwise_time_us",
        "mxfp8_time_us",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                experiment.config.e,
                experiment.config.m,
                experiment.config.n,
                experiment.config.k,
                experiment.result.bf16_us,
                experiment.result.fp8_rowwise_us,
                experiment.result.mxfp8_us,
            ]
        )
    print(tabulate(rows, headers=headers))


# benchmark fp8 grouped mm
def bench_fp8_rowwise_grouped_mm(A, B_t, offs) -> float:
    # Convert A to float8, row-major for left operand of grouped GEMM.
    A_scales = tensor_to_scale(
        A,
        torch.float8_e4m3fn,
        scaling_granularity=ScalingGranularity.AXISWISE,
        axiswise_dim=-1,
        round_scales_to_power_of_2=True,
    )
    A_scaled = A.to(torch.float32) * A_scales
    A_fp8_row_major = to_fp8_saturated(A_scaled, torch.float8_e4m3fn)

    # Convert B_t to float8, column-major for right operand of grouped GEMM.
    B_t_scales = tensor_to_scale(
        B_t,
        torch.float8_e4m3fn,
        scaling_granularity=ScalingGranularity.AXISWISE,
        axiswise_dim=-2,
        round_scales_to_power_of_2=True,
    )
    B_t_scaled = B_t.to(torch.float32) * B_t_scales
    B_t_fp8_col_major = to_fp8_saturated(B_t_scaled, torch.float8_e4m3fn)

    # Bench the gemm
    fp8_us = benchmark_cuda_function_in_microseconds(
        torch._scaled_grouped_mm,
        A_fp8_row_major,
        B_t_fp8_col_major,
        A_scales.squeeze(1).reciprocal(),
        B_t_scales.squeeze(1).reciprocal(),
        offs,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    return fp8_us


def bench_mxfp8_grouped_mm(A, B_t, offs, block_size=32) -> float:
    # A_mx shape: (M, K)
    # A_scale shape: (M, K//block_size)
    A_scales, A_fp8 = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

    # B_mx shape: (E, N, K)
    # B_scale shape: (E, N, K//block_size)
    B_scales, B_fp8 = to_mx(
        B_t.transpose(-2, -1),
        elem_dtype=torch.float8_e4m3fn,
        block_size=block_size,
    )

    # Convert scales for each group to blocked format.
    Mg, K = A_fp8.shape
    A_scales_blocked, starting_row_after_padding = to_blocked_per_group_2d(
        A_scales, offs, Mg, K
    )
    B_scales_blocked = to_blocked_per_group_3d(B_scales)

    # From this, we compute `group_sizes` and `starting_row_after_padding`:
    # group_sizes = [32, 32, 64]
    # starting_row_after_padding = [0, 32, 64, 128]
    zero = torch.tensor([0], dtype=offs.dtype, device=offs.device)
    group_sizes = torch.diff(offs, prepend=zero).to(torch.int64)

    # Run the grouped mm
    mxfp8_us = benchmark_cuda_function_in_microseconds(
        torch.ops.fbgemm.mx8mx8bf16_grouped_stacked,
        A_fp8,
        B_fp8,
        A_scales_blocked,
        B_scales_blocked,
        group_sizes,
        starting_row_after_padding=starting_row_after_padding,
    )
    return mxfp8_us


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    args = arg_parser.parse_args()
    main(args)
