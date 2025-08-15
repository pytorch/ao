# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
import argparse
import itertools
import time
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from torchao.prototype.moe_training import _scaled_grouped_mm

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    A_shape: tuple[int]
    B_shape: tuple[int]


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    A_shapes = [(2**8, 8192), (2**12, 8192), (2**16, 8192)]
    B_shapes = [(4, 8192, 8192), (8, 8192, 8192), (16, 8192, 8192)]
    high_precision_dtypes = [torch.bfloat16]
    configs = []
    for A_shape, B_shape, high_precision_dtype in itertools.product(
        A_shapes,
        B_shapes,
        high_precision_dtypes,
    ):
        configs.append(
            ExperimentConfig(
                A_shape=A_shape,
                B_shape=B_shape,
                high_precision_dtype=high_precision_dtype,
            )
        )
    return configs


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    # define test inputs
    A = torch.randn(
        *config.A_shape,
        dtype=config.high_precision_dtype,
        device=device,
        requires_grad=True,
    )
    B_t = torch.randn(
        *config.B_shape,
        dtype=config.high_precision_dtype,
        device=device,
        requires_grad=True,
    ).transpose(-2, -1)

    # - configure input to be row-major with groups divided along the column dimension,
    #   representing the left operand of grad_weight = grad_output_t @ input
    #   that occurs in the backward pass of the differentiable scaled grouped mm.
    # - the transposed tensor in col-major format with groups along the row dimension,
    #    which represents the right operand.
    n_groups = config.B_shape[0]
    group_size = A.shape[0] // n_groups
    offs = torch.arange(
        group_size,
        group_size * n_groups + 1,
        group_size,
        device=device,
        dtype=torch.int32,
    )

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    def forward_backward(A, B_t, offs):
        out = _scaled_grouped_mm(
            A,
            B_t,
            offs=offs,
            out_dtype=torch.bfloat16,
        )
        out.sum().backward()
        torch.cuda.synchronize()

    # benchmark torch
    torch_func = torch.compile(forward_backward) if args.compile else forward_backward
    warmup(torch_func, A, B_t, offs)
    start_time_ns = time.perf_counter_ns()
    torch_func(A, B_t, offs)
    torch_time_ns = time.perf_counter_ns() - start_time_ns
    time_us = torch_time_ns / 1e3

    return ExperimentResult(
        time_us=round(time_us, 3),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "A_shape",
        "B_shape",
        "time_us",
    ]
    rows = []
    for experiment in experiments:
        A_shape = f"({experiment.config.A_shape[0]}, {experiment.config.A_shape[1]})"
        B_shape = f"({experiment.config.B_shape[0]}, {experiment.config.B_shape[1]}, {experiment.config.B_shape[2]})"
        rows.append(
            [
                A_shape,
                B_shape,
                experiment.result.time_us,
            ]
        )
    print(tabulate(rows, headers=headers))


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
    arg_parser.add_argument("--compile", action="store_true")
    args = arg_parser.parse_args()
    main(args)
