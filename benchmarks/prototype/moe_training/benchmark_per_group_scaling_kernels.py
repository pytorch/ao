# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.jagged_float8_scales import (
    triton_fp8_col_major_jagged_colwise_scales,
    triton_fp8_row_major_jagged_rowwise_scales,
)
from torchao.prototype.moe_training.utils import (
    torch_to_float8_per_group_colwise,
    torch_to_float8_per_group_rowwise,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    input_shape: tuple[int]
    n_groups: int


@dataclass(frozen=True)
class ExperimentResult:
    torch_time_us: float
    triton_time_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [(2**8, 4096), (2**12, 4096), (2**16, 4096)]
    n_groups_list = [4, 8, 16]
    high_precision_dtypes = [torch.bfloat16]
    configs = []
    for input_shape, n_groups, high_precision_dtype in itertools.product(
        input_shapes, n_groups_list, high_precision_dtypes
    ):
        configs.append(
            ExperimentConfig(
                input_shape=input_shape,
                n_groups=n_groups,
                high_precision_dtype=high_precision_dtype,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # define test inputs
    input_tensor = torch.randn(
        *config.input_shape,
        dtype=config.high_precision_dtype,
        device=device,
    )
    input_row_major = input_tensor.clone().detach()
    input_col_major = input_tensor.clone().detach().t()

    # - configure input to be row-major with groups divided along the column dimension,
    #   representing the left operand of grad_weight = grad_output_t @ input
    #   that occurs in the backward pass of the differentiable scaled grouped mm.
    # - the transposed tensor in col-major format with groups along the row dimension,
    #    which represents the right operand.
    group_size = input_row_major.shape[1] // config.n_groups
    n_groups = config.n_groups
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

    def run_torch(
        input_row_major: torch.Tensor, input_col_major: torch.Tensor, offs: torch.Tensor
    ):
        _ = torch_to_float8_per_group_rowwise(
            input_row_major,
            offs,
            target_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )
        _ = torch_to_float8_per_group_colwise(
            input_col_major,
            offs,
            target_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

    def run_triton(
        input_row_major: torch.Tensor, input_col_major: torch.Tensor, offs: torch.Tensor
    ):
        _ = triton_fp8_row_major_jagged_rowwise_scales(
            input_row_major,
            offs,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )
        _ = triton_fp8_col_major_jagged_colwise_scales(
            input_col_major,
            offs,
            output_dtype=torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

    # bench torch
    compiled_run_torch = torch.compile(run_torch)
    torch_time_us = benchmark_cuda_function_in_microseconds(
        compiled_run_torch, input_row_major, input_col_major, offs
    )

    # bench triton
    warmup(run_triton, input_row_major, input_col_major, offs)
    triton_time_us = benchmark_cuda_function_in_microseconds(
        run_triton, input_row_major, input_col_major, offs
    )

    return ExperimentResult(
        torch_time_us=torch_time_us,
        triton_time_us=triton_time_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "n_groups",
        "high_precision_dtype",
        "torch_time_us",
        "triton_time_us",
    ]
    rows = []
    for experiment in experiments:
        input_shape = (
            f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]})"
        )
        rows.append(
            [
                input_shape,
                experiment.config.n_groups,
                experiment.config.high_precision_dtype,
                experiment.result.torch_time_us,
                experiment.result.triton_time_us,
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


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
