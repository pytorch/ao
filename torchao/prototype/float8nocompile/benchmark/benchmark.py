# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import Callable, List

import torch
from tabulate import tabulate
from torch import nn
from torch._inductor.utils import do_bench_using_profiling
from torch.nn import functional as F
from tqdm import tqdm

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.prototype.float8nocompile.float8nocompile_linear_utils import (
    convert_to_float8_nocompile_training,
)
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    layer_sizes: list[int]
    input_shape: tuple[int]
    kernel_algo: KernelAlgorithm


@dataclass(frozen=True)
class ExperimentResult:
    float8nocompile_time: float
    eager_time: float
    compiled_time: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


class TestModel(nn.Module):
    def __init__(self, layer_sizes=[32, 64, 32]):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False)
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def get_configs() -> List[ExperimentConfig]:
    algos = [KernelAlgorithm.ATOMIC_MAX, KernelAlgorithm.REDUCTION]
    layer_sizes = [[4096, 4096]]
    input_shapes = [(2**4, 4096), (2**8, 4096), (2**12, 4096), (2**16, 4096)]
    high_precision_dtypes = [torch.bfloat16]
    configs = []
    for algo, layer_size, input_shape, high_precision_dtype in itertools.product(
        algos, layer_sizes, input_shapes, high_precision_dtypes
    ):
        configs.append(
            ExperimentConfig(
                layer_sizes=layer_size,
                input_shape=input_shape,
                high_precision_dtype=high_precision_dtype,
                kernel_algo=algo,
            )
        )
    return configs


def forward_backward(model, input_tensor):
    output = model(input_tensor)
    loss = F.mse_loss(output, torch.zeros_like(output))
    loss.backward()


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # eager float8 baseline
    eager_float8_model = convert_to_float8_training(
        TestModel(config.layer_sizes).to(device)
    )

    # compiled float8 baseline
    compiled_float8_model = torch.compile(eager_float8_model, fullgraph=True)

    # float8nocompile triton implementation
    float8nocompile_model = convert_to_float8_nocompile_training(
        TestModel(config.layer_sizes).to(device), kernel_algo=config.kernel_algo
    )

    # define test inputs
    input_tensor = torch.randn(
        *config.input_shape,
        requires_grad=True,
        dtype=config.high_precision_dtype,
        device=device,
    )
    input_eager = input_tensor.clone().detach().requires_grad_(True)
    input_compiled = input_tensor.clone().detach().requires_grad_(True)
    input_triton = input_tensor.clone().detach().requires_grad_(True)

    # benchmark forward + backward for each model
    eager_time = benchmark_cuda_function_in_microseconds(
        forward_backward,
        eager_float8_model,
        input_eager,
    )

    compiled_time = benchmark_cuda_function_in_microseconds(
        forward_backward,
        compiled_float8_model,
        input_compiled,
    )

    float8nocompile_time = benchmark_cuda_function_in_microseconds(
        forward_backward,
        float8nocompile_model,
        input_triton,
    )

    return ExperimentResult(
        eager_time=eager_time,
        compiled_time=compiled_time,
        float8nocompile_time=float8nocompile_time,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "kernel_algo",
        "high_precision_dtype",
        "eager_time",
        "compiled_time",
        "float8nocompile",
    ]
    rows = []
    for experiment in experiments:
        input_shape = (
            f"({experiment.config.input_shape[0]}, {experiment.config.input_shape[1]})"
        )
        rows.append(
            [
                input_shape,
                str(experiment.config.kernel_algo),
                experiment.config.high_precision_dtype,
                experiment.result.eager_time,
                experiment.result.compiled_time,
                experiment.result.float8nocompile_time,
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


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
