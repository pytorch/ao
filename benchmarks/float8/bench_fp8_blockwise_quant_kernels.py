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
from utils import benchmark_microseconds

from torchao.prototype.blockwise_fp8.kernels import (
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_quant,
    torch_blockwise_scale_act_quant,
    torch_blockwise_scale_weight_quant,
    triton_quantize_fp8_block,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    A_shape: tuple[int]
    block_m: int
    block_k: int


@dataclass(frozen=True)
class ExperimentResult:
    torch_us: float
    fbgemm_us: float
    deepgemm_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    A_shapes = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
    ]
    block_m_opts = [1, 128]
    block_k_opts = [
        128,
    ]
    configs = []
    for A_shape, block_m, block_k in itertools.product(
        A_shapes,
        block_m_opts,
        block_k_opts,
    ):
        configs.append(
            ExperimentConfig(
                A_shape=A_shape,
                block_m=block_m,
                block_k=block_k,
            )
        )
    return configs


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    A = torch.randn(
        *config.A_shape,
        dtype=torch.bfloat16,
        device=device,
    )

    # Torch and DeepGEMM implementations are specific to activation quantization (1 x block_size)
    # and weight quantization (block_size x block_size)
    if config.block_m == 1:
        torch_func = torch.compile(torch_blockwise_scale_act_quant)
        deepgemm_func = fp8_blockwise_act_quant
    else:
        torch_func = torch.compile(torch_blockwise_scale_weight_quant)
        deepgemm_func = fp8_blockwise_weight_quant

    # Validate output shapes and strides
    torch_out, torch_scale = torch_func(A, tile_size=config.block_k)
    deepgemm_out, deepgemm_scale = deepgemm_func(A, block_size=config.block_k)
    fbgemm_out, fbgemm_scale = triton_quantize_fp8_block(
        A, block_m=config.block_m, block_k=config.block_k, k_major=True
    )
    assert torch_out.shape == deepgemm_out.shape == fbgemm_out.shape
    assert torch_out.stride() == deepgemm_out.stride() == fbgemm_out.stride()
    assert torch_scale.shape == deepgemm_scale.shape == fbgemm_scale.shape
    assert torch_scale.stride() == deepgemm_scale.stride() == fbgemm_scale.stride()

    # Do benchmarking
    torch_us = benchmark_microseconds(torch_func, A, tile_size=config.block_k)
    deepgemm_us = benchmark_microseconds(
        fp8_blockwise_act_quant, A, block_size=config.block_k
    )
    fbgemm_us = benchmark_microseconds(
        triton_quantize_fp8_block,
        A,
        block_m=config.block_m,
        block_k=config.block_k,
        k_major=True,
    )

    return ExperimentResult(
        torch_us=round(torch_us, 3),
        fbgemm_us=round(fbgemm_us, 3),
        deepgemm_us=round(deepgemm_us, 3),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "A_shape",
        "block_shape",
        "torch_us",
        "fbgemm_us",
        "deepgemm_us",
    ]
    rows = []
    for experiment in experiments:
        A_shape = f"({experiment.config.A_shape[0]}, {experiment.config.A_shape[1]})"
        block_shape = f"({experiment.config.block_m},{experiment.config.block_k})"
        rows.append(
            [
                A_shape,
                block_shape,
                experiment.result.torch_us,
                experiment.result.fbgemm_us,
                experiment.result.deepgemm_us,
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
