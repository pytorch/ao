# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import (
    bench_fwd_bwd_microseconds,
    bench_fwd_microseconds,
    profile_fwd_bwd,
)
from torchao.prototype.moe_training import _scaled_grouped_mm
from torchao.prototype.moe_training.conversion_utils import MoEScalingType
from torchao.prototype.moe_training.utils import generate_jagged_offs

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

# Workaround for https://github.com/pytorch/ao/pull/2990#issuecomment-3285762681
torch._dynamo.config.automatic_dynamic_shapes = False


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    A_shape: tuple[int]
    B_shape: tuple[int]
    recipe: MoEScalingType


@dataclass(frozen=True)
class ExperimentResult:
    bf16_e2e_us: float
    scaled_e2e_us: float
    scaled_e2e_speedup: float
    bf16_fwd_us: float
    scaled_fwd_us: float
    scaled_fwd_speedup: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Llama4 shapes
    A_shapes = [(16640, 5120)]
    B_shapes = [(1, 8192, 5120), (4, 8192, 5120), (16, 8192, 5120), (64, 8192, 5120)]
    recipes = [MoEScalingType.FP8_ROWWISE, MoEScalingType.MXFP8]
    high_precision_dtypes = [torch.bfloat16]
    configs = []
    for A_shape, B_shape, recipe, high_precision_dtype in itertools.product(
        A_shapes,
        B_shapes,
        recipes,
        high_precision_dtypes,
    ):
        configs.append(
            ExperimentConfig(
                A_shape=A_shape,
                B_shape=B_shape,
                recipe=recipe,
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
    Mg = A.shape[0]
    token_group_alignment_size = 32 if config.recipe == MoEScalingType.MXFP8 else 16
    offs = generate_jagged_offs(n_groups, Mg, multiple_of=token_group_alignment_size)

    labels = torch.ones(
        (A.shape[0], B_t.shape[-1]), device=device, dtype=torch.bfloat16
    )

    # E2E bf16 benchmark + profiling
    bf16_e2e_us = bench_fwd_bwd_microseconds(
        torch._grouped_mm,
        A,
        B_t,
        offs,
        labels=labels,
        use_compile=args.compile,
        fullgraph=False,
    )
    if args.profile:
        profile_fwd_bwd(
            torch._grouped_mm,
            A,
            B_t,
            offs,
            labels=labels,
            use_compile=args.compile,
            fullgraph=False,
            profile_name="bf16_profile",
        )

    # E2E scaled benchmark + profiling
    scaled_e2e_us = bench_fwd_bwd_microseconds(
        _scaled_grouped_mm,
        A,
        B_t,
        offs,
        scaling_type=config.recipe,
        labels=labels,
        use_compile=args.compile,
        fullgraph=False,
    )
    if args.profile:
        profile_fwd_bwd(
            _scaled_grouped_mm,
            A,
            B_t,
            offs,
            scaling_type=config.recipe,
            labels=labels,
            use_compile=args.compile,
            profile_name="scaled_profile",
            fullgraph=False,
        )

    # Forward pass benchmarks
    bf16_fwd_us = bench_fwd_microseconds(
        torch._grouped_mm,
        A,
        B_t,
        offs,
        use_compile=args.compile,
        fullgraph=True,
    )
    scaled_fwd_us = bench_fwd_microseconds(
        _scaled_grouped_mm,
        A,
        B_t,
        offs,
        scaling_type=config.recipe,
        use_compile=args.compile,
        fullgraph=True,
    )

    return ExperimentResult(
        bf16_e2e_us=round(bf16_e2e_us, 3),
        scaled_e2e_us=round(scaled_e2e_us, 3),
        scaled_e2e_speedup=round(bf16_e2e_us / scaled_e2e_us, 3),
        bf16_fwd_us=round(bf16_fwd_us, 3),
        scaled_fwd_us=round(scaled_fwd_us, 3),
        scaled_fwd_speedup=round(bf16_fwd_us / scaled_fwd_us, 3),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "A_shape",
        "B_shape",
        "recipe",
        "bf16_e2e_us",
        "scaled_e2e_us",
        "scaled_e2e_speedup",
        "bf16_fwd_us",
        "scaled_fwd_us",
        "scaled_fwd_speedup",
    ]
    rows = []
    for experiment in experiments:
        A_shape = f"({experiment.config.A_shape[0]}, {experiment.config.A_shape[1]})"
        B_shape = f"({experiment.config.B_shape[0]}, {experiment.config.B_shape[1]}, {experiment.config.B_shape[2]})"
        rows.append(
            [
                A_shape,
                B_shape,
                experiment.config.recipe,
                experiment.result.bf16_e2e_us,
                experiment.result.scaled_e2e_us,
                f"{experiment.result.scaled_e2e_speedup}x",
                experiment.result.bf16_fwd_us,
                experiment.result.scaled_fwd_us,
                f"{experiment.result.scaled_fwd_speedup}x",
            ]
        )
    print(tabulate(rows, headers=headers))


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        if (
            config.recipe == MoEScalingType.FP8_ROWWISE
            and torch.cuda.get_device_capability() != (9, 0)
        ):
            logging.warning(
                f"Skipping FP8 rowwise benchmarks, only supported on compute capability 9.0 and found {torch.cuda.get_device_capability()}"
            )
            continue

        elif (
            config.recipe == MoEScalingType.MXFP8
            and torch.cuda.get_device_capability() != (10, 0)
        ):
            logging.warning(
                f"Skipping MXFP8 benchmarks, only supported on compute capability 10.0 and found {torch.cuda.get_device_capability()}"
            )
            continue

        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--compile", action="store_true")
    arg_parser.add_argument("--profile", action="store_true")
    args = arg_parser.parse_args()
    main(args)
