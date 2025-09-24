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

# Dynamic shapes hurt performance
torch._dynamo.config.automatic_dynamic_shapes = False


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    MNKG: tuple[int]
    recipe: MoEScalingType


@dataclass(frozen=True)
class ExperimentResult:
    bf16_fwd_bwd_us: float
    scaled_fwd_bwd_us: float
    scaled_fwd_bwd_speedup: float
    bf16_fwd_us: float
    scaled_fwd_us: float
    scaled_fwd_speedup: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    MNKG_list = [
        # Llama4 16e with various experts per device (i.e., different EP degrees)
        (16384, 8192, 5120, 1),
        (16384, 8192, 5120, 2),
        (16384, 8192, 5120, 4),
        (16384, 8192, 5120, 8),
        (128000, 8192, 5120, 1),
        (128000, 8192, 5120, 2),
        (128000, 8192, 5120, 4),
        (128000, 8192, 5120, 8),
        # DSV3 236B with various experts per device (i.e., different EP degrees)
        (16384, 1536, 5120, 1),
        (16384, 1536, 5120, 2),
        (16384, 1536, 5120, 4),
        (16384, 1536, 5120, 8),
        (128000, 1536, 5120, 1),
        (128000, 1536, 5120, 2),
        (128000, 1536, 5120, 4),
        (128000, 1536, 5120, 8),
        # DSV3 671B with various experts per device (i.e., different EP degrees)
        (16384, 2048, 7168, 1),
        (16384, 2048, 7168, 2),
        (16384, 2048, 7168, 4),
        (16384, 2048, 7168, 8),
        (128000, 2048, 7168, 1),
        (128000, 2048, 7168, 2),
        (128000, 2048, 7168, 4),
        (128000, 2048, 7168, 8),
    ]
    recipes = [MoEScalingType.FP8_ROWWISE, MoEScalingType.MXFP8]
    high_precision_dtypes = [torch.bfloat16]
    configs = []
    for MNKG, recipe, high_precision_dtype in itertools.product(
        MNKG_list,
        recipes,
        high_precision_dtypes,
    ):
        configs.append(
            ExperimentConfig(
                MNKG=MNKG,
                recipe=recipe,
                high_precision_dtype=high_precision_dtype,
            )
        )
    return configs


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    total_M, N, K, G = config.MNKG

    # define test inputs
    A = torch.randn(
        (total_M, K),
        dtype=config.high_precision_dtype,
        device=device,
        requires_grad=True,
    )
    B_t = torch.randn(
        (G, N, K),
        dtype=config.high_precision_dtype,
        device=device,
        requires_grad=True,
    ).transpose(-2, -1)

    # - configure input to be row-major with groups divided along the column dimension,
    #   representing the left operand of grad_weight = grad_output_t @ input
    #   that occurs in the backward pass of the differentiable scaled grouped mm.
    # - the transposed tensor in col-major format with groups along the row dimension,
    #    which represents the right operand.
    token_group_alignment_size = 32 if config.recipe == MoEScalingType.MXFP8 else 16
    offs = generate_jagged_offs(G, total_M, multiple_of=token_group_alignment_size)

    labels = torch.ones(
        (A.shape[0], B_t.shape[-1]), device=device, dtype=torch.bfloat16
    )

    # fwd_bwd bf16 benchmark + profiling
    bf16_fwd_bwd_us = bench_fwd_bwd_microseconds(
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

    # fwd_bwd scaled benchmark + profiling
    scaled_fwd_bwd_us = bench_fwd_bwd_microseconds(
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
        bf16_fwd_bwd_us=round(bf16_fwd_bwd_us, 3),
        scaled_fwd_bwd_us=round(scaled_fwd_bwd_us, 3),
        scaled_fwd_bwd_speedup=round(bf16_fwd_bwd_us / scaled_fwd_bwd_us, 3),
        bf16_fwd_us=round(bf16_fwd_us, 3),
        scaled_fwd_us=round(scaled_fwd_us, 3),
        scaled_fwd_speedup=round(bf16_fwd_us / scaled_fwd_us, 3),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M,N,K,G",
        "recipe",
        "bf16_fwd_bwd_us",
        "scaled_fwd_bwd_us",
        "scaled_fwd_bwd_speedup",
        "bf16_fwd_us",
        "scaled_fwd_us",
        "scaled_fwd_speedup",
    ]
    rows = []
    for experiment in experiments:
        rows.append(
            [
                str(experiment.config.MNKG),
                experiment.config.recipe,
                experiment.result.bf16_fwd_bwd_us,
                experiment.result.scaled_fwd_bwd_us,
                f"{experiment.result.scaled_fwd_bwd_speedup}x",
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
