# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
import argparse
import csv
import itertools
import logging
import os
from dataclasses import dataclass
from typing import List, Union

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import (
    bench_fwd_bwd_microseconds,
    bench_fwd_microseconds,
    profile_fwd_bwd,
)
from torchao.prototype.moe_training.config import (
    Float8TrainingOpConfig,
    Float8TrainingRecipe,
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)
from torchao.prototype.moe_training.utils import (
    _quantize_then_scaled_grouped_mm,
    generate_jagged_offs,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_MI300, is_MI350, is_ROCM

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

# Dynamic shapes hurt performance
torch._dynamo.config.automatic_dynamic_shapes = False


@dataclass(frozen=True)
class ExperimentConfig:
    high_precision_dtype: torch.dtype
    MNKG: tuple[int]
    recipe: Union[Float8TrainingRecipe, MXFP8TrainingRecipe, KernelPreference]


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
        # Llama4 16e with various experts per device (i.e., EP degree=8 or 16
        (32768, 8192, 5120, 1),
        (32768, 8192, 5120, 2),
        (128000, 8192, 5120, 1),
        (128000, 8192, 5120, 2),
        # DSV3 671B with various experts per device (i.e., EP degree=32 or 64
        (32768, 2048, 7168, 4),
        (32768, 2048, 7168, 8),
        (128000, 2048, 7168, 4),
        (128000, 2048, 7168, 8),
    ]
    recipes = [
        MXFP8TrainingRecipe.MXFP8_RCEIL,
        MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP,
        KernelPreference.TE,
    ]
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

    offs = generate_jagged_offs(G, total_M, multiple_of=1)

    # fwd_bwd bf16 benchmark + profiling
    bf16_fwd_bwd_us = bench_fwd_bwd_microseconds(
        torch._grouped_mm,
        A,
        B_t,
        offs,
        use_compile=args.compile,
        fullgraph=False,
    )
    if args.profile:
        profile_fwd_bwd(
            torch._grouped_mm,
            A,
            B_t,
            offs,
            use_compile=args.compile,
            fullgraph=False,
            profile_name="bf16_profile",
        )

    # Create config object from recipe
    if isinstance(config.recipe, Float8TrainingRecipe):
        quant_config = Float8TrainingOpConfig.from_recipe(config.recipe)
    elif config.recipe == KernelPreference.TE:
        quant_config = MXFP8TrainingOpConfig(kernel_preference=KernelPreference.TE)
    else:
        quant_config = MXFP8TrainingOpConfig.from_recipe(config.recipe)

    # fwd_bwd scaled benchmark + profiling
    scaled_fwd_bwd_us = bench_fwd_bwd_microseconds(
        _quantize_then_scaled_grouped_mm,
        A,
        B_t,
        quant_config,
        offs,
        use_compile=args.compile,
        fullgraph=False,
    )
    if args.profile:
        profile_fwd_bwd(
            _quantize_then_scaled_grouped_mm,
            A,
            B_t,
            quant_config,
            offs,
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
        _quantize_then_scaled_grouped_mm,
        A,
        B_t,
        quant_config,
        offs,
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


def print_results(experiments: List[Experiment], csv_path: str = None):
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
                experiment.result.scaled_fwd_bwd_speedup,
                experiment.result.bf16_fwd_us,
                experiment.result.scaled_fwd_us,
                experiment.result.scaled_fwd_speedup,
            ]
        )

    display_rows = [
        row[:4] + [f"{row[4]}x"] + row[5:7] + [f"{row[7]}x"] for row in rows
    ]
    print(tabulate(display_rows, headers=headers))

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nCSV results saved to {csv_path}")


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        if config.recipe == Float8TrainingRecipe.FP8_ROWWISE:
            if is_ROCM():
                if not (is_MI300() or is_MI350()):
                    logging.warning(
                        "Skipping FP8 rowwise benchmarks, requires MI300 or MI350 on ROCm"
                    )
                    continue
            else:
                if torch.cuda.get_device_capability() != (9, 0):
                    logging.warning(
                        f"Skipping FP8 rowwise benchmarks, only supported on compute capability 9.0 and found {torch.cuda.get_device_capability()}"
                    )
                    continue

        elif config.recipe in (
            MXFP8TrainingRecipe.MXFP8_RCEIL,
            MXFP8TrainingRecipe.MXFP8_RCEIL_WGRAD_WITH_HP,
        ) and torch.cuda.get_device_capability() != (10, 0):
            logging.warning(
                f"Skipping MXFP8 benchmarks, only supported on compute capability 10.0 and found {torch.cuda.get_device_capability()}"
            )
            continue

        elif config.recipe == KernelPreference.TE:
            if torch.cuda.get_device_capability()[0] < 9:
                logging.warning(
                    f"Skipping TE MXFP8 benchmarks, requires SM90+ and found {torch.cuda.get_device_capability()}"
                )
                continue
            try:
                import transformer_engine  # noqa: F401
            except ImportError:
                logging.warning(
                    "Skipping TE MXFP8 benchmarks, TransformerEngine not installed"
                )
                continue

        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results, csv_path=args.csv)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--compile", action="store_true")
    arg_parser.add_argument("--profile", action="store_true")
    arg_parser.add_argument("--csv", type=str, default=None, help="Path to save CSV results")
    args = arg_parser.parse_args()
    main(args)
