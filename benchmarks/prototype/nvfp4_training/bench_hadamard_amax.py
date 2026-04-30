# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax

device = torch.device("cuda")

M_SHAPES = [128, 256, 1024, 8192]
N_SHAPES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

DEEPSEEK_V3_671B_SEQ_LEN = 4096
DEEPSEEK_V3_671B_MICROBATCH_SIZE = 1
DEEPSEEK_V3_671B_NUM_ROUTED_EXPERTS = 256
DEEPSEEK_V3_671B_NUM_EXPERTS_PER_TOK = 8
LLAMA_BATCH_SIZE = 1
LLAMA_SEQ_LEN = 2048


@dataclass(frozen=True)
class ExperimentConfig:
    m: int
    n: int
    model: str = ""
    shape: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float
    gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(m=m, n=n) for m, n in itertools.product(M_SHAPES, N_SHAPES)
    ]


def get_representative_model_configs() -> List[ExperimentConfig]:
    deepseek_m = DEEPSEEK_V3_671B_MICROBATCH_SIZE * DEEPSEEK_V3_671B_SEQ_LEN
    deepseek_routed_expert_avg_m = (
        deepseek_m
        * DEEPSEEK_V3_671B_NUM_EXPERTS_PER_TOK
        // DEEPSEEK_V3_671B_NUM_ROUTED_EXPERTS
    )
    llama_m = LLAMA_BATCH_SIZE * LLAMA_SEQ_LEN

    return [
        # DeepSeek-V3 671B activation-input shapes for the linear layers.
        ExperimentConfig(deepseek_m, 512, "DeepSeek-V3 671B", "attn.wkv_b input"),
        ExperimentConfig(deepseek_m, 1536, "DeepSeek-V3 671B", "attn.wq_b input"),
        ExperimentConfig(
            deepseek_m, 2048, "DeepSeek-V3 671B", "shared expert w2 input"
        ),
        ExperimentConfig(deepseek_m, 7168, "DeepSeek-V3 671B", "hidden-state input"),
        ExperimentConfig(deepseek_m, 16384, "DeepSeek-V3 671B", "attn.wo input"),
        ExperimentConfig(deepseek_m, 18432, "DeepSeek-V3 671B", "dense ffn.w2 input"),
        ExperimentConfig(
            deepseek_routed_expert_avg_m,
            2048,
            "DeepSeek-V3 671B",
            "avg routed expert w2 input",
        ),
        ExperimentConfig(
            deepseek_routed_expert_avg_m,
            7168,
            "DeepSeek-V3 671B",
            "avg routed expert w1/w3 input",
        ),
        # Llama 3 8B and 70B activation-input shapes.
        ExperimentConfig(llama_m, 4096, "Llama 3 8B", "hidden-state input"),
        ExperimentConfig(llama_m, 14336, "Llama 3 8B", "mlp.down input"),
        ExperimentConfig(llama_m, 8192, "Llama 3 70B", "hidden-state input"),
        ExperimentConfig(llama_m, 28672, "Llama 3 70B", "mlp.down input"),
    ]


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    x = torch.randn(config.m, config.n, dtype=torch.bfloat16, device=device)

    time_us = benchmark_cuda_function_in_microseconds(triton_rht_amax, x)

    read_bytes = x.numel() * (torch.finfo(torch.bfloat16).bits // 8)
    gbps = (read_bytes / 1e9) / (time_us / 1e6)

    return ExperimentResult(time_us=time_us, gbps=gbps)


def print_results(experiments: List[Experiment]):
    has_labels = any(e.config.model or e.config.shape for e in experiments)
    headers = ["M", "N", "time_us", "gbps"]
    rows = []
    for e in experiments:
        row = [
            e.config.m,
            e.config.n,
            round(e.result.time_us, 3),
            round(e.result.gbps, 3),
        ]
        if has_labels:
            row = [e.config.model, e.config.shape] + row
        rows.append(row)
    if has_labels:
        headers = ["model", "shape"] + headers
    print(tabulate(rows, headers=headers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-set",
        choices=("sweep", "representative-models"),
        default="sweep",
        help="Benchmark the original sweep or selected model-derived shapes.",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    configs = (
        get_representative_model_configs()
        if args.shape_set == "representative-models"
        else get_configs()
    )
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    main()
