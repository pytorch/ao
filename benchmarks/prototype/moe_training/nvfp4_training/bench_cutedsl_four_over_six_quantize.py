# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION.
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

import torchao.prototype.moe_training.nvfp4_training.four_over_six as four_over_six_module
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    four_over_six_quantize,
)

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    block: str
    row_scaled: bool


@dataclass(frozen=True)
class ExperimentResult:
    # time
    cutedsl_us: float
    torch_ref_us: float
    # mem bw
    cutedsl_gbps: float
    torch_ref_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    input_shapes = [
        # DeepSeekV3 671b shapes
        (8192, 2048),
        (8192, 7168),
        (32768, 2048),
        (32768, 7168),
        (131072, 2048),
        (131072, 7168),
    ]
    cases = [
        ("1x16", False),  # activations, per-tensor amax
        ("1x16", True),  # activations, row-scaled amax
        ("16x16", False),  # weights
    ]
    configs = []
    for shape, (block, row_scaled) in itertools.product(input_shapes, cases):
        configs.append(
            ExperimentConfig(input_shape=shape, block=block, row_scaled=row_scaled)
        )
    return configs


def _reference_quantize(x, global_amax, **kwargs):
    """Pure-PyTorch four_over_six_quantize body (dispatch gate disabled)."""
    orig = four_over_six_module._cutedsl_quantize_eligible
    four_over_six_module._cutedsl_quantize_eligible = lambda t: False
    try:
        return four_over_six_quantize(x, global_amax, **kwargs)
    finally:
        four_over_six_module._cutedsl_quantize_eligible = orig


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    x = torch.randn(*config.input_shape, dtype=torch.bfloat16, device=device)
    if config.row_scaled:
        amax = x.abs().amax(dim=1).to(torch.float32)
    else:
        amax = x.abs().amax().to(torch.float32)

    quantize_kwargs = dict(block=config.block, err_mode="mae", e4m3_scale_bound=256)

    # Correctness first: the CuTe DSL fast path must be bitwise identical.
    assert four_over_six_module._cutedsl_quantize_eligible(x)
    codes, scales = four_over_six_quantize(x, amax, **quantize_kwargs)
    ref_codes, ref_scales = _reference_quantize(x, amax, **quantize_kwargs)
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))

    cutedsl_us = benchmark_cuda_function_in_microseconds(
        four_over_six_quantize, x, amax, **quantize_kwargs
    )
    torch_ref_us = benchmark_cuda_function_in_microseconds(
        _reference_quantize, x, amax, **quantize_kwargs
    )

    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    read_bytes = x.numel() * bytes_per_input_el
    write_bytes = codes.numel() * 1 + scales.numel() * 1
    cutedsl_gbps = ((read_bytes + write_bytes) / 1e9) / (cutedsl_us / 1e6)
    torch_ref_gbps = ((read_bytes + write_bytes) / 1e9) / (torch_ref_us / 1e6)

    return ExperimentResult(
        cutedsl_us=cutedsl_us,
        torch_ref_us=torch_ref_us,
        cutedsl_gbps=cutedsl_gbps,
        torch_ref_gbps=torch_ref_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "block",
        "row_scaled",
        "cutedsl_us",
        "torch_ref_us",
        "speedup",
        "cutedsl_gbps",
        "torch_ref_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.torch_ref_us / experiment.result.cutedsl_us
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.block,
                experiment.config.row_scaled,
                f"{experiment.result.cutedsl_us:.2f}",
                f"{experiment.result.torch_ref_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.cutedsl_gbps:.1f}",
                f"{experiment.result.torch_ref_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))


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
