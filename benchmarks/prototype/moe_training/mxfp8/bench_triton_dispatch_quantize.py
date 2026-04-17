# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark: fused Triton dispatch+pad+quantize+blocked-scales vs. today's
3-stage pipeline (pad_token_groups -> triton_to_mxfp8_dim0 ->
mx_block_rearrange_2d_M_groups_cuda).

Target landing bar (ao#4184 review): >= 5 TB/s HBM throughput for the fused
kernel at total_M in {4096, 8192, 16384, 32768} on B200.
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    _mxfp8_cuda_kernels_available,
    fused_pad_token_groups_cuda,
    mx_block_rearrange_2d_M_groups_cuda,
    torch_pad_token_groups,
    triton_mx_block_rearrange_2d_M_groups,
    triton_mxfp8_pad_and_quantize,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.utils import ceil_div

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    k: int
    num_groups: int


@dataclass(frozen=True)
class ExperimentResult:
    stage3_align32_us: float  # today's path
    stage3_align128_us: float  # apples-to-apples (same output layout)
    fused_us: float
    stage3_align128_gbps: float
    fused_gbps: float
    speedup_vs_align128: float
    speedup_vs_align32: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Daniel's "realistic batch sizes" range for the MoE forward (ao#4184).
    num_tokens_list = [4096, 8192, 16384, 32768]
    k_list = [2048, 5120, 7168]
    num_groups_list = [4, 8, 16]
    return [
        ExperimentConfig(num_tokens=t, k=k, num_groups=g)
        for t, k, g in itertools.product(num_tokens_list, k_list, num_groups_list)
    ]


def _three_stage_reference(
    x: torch.Tensor,
    group_offsets: torch.Tensor,
    alignment: int,
):
    """Run the same three kernels used by _compute_fwd_sm100 today and return
    the final padded blocked-scales tensor + fp8 data.

    Matches the production path exactly: no D2H syncs, upper-bound allocation
    carried through quantize + rearrange (those kernels don't care about slack
    rows)."""
    if alignment == 32 and _mxfp8_cuda_kernels_available:
        padded, _start, padded_end = fused_pad_token_groups_cuda(
            x, group_offsets, alignment_size=alignment
        )
    else:
        padded, _start, padded_end = torch_pad_token_groups(
            x, group_offsets, alignment_size=alignment
        )
    qdata, scales = triton_to_mxfp8_dim0(
        padded, inner_block_size=32, scaling_mode="rceil"
    )
    if _mxfp8_cuda_kernels_available:
        blocked = mx_block_rearrange_2d_M_groups_cuda(
            scales, padded_end.to(torch.int32)
        )
    else:
        blocked = triton_mx_block_rearrange_2d_M_groups(
            scales, padded_end.to(torch.int32)
        )
    return qdata, blocked, padded_end


def _bytes_touched_3stage(num_tokens, k, num_groups, alignment):
    # Upper-bound accounting: matches the allocations the pipeline actually
    # makes (incl. the padded-bf16 scratch buffer).
    padded_M = num_tokens + num_groups * alignment
    padded_cols = ceil_div(k // 32, 4) * 4
    blocked_rows = padded_M + num_groups * 128
    # pad: read bf16, write bf16
    pad = num_tokens * k * 2 + padded_M * k * 2
    # quant: read bf16, write fp8, write scales
    quant = padded_M * k * 2 + padded_M * k * 1 + padded_M * (k // 32) * 1
    # rearrange: read scales, write blocked
    rearrange = padded_M * (k // 32) * 1 + blocked_rows * padded_cols * 1
    return pad + quant + rearrange


def _bytes_touched_fused(num_tokens, k, num_groups):
    padded_M = num_tokens + num_groups * 128
    padded_cols = ceil_div(k // 32, 4) * 4
    # read bf16 + indices, write fp8 + blocked scales.
    return (
        num_tokens * k * 2
        + padded_M * 4  # int32 src_indices
        + padded_M * k * 1
        + padded_M * padded_cols * 1
    )


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    num_tokens, k, num_groups = config.num_tokens, config.k, config.num_groups

    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=device)
    group_offsets = generate_jagged_offs(
        num_groups, num_tokens, multiple_of=1, device=device
    ).to(torch.int32)

    def run_3stage_align32():
        return _three_stage_reference(x, group_offsets, alignment=32)

    def run_3stage_align128():
        return _three_stage_reference(x, group_offsets, alignment=128)

    def run_fused():
        return triton_mxfp8_pad_and_quantize(x, group_offsets, scaling_mode="rceil")

    def warmup(fn):
        for _ in range(5):
            fn()

    warmup(run_3stage_align32)
    warmup(run_3stage_align128)
    warmup(run_fused)

    stage3_align32_us = benchmark_cuda_function_in_microseconds(run_3stage_align32)
    stage3_align128_us = benchmark_cuda_function_in_microseconds(run_3stage_align128)
    fused_us = benchmark_cuda_function_in_microseconds(run_fused)

    bytes_3stage_128 = _bytes_touched_3stage(num_tokens, k, num_groups, 128)
    bytes_fused = _bytes_touched_fused(num_tokens, k, num_groups)
    stage3_align128_gbps = (bytes_3stage_128 / 1e9) / (stage3_align128_us / 1e6)
    fused_gbps = (bytes_fused / 1e9) / (fused_us / 1e6)

    return ExperimentResult(
        stage3_align32_us=stage3_align32_us,
        stage3_align128_us=stage3_align128_us,
        fused_us=fused_us,
        stage3_align128_gbps=stage3_align128_gbps,
        fused_gbps=fused_gbps,
        speedup_vs_align128=stage3_align128_us / fused_us,
        speedup_vs_align32=stage3_align32_us / fused_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "num_tokens",
        "k",
        "groups",
        "3stage_a32_us",
        "3stage_a128_us",
        "fused_us",
        "3stage_a128_GB/s",
        "fused_GB/s",
        "speedup_vs_a128",
        "speedup_vs_a32",
    ]
    rows = []
    for exp in experiments:
        rows.append(
            [
                exp.config.num_tokens,
                exp.config.k,
                exp.config.num_groups,
                f"{exp.result.stage3_align32_us:.1f}",
                f"{exp.result.stage3_align128_us:.1f}",
                f"{exp.result.fused_us:.1f}",
                f"{exp.result.stage3_align128_gbps:.0f}",
                f"{exp.result.fused_gbps:.0f}",
                f"{exp.result.speedup_vs_align128:.2f}x",
                f"{exp.result.speedup_vs_align32:.2f}x",
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
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling with PyTorch profiler"
    )
    args = parser.parse_args()
    main(args)
