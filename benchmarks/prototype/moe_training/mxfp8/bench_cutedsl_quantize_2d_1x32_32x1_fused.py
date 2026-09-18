# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

"""One fused MXFP8 cast of grad_out vs the three kernels it replaces.

    python benchmarks/prototype/moe_training/mxfp8/bench_cutedsl_quantize_2d_1x32_32x1_fused.py

    MXFP8_BENCH_VALIDATE=1     also check every output bytewise against the baseline
    MXFP8_BENCH_EXTRA_NON_NICE=1   add small and awkward shapes

Baseline is the composition a grouped-GEMM backward runs today: the 1x32 CuTeDSL
cast for dgrad, the 32x1 CUDA cast for wgrad, and a standalone Triton K-groups
swizzle of the colwise scales. The fused kernel reads the input once and emits
all four outputs, moving 4.0625 bytes per input element against the
composition's 6.125.

A NOTE ON WHAT THIS MEASURES. Both sides are timed under CUDA-GRAPH REPLAY, not
eagerly, which is the difference between measuring the kernels and measuring
torch's dispatcher.

torchao's triton_op dispatch costs on the order of 100us per call independent of
problem size, and the baseline pays it THREE times against the fused kernel's
one. Timed eagerly that overhead swamps every shape below roughly 0.5 GB and
reports speedups of 15x on small shapes -- an artifact of launch overhead, not a
kernel win. Graph replay removes dispatch from both sides and leaves the
kernels, which is also what a real training loop sees, since those capture
graphs.

The sibling benchmarks in this directory use eager do_bench. This one
deliberately does not: a 3-kernel baseline against a 1-kernel candidate is
exactly the case where dispatch overhead dominates the number being reported.
"""

import itertools
import os
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench_cudagraph

from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_quantize_2d_1x32_32x1_fused import (
    quantize_fused,
    s_col_shape,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_2d_1x32_cutedsl,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import (
    KernelPreference,
    MXFP8Dim1CastKernelChoice,
)
from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper

device = torch.device("cuda")
VALIDATE = os.environ.get("MXFP8_BENCH_VALIDATE", "0") == "1"
EXTRA_NON_NICE = os.environ.get("MXFP8_BENCH_EXTRA_NON_NICE", "0") == "1"

BLOCK_SIZE = 32

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    num_groups: int


@dataclass(frozen=True)
class ExperimentResult:
    # time
    fused_us: float
    unfused_us: float
    # mem bw
    fused_gbps: float
    unfused_gbps: float


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
    if EXTRA_NON_NICE:
        input_shapes += [
            (256, 128),
            (1152, 1408),
        ]
    num_groups_list = [4, 8]
    configs = []
    for shape, num_groups in itertools.product(input_shapes, num_groups_list):
        total_M, N = shape
        # The kernel requires both dims to be multiples of 128, and every group
        # size to be a multiple of 128 as well.
        if total_M % 128 or N % 128 or num_groups > total_M // 128:
            continue
        configs.append(ExperimentConfig(input_shape=shape, num_groups=num_groups))
    return configs


def benchmark_cuda_graph_in_microseconds(f, *args, **kwargs) -> float:
    """Median device time of one call, under CUDA-graph replay."""
    return do_bench_cudagraph(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def assert_graph_capture_is_real(x: torch.Tensor, offs: torch.Tensor):
    """Fail loudly if the fused kernel captures as an EMPTY graph.

    A CuTeDSL kernel that launches on the default stream instead of the caller's
    captures nothing: no error, and the replay then times at about 1us, which
    reads as a 100x win. Every campaign winner for this kernel shipped with that
    bug before it was patched, so check rather than assume. Replay into zeroed
    buffers and compare against the eager result.
    """
    total_M, N = x.shape
    num_groups = offs.numel()
    s_col_rows, s_col_cols = s_col_shape(total_M, N, num_groups)
    bufs = (
        torch.empty((total_M, N), device=x.device, dtype=torch.float8_e4m3fn),
        torch.empty((total_M, N // BLOCK_SIZE), device=x.device, dtype=torch.uint8),
        torch.empty((N, total_M), device=x.device, dtype=torch.float8_e4m3fn),
        torch.empty((s_col_rows, s_col_cols), device=x.device, dtype=torch.uint8),
    )

    def run():
        quantize_fused(
            x,
            offs,
            block_size=BLOCK_SIZE,
            scaling_mode="rceil",
            q_row_out=bufs[0],
            s_row_out=bufs[1],
            q_col_out=bufs[2],
            s_col_out=bufs[3],
        )

    run()
    eager = [b.view(torch.uint8).clone() for b in bufs]

    for b in bufs:
        b.view(torch.uint8).zero_()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for b in bufs:
        b.view(torch.uint8).zero_()
    g.replay()
    torch.cuda.synchronize()

    for name, got, exp in zip(("q_row", "s_row", "q_col", "s_col"), bufs, eager):
        if not torch.equal(got.view(torch.uint8), exp):
            raise RuntimeError(
                f"CUDA-graph replay != eager on {name}. The kernel almost "
                "certainly ignores the caller's stream, so it captured as an "
                "empty graph and any timing below would be meaningless."
            )


def unfused_baseline(x: torch.Tensor, group_offs: torch.Tensor):
    """The three kernels a grouped-GEMM backward dispatches for grad_out today."""
    q_row, s_row = mxfp8_quantize_2d_1x32_cutedsl(
        x, block_size=BLOCK_SIZE, scaling_mode="rceil", offs=group_offs
    )
    mx = _to_mxfp8_dim1_kernel_wrapper(
        x,
        BLOCK_SIZE,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=x.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )
    s_col = triton_mx_block_rearrange_2d_K_groups(mx.scale, group_offs // BLOCK_SIZE)
    return q_row, s_row, mx.qdata, s_col


def validate_outputs(fused, unfused, num_groups: int):
    names = ("q_row", "s_row", "q_col", "s_col")
    for name, got, exp in zip(names, fused, unfused):
        got = got.view(torch.uint8)
        exp = exp.view(torch.uint8)
        if name == "s_col":
            # Trailing per-group padding slack, never read by the grouped GEMM.
            used = exp.shape[1] - num_groups * 4
            got, exp = got[:, :used], exp[:, :used]
        torch.testing.assert_close(got, exp, rtol=0, atol=0, msg=f"{name} mismatch")


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    total_M, N = config.input_shape
    num_groups = config.num_groups

    x = torch.randn(total_M, N, dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(num_groups, total_M, multiple_of=128, device=device).to(
        torch.int32
    )

    fused_out = quantize_fused(x, offs, block_size=BLOCK_SIZE, scaling_mode="rceil")
    if VALIDATE:
        validate_outputs(fused_out, unfused_baseline(x, offs), num_groups)

    assert_graph_capture_is_real(x, offs)

    fused_time_us = benchmark_cuda_graph_in_microseconds(
        quantize_fused, x, offs, block_size=BLOCK_SIZE, scaling_mode="rceil"
    )
    unfused_time_us = benchmark_cuda_graph_in_microseconds(unfused_baseline, x, offs)

    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    q_row, s_row, q_col, s_col = fused_out
    write_bytes = (
        q_row.numel() * bytes_per_output_el
        + q_col.numel() * bytes_per_output_el
        + s_row.numel() * bytes_per_scale_el
        + s_col.numel() * bytes_per_scale_el
    )
    # The two sides move DIFFERENT byte volumes: the baseline reads x twice and
    # rewrites the colwise scales, so each is charged its own traffic. That makes
    # the GB/s columns comparable against a roofline but NOT against each other;
    # the speedup column carries the win.
    fused_read_bytes = x.numel() * bytes_per_input_el
    unfused_read_bytes = 2 * fused_read_bytes + s_col.numel() * bytes_per_scale_el
    unfused_write_bytes = write_bytes + s_col.numel() * bytes_per_scale_el

    fused_gbps = ((fused_read_bytes + write_bytes) / 1e9) / (fused_time_us / 1e6)
    unfused_gbps = ((unfused_read_bytes + unfused_write_bytes) / 1e9) / (
        unfused_time_us / 1e6
    )

    return ExperimentResult(
        fused_us=fused_time_us,
        unfused_us=unfused_time_us,
        fused_gbps=fused_gbps,
        unfused_gbps=unfused_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "num_groups",
        "fused_us",
        "unfused_us",
        "speedup",
        "fused_gbps",
        "unfused_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.unfused_us / experiment.result.fused_us
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.num_groups,
                f"{experiment.result.fused_us:.2f}",
                f"{experiment.result.unfused_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.fused_gbps:.1f}",
                f"{experiment.result.unfused_gbps:.1f}",
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

    print_results(results)


if __name__ == "__main__":
    main()
