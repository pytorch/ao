# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
#
# The baseline is a torch.compile-fused SwiGLU (one Triton kernel) followed by
# the standalone MXFP8 quantizers:
#   python benchmarks/prototype/moe_training/mxfp8/bench_cutedsl_gated_act_mxfp8.py [--compile]

import argparse
import itertools
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_gated_act_mxfp8 import (
    gated_act_mxfp8_cutedsl_backward,
    gated_act_mxfp8_cutedsl_forward,
)

device = torch.device("cuda")
VALIDATE = os.environ.get("MXFP8_BENCH_VALIDATE", "0") == "1"

SCALING_MODE = "rceil"

# Backward E4M3 only; see eager_reference. Keep in sync with
# _MAX_DIFFERING_FRACTION in test/prototype/moe_training/
# test_cutedsl_gated_act_mxfp8.py.
MAX_DIFFERING_FRACTION = 1e-5


def _swiglu_fwd(gate, up):
    return (F.silu(gate.float()) * up.float()).bfloat16()


def _swiglu_bwd(grad_h, gate, up):
    # Mirror the kernel's evaluation order (silu path); see eager_reference.
    gate, up, grad_h = gate.float(), up.float(), grad_h.float()
    sig = torch.sigmoid(gate)
    act = gate * sig
    dact = act * (1.0 - sig) + sig
    return torch.cat(
        [
            ((dact * grad_h) * up).bfloat16(),
            (act * grad_h).bfloat16(),
        ],
        dim=1,
    )


# One fused Triton kernel each, so the timing baseline's activation never
# round-trips intermediates through DRAM. dynamic=False keeps every shape on
# a static specialization (the auto-dynamic recompile is ~3x slower at large
# shapes); run_experiment resets dynamo between configs so specializations
# never accumulate toward the recompile limit.
_swiglu_fwd_c = torch.compile(_swiglu_fwd, fullgraph=True, dynamic=False)
_swiglu_bwd_c = torch.compile(_swiglu_bwd, fullgraph=True, dynamic=False)


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    direction: str
    scales: str


@dataclass(frozen=True)
class ExperimentResult:
    # time
    baseline_us: float
    fused_us: float
    # mem bw
    baseline_gbps: float
    fused_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    # (M, K): token counts x gate/up widths. 2048 = DSv3 expert FFN
    # intermediate; 7168/8192 are DSv3/Llama3-70B model dims used as
    # representative large widths. (128, 128) is the minimum legal size
    # (launch-bound); (131072, 8192) backward lands just under the kernel's
    # INT32 addressing bound (2*K*M = 2^31 exactly).
    input_shapes = [
        (128, 128),
        (4096, 2048),
        (4096, 7168),
        (16384, 7168),
        (131072, 8192),
    ]
    if args.shape is not None:
        input_shapes = [tuple(args.shape)]
    directions = (
        ["forward", "backward"] if args.direction == "both" else [args.direction]
    )
    scales_modes = (
        ["rowwise", "colwise", "both"] if args.scales == "all" else [args.scales]
    )
    configs = []
    for shape, direction, scales in itertools.product(
        input_shapes, directions, scales_modes
    ):
        configs.append(
            ExperimentConfig(
                input_shape=shape,
                direction=direction,
                scales=scales,
            )
        )
    return configs


def _quantize_reference(reference, rowwise, colwise):
    empty_qdata = reference.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_scales = reference.new_empty(0, dtype=torch.float8_e8m0fnu)
    row = (
        mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode=SCALING_MODE)
        if rowwise
        else (empty_qdata, empty_scales)
    )
    col = (
        mxfp8_quantize_2d_32x1_cutedsl(reference, scaling_mode=SCALING_MODE)
        if colwise
        else (empty_qdata, empty_scales)
    )
    return row[0], col[0], row[1], col[1]


def baseline(gated_input, grad_h, rowwise, colwise):
    # torch.compile-fused SwiGLU, then the standalone MXFP8 quantizers: the
    # activation is already fused, so the measured win is removing the
    # bfloat16 round trip between it and the cast.
    k = gated_input.shape[1] // 2
    gate, up = gated_input[:, :k], gated_input[:, k:]
    if grad_h is None:
        reference = _swiglu_fwd_c(gate, up)
    else:
        reference = _swiglu_bwd_c(grad_h, gate, up)
    return _quantize_reference(reference, rowwise, colwise)


def eager_reference(gated_input, grad_h, rowwise, colwise):
    # Ground truth for validation (not the timing baseline): the kernel's fast
    # sigmoid and d_silu FMA contraction have no bit-exact eager equivalent,
    # so exact agreement is only achievable in the forward direction. Keep in
    # sync with _eager_reference in test/prototype/moe_training/
    # test_cutedsl_gated_act_mxfp8.py: both mirror the kernel's evaluation
    # order.
    k = gated_input.shape[1] // 2
    gate, up = gated_input[:, :k], gated_input[:, k:]
    if grad_h is None:
        reference = _swiglu_fwd(gate, up)
    else:
        reference = _swiglu_bwd(grad_h, gate, up)
    return _quantize_reference(reference, rowwise, colwise)


def fused(gated_input, grad_h, rowwise, colwise):
    if grad_h is None:
        return gated_act_mxfp8_cutedsl_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        )
    return gated_act_mxfp8_cutedsl_backward(
        grad_h, gated_input, rowwise=rowwise, colwise=colwise
    )


def _e4m3_ordinal(u):
    # Map sign-magnitude E4M3 bytes onto a signed number line so adjacent
    # codes differ by 1 across the +/-0 boundary (raw byte distance jumps to
    # 128 there).
    s = u.to(torch.int16)
    return torch.where(s >= 0x80, 0x80 - s, s)


def check(actual, expected, msg, exact):
    # Scales and forward data are bitwise exact; backward data within one code.
    assert actual.shape == expected.shape, f"{msg}: {actual.shape} vs {expected.shape}"
    assert actual.stride() == expected.stride(), f"{msg}: stride mismatch"
    a, e = actual.view(torch.uint8), expected.view(torch.uint8)
    if exact or actual.dtype == torch.float8_e8m0fnu:
        assert bool((a == e).all()), f"{msg}: not bitwise identical"
        return
    # A disabled direction is zero-sized: torch.max() has no empty-reduction identity.
    if actual.numel() == 0:
        return
    gap = (_e4m3_ordinal(a) - _e4m3_ordinal(e)).abs()
    assert int(gap.max()) <= 1, f"{msg}: max E4M3 code gap > 1"
    count = int((gap != 0).sum())
    # Count floor mirrors the test suite's: at small shapes the fractional
    # bound alone allows less than one differing code.
    limit = max(8, int(MAX_DIFFERING_FRACTION * a.numel()))
    assert count <= limit, f"{msg}: {count} codes differ, limit {limit}"


def validate_outputs(actual, gated_input, grad_h, rowwise, colwise):
    M, two_k = gated_input.shape
    direction = "forward" if grad_h is None else "backward"
    expected = eager_reference(gated_input, grad_h, rowwise, colwise)
    for i, (a, e) in enumerate(zip(actual, expected)):
        check(
            a, e, f"M={M} K={two_k // 2} {direction} output {i}", exact=grad_h is None
        )


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    M, K = config.input_shape
    is_backward = config.direction == "backward"
    rowwise = config.scales in ("rowwise", "both")
    colwise = config.scales in ("colwise", "both")

    gated_input = torch.randn(M, 2 * K, dtype=torch.bfloat16, device=device)
    grad_h = (
        torch.randn(M, K, dtype=torch.bfloat16, device=device) if is_backward else None
    )
    bench_args = (gated_input, grad_h, rowwise, colwise)
    if args.compile:
        baseline_fn = torch.compile(baseline, fullgraph=True)
        fused_fn = torch.compile(fused, fullgraph=True)
    else:
        baseline_fn, fused_fn = baseline, fused

    try:
        outputs = fused_fn(*bench_args)
        if VALIDATE:
            validate_outputs(outputs, *bench_args)
        baseline_time_us = benchmark_cuda_function_in_microseconds(
            baseline_fn, *bench_args
        )
        fused_time_us = benchmark_cuda_function_in_microseconds(fused_fn, *bench_args)
    finally:
        torch._dynamo.reset()

    # Memory bandwidth calculations, using the logical traffic of the fused op;
    # the baseline additionally round-trips the bf16 activation through DRAM.
    bytes_per_input_el = torch.finfo(torch.bfloat16).bits / 8
    bytes_per_output_el = torch.finfo(torch.float8_e4m3fn).bits / 8
    bytes_per_scale_el = torch.finfo(torch.float8_e8m0fnu).bits / 8

    read_bytes = gated_input.numel() * bytes_per_input_el
    if grad_h is not None:
        read_bytes += grad_h.numel() * bytes_per_input_el
    output_rowwise, output_colwise, scales_rowwise, scales_colwise = outputs
    write_bytes = (
        output_rowwise.numel() + output_colwise.numel()
    ) * bytes_per_output_el + (
        scales_rowwise.numel() + scales_colwise.numel()
    ) * bytes_per_scale_el

    baseline_gbps = ((read_bytes + write_bytes) / 1e9) / (baseline_time_us / 1e6)
    fused_gbps = ((read_bytes + write_bytes) / 1e9) / (fused_time_us / 1e6)

    return ExperimentResult(
        baseline_us=baseline_time_us,
        fused_us=fused_time_us,
        baseline_gbps=baseline_gbps,
        fused_gbps=fused_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "direction",
        "scales",
        "baseline_us",
        "fused_us",
        "speedup",
        "baseline_gbps",
        "fused_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.baseline_us / experiment.result.fused_us
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.direction,
                experiment.config.scales,
                f"{experiment.result.baseline_us:.2f}",
                f"{experiment.result.fused_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.baseline_gbps:.1f}",
                f"{experiment.result.fused_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)
    configs = get_configs(args)
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))
        torch.cuda.empty_cache()

    # Use Tabulate to print results
    print(f"\nmode: {'compile' if args.compile else 'eager'}")
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action="store_true",
        help="benchmark torch.compile(fullgraph=True) instead of eager",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=None,
        metavar=("M", "K"),
        help="run a single (M, K) shape instead of the sweep",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )
    parser.add_argument(
        "--scales",
        choices=("rowwise", "colwise", "both", "all"),
        default="all",
        help="'both' is the single both-scales mode; 'all' sweeps all three",
    )
    args = parser.parse_args()
    main(args)
