# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_32x1_cutedsl,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common import KernelPreference


def _quantize_dim1_cuda(x, block_size, scale_mode):
    x_mx = _to_mxfp8_dim1_kernel_wrapper(
        x,
        block_size,
        elem_dtype=torch.float8_e4m3fn,
        hp_dtype=x.dtype,
        kernel_preference=KernelPreference.AUTO,
        cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
        scale_calculation_mode=scale_mode,
    )
    return x_mx.qdata, x_mx.scale


def _quantize_dim1_cutedsl(x, block_size, scale_mode):
    qdata, scales = mxfp8_quantize_2d_32x1_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scale_mode.value,
        blocked_scale_output=False,
    )
    return qdata.t(), scales


def _wgrad(grad_output, input_act, offs, block_size, scale_mode, choice):
    """The dim1 (wgrad) path: cast both operands, block the scales, grouped mm."""
    quantize_dim1 = (
        _quantize_dim1_cutedsl
        if choice is MXFP8Dim1CastKernelChoice.CUTEDSL
        else _quantize_dim1_cuda
    )

    grad_output_t_data, grad_output_t_scales = quantize_dim1(
        grad_output, block_size, scale_mode
    )
    input_act_t_data, input_act_t_scales = quantize_dim1(
        input_act, block_size, scale_mode
    )

    scale_group_offsets = offs // block_size
    grad_output_t_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
        grad_output_t_scales,
        scale_group_offsets,
    )
    input_act_t_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
        input_act_t_scales,
        scale_group_offsets,
    )

    grad_weight = torch._scaled_grouped_mm(
        grad_output_t_data,
        input_act_t_data.transpose(-2, -1),
        grad_output_t_scales_blocked,
        input_act_t_scales_blocked,
        offs=offs,
        out_dtype=torch.bfloat16,
    )
    # Grouped output is 3-D (one weight-gradient per expert), hence the transpose
    # of the last two dims rather than a plain .t().
    return grad_weight.transpose(-2, -1)


def benchmark_cuda_graph_function_in_microseconds(f, *args, iters=1000, **kwargs):
    for _ in range(10):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f(*args, **kwargs)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def benchmark_function(f, *args, use_cuda_graph=False, graph_iters=1000, **kwargs):
    if use_cuda_graph:
        return benchmark_cuda_graph_function_in_microseconds(
            f,
            *args,
            iters=graph_iters,
            **kwargs,
        )
    return benchmark_cuda_function_in_microseconds(f, *args, **kwargs)


def _parse_shape(spec: str):
    """Parses `MxNxKxGROUPS` or `LABEL:MxNxKxGROUPS`."""
    label, _, dims = spec.rpartition(":")
    parts = dims.split("x")
    if len(parts) != 4:
        raise SystemExit(f"Malformed shape {spec!r}, expected [LABEL:]MxNxKxGROUPS")
    m, n, k, groups = (int(p) for p in parts)
    return (label or None, m, n, k, groups)


def _markdown(rows, columns):
    widths = [
        max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c)
        for c in columns
    ]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    sep = "|" + "|".join("---:".rjust(w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(str(r[c]).rjust(w) for c, w in zip(columns, widths)) + " |"
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the MXFP8 dim1 (wgrad) path with the CUDA vs CuTeDSL "
        "dim1 cast kernel."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        metavar="[LABEL:]MxNxKxGROUPS",
        help="shapes to benchmark, e.g. 4096x4096x2048x8 or dsv3:4096x4096x2048x8",
    )
    parser.add_argument("--multiple-of", type=int, default=128)
    parser.add_argument(
        "--scale-mode",
        choices=("floor", "rceil"),
        default="rceil",
    )
    parser.add_argument(
        "--bench-only",
        choices=("cuda", "cutedsl"),
        default=None,
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    shapes = [_parse_shape(s) for s in args.shapes]
    labelled = any(label is not None for label, *_ in shapes)
    block_size = 32
    scale_mode = ScaleCalculationMode(args.scale_mode)
    rows_out = []

    for label, m, n, k, groups in shapes:
        if args.deterministic:
            grad_output = (
                torch.arange(m * n, device="cuda", dtype=torch.float32)
                .remainder(17)
                .sub(8)
                .div(16)
                .reshape(m, n)
                .to(torch.bfloat16)
            )
            input_act = (
                torch.arange(m * k, device="cuda", dtype=torch.float32)
                .remainder(17)
                .sub(8)
                .div(16)
                .reshape(m, k)
                .to(torch.bfloat16)
            )
        else:
            grad_output = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
            input_act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        offsets = generate_jagged_offs(
            groups, m, multiple_of=args.multiple_of, device="cuda"
        )

        row = {"Model layer": label or ""} if labelled else {}
        row["MxNxK"] = f"{m}x{n}x{k}"
        row["Groups"] = groups

        if args.bench_only in (None, "cuda"):
            cuda_us = benchmark_function(
                _wgrad,
                grad_output,
                input_act,
                offsets,
                block_size,
                scale_mode,
                MXFP8Dim1CastKernelChoice.CUDA,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["CUDA us"] = f"{cuda_us:.2f}"
        if args.bench_only in (None, "cutedsl"):
            cutedsl_us = benchmark_function(
                _wgrad,
                grad_output,
                input_act,
                offsets,
                block_size,
                scale_mode,
                MXFP8Dim1CastKernelChoice.CUTEDSL,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["CuTeDSL us"] = f"{cutedsl_us:.2f}"
        if args.bench_only is None:
            row["CuTeDSL speedup"] = f"{cuda_us / cutedsl_us:.2f}"
        rows_out.append(row)

    print(_markdown(rows_out, list(rows_out[0].keys())))


if __name__ == "__main__":
    main()
