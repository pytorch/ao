# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    _mx_block_rearrange_2d_k_groups_cutedsl,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    torch_to_blocked_2d_K_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.utils import ceil_div


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
    """Parses `MxTOTALKxGROUPS` or `LABEL:MxTOTALKxGROUPS`."""
    label, _, dims = spec.rpartition(":")
    parts = dims.split("x")
    if len(parts) != 3:
        raise SystemExit(f"Malformed shape {spec!r}, expected [LABEL:]MxTOTALKxGROUPS")
    m, total_k, groups = (int(p) for p in parts)
    return (label or None, m, total_k, groups)


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
        description="Benchmark torch / triton / CuTeDSL MXFP8 K-groups scale rearrange."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        metavar="[LABEL:]MxTOTALKxGROUPS",
        help="shapes to benchmark, e.g. 2048x4096x8 or dsv3:2048x4096x8",
    )
    parser.add_argument("--multiple-of", type=int, default=1)
    parser.add_argument(
        "--chunk-width",
        type=int,
        default=None,
        help="CuTeDSL column chunk width; default lets the kernel choose",
    )
    parser.add_argument(
        "--bench-only",
        nargs="+",
        choices=("torch", "triton", "cutedsl"),
        default=None,
        help="restrict to one or more backends; default is all",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    shapes = [_parse_shape(s) for s in args.shapes]
    labelled = any(label is not None for label, _, _, _ in shapes)
    selected = (
        set(args.bench_only) if args.bench_only else {"torch", "triton", "cutedsl"}
    )
    if args.cuda_graph_bench and "torch" in selected:
        # torch_to_blocked_2d_K_groups calls .tolist(), a synchronous D2H copy
        # that CUDA graph capture never tolerates, for any shape.
        if selected == {"torch"}:
            raise SystemExit("torch cannot be benchmarked under --cuda-graph-bench")
        print(
            "warning: excluding torch from --cuda-graph-bench (unpinned D2H copy)",
            file=sys.stderr,
        )
        selected.discard("torch")
    block_size = 32
    rows_out = []

    for label, m, total_k, groups in shapes:
        scale_cols = ceil_div(total_k, block_size)
        if args.deterministic:
            raw = (
                torch.arange(m * scale_cols, device="cuda", dtype=torch.int32)
                .remainder(251)
                .to(torch.uint8)
                .reshape(m, scale_cols)
            )
        else:
            raw = torch.randint(
                0, 256, (m, scale_cols), device="cuda", dtype=torch.uint8
            )
        scales = raw.view(torch.float8_e8m0fnu)
        offsets = generate_jagged_offs(
            groups, scale_cols, multiple_of=args.multiple_of, device="cuda"
        )

        row = {"Model layer": label or ""} if labelled else {}
        row["M x TotalK"] = f"{m}x{total_k}"
        row["Groups"] = groups

        if "torch" in selected:
            torch_us = benchmark_function(
                torch_to_blocked_2d_K_groups,
                scales,
                offsets,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["Torch us"] = f"{torch_us:.2f}"
        if "triton" in selected:
            triton_us = benchmark_function(
                triton_mx_block_rearrange_2d_K_groups,
                scales,
                offsets,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["Triton us"] = f"{triton_us:.2f}"
        if "cutedsl" in selected:
            out = _mx_block_rearrange_2d_k_groups_cutedsl(
                scales, offsets, args.chunk_width
            )
            cutedsl_us = benchmark_function(
                _mx_block_rearrange_2d_k_groups_cutedsl,
                scales,
                offsets,
                args.chunk_width,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["CuTeDSL us"] = f"{cutedsl_us:.2f}"
            # Rearrange is purely memory bound, so GB/s says how close to roofline
            # the kernel gets in a way raw microseconds cannot.
            moved_bytes = (
                (scales.numel() + out.numel())
                * torch.finfo(torch.float8_e8m0fnu).bits
                / 8
            )
            row["CuTeDSL GB/s"] = f"{(moved_bytes / 1e9) / (cutedsl_us / 1e6):.1f}"
        if "triton" in selected and "cutedsl" in selected:
            row["CuTeDSL speedup"] = f"{triton_us / cutedsl_us:.2f}"
        rows_out.append(row)

    print(_markdown(rows_out, list(rows_out[0].keys())))


if __name__ == "__main__":
    main()
