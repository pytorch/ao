# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8
from torchao.quantization.quantize_.workflows.float8.kernels import (
    _to_sparse_semi_structured_cutedsl,
)
from torchao.utils import is_sm_at_least_90


def create_semi_structured_tensor(r, c, dtype):
    """Returns a 1:2 sparse matrix of size (r, c), which is also 2:4 sparse."""
    choice_indices = torch.randint(0, 2, (r * c // 2,)).cuda()
    mask = (
        torch.nn.functional.one_hot(choice_indices, num_classes=2)
        .reshape(r, c)
        .contiguous()
        .to(torch.int32)
    )
    sparse_weight = mask + (torch.rand(r, c).cuda() * mask)
    return sparse_weight.to(dtype)


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


def _dtype(name: str) -> torch.dtype:
    if name == "e4m3":
        return torch.float8_e4m3fn
    if name == "e5m2":
        return torch.float8_e5m2
    raise AssertionError(f"Unsupported dtype: {name}")


def _make_deterministic_sparse(rows: int, cols: int, dtype: torch.dtype):
    values = (
        torch.arange(rows * cols, device="cuda", dtype=torch.int32)
        .remainder(126)
        .to(torch.uint8)
        + 1
    ).reshape(rows, cols)
    mask = torch.arange(cols, device="cuda").remainder(4)
    keep = (mask == 0) | (mask == 3)
    raw = torch.where(keep.reshape(1, cols), values, torch.zeros_like(values))
    return raw.contiguous().view(dtype)


def _parse_shape(spec: str):
    """Parses `ROWSxCOLS` or `LABEL:ROWSxCOLS` into (label, rows, cols)."""
    label, _, dims = spec.rpartition(":")
    rows, _, cols = dims.partition("x")
    if not rows or not cols:
        raise SystemExit(f"Malformed shape {spec!r}, expected [LABEL:]ROWSxCOLS")
    return (label or None, int(rows), int(cols))


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
        description="Benchmark legacy (CUTLASS) vs CuTeDSL FP8 2:4 sparse conversion."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        metavar="[LABEL:]ROWSxCOLS",
        help="shapes to benchmark, e.g. 8192x8192 or L2-7B/qkv_o:4096x4096",
    )
    parser.add_argument("--dtype", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument(
        "--bench-only",
        choices=("legacy", "cutedsl"),
        default=None,
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not is_sm_at_least_90():
        raise SystemExit(f"Requires SM 9.x+; got {torch.cuda.get_device_capability()}")

    shapes = [_parse_shape(s) for s in args.shapes]
    for _, _, cols in shapes:
        if cols % 8 != 0:
            raise SystemExit(f"cols must be divisible by 8, got {cols}")

    dtype = _dtype(args.dtype)
    labelled = any(label is not None for label, _, _ in shapes)
    rows_out = []
    for label, rows, cols in shapes:
        if args.deterministic:
            weight = _make_deterministic_sparse(rows, cols, dtype)
        else:
            weight = create_semi_structured_tensor(rows, cols, dtype=dtype)
        weight = weight.contiguous()

        row = {"Model layer": label or ""} if labelled else {}
        row["Weight MxK"] = f"{rows}x{cols}"
        if args.bench_only in (None, "legacy"):
            legacy_us = benchmark_function(
                to_sparse_semi_structured_cutlass_sm9x_f8,
                weight,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["Legacy us"] = f"{legacy_us:.2f}"
        if args.bench_only in (None, "cutedsl"):
            cutedsl_us = benchmark_function(
                _to_sparse_semi_structured_cutedsl,
                weight,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["CuTeDSL us"] = f"{cutedsl_us:.2f}"
        if args.bench_only is None:
            row["CuTeDSL speedup"] = f"{legacy_us / cutedsl_us:.2f}"
        rows_out.append(row)

    print(_markdown(rows_out, list(rows_out[0].keys())))


if __name__ == "__main__":
    main()
