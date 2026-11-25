# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
    to_sparse_semi_structured_cutlass_sm9x_f8,
)
from torchao.quantization.quantize_.workflows.float8.kernels import (
    _rowwise_scaled_linear_sparse_cutedsl,
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


def _float8_dtype(name: str) -> torch.dtype:
    if name == "e4m3":
        return torch.float8_e4m3fn
    if name == "e5m2":
        return torch.float8_e5m2
    raise AssertionError(f"Unsupported dtype: {name}")


def _float_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise AssertionError(f"Unsupported dtype: {name}")


def _make_deterministic_dense(rows: int, cols: int, dtype: torch.dtype):
    values = (
        torch.arange(rows * cols, device="cuda", dtype=torch.float32)
        .remainder(17)
        .sub(8)
        .div(16)
    ).reshape(rows, cols)
    return values.to(dtype).contiguous()


def _make_deterministic_sparse(rows: int, cols: int, dtype: torch.dtype):
    values = (
        torch.arange(rows * cols, device="cuda", dtype=torch.float32)
        .remainder(17)
        .sub(8)
        .div(16)
    ).reshape(rows, cols)
    mask = torch.arange(cols, device="cuda").remainder(4)
    keep = (mask == 0) | (mask == 3)
    raw = torch.where(keep.reshape(1, cols), values, torch.zeros_like(values))
    return raw.to(dtype).contiguous()


def _parse_shape(spec: str):
    """Parses `MxNxK` or `LABEL:MxNxK` into (label, m, n, k)."""
    label, _, dims = spec.rpartition(":")
    parts = dims.split("x")
    if len(parts) != 3:
        raise SystemExit(f"Malformed shape {spec!r}, expected [LABEL:]MxNxK")
    m, n, k = (int(p) for p in parts)
    return (label or None, m, n, k)


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
        description="Benchmark legacy (CUTLASS) vs CuTeDSL rowwise-scaled FP8 2:4 "
        "sparse linear."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        metavar="[LABEL:]MxNxK",
        help="shapes to benchmark, e.g. 2048x11008x4096 or "
        "L2-7B/gate_up:2048x11008x4096",
    )
    parser.add_argument("--input-dtype", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument("--weight-dtype", choices=("e4m3", "e5m2"), default="e4m3")
    parser.add_argument(
        "--scale-dtype",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--out-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--bias", action="store_true")
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
    for _, _, n, k in shapes:
        if k % 32 != 0:
            raise SystemExit(f"K must be divisible by 32, got {k}")
        if n % 8 != 0:
            raise SystemExit(f"N must be divisible by 8, got {n}")

    input_dtype = _float8_dtype(args.input_dtype)
    weight_dtype = _float8_dtype(args.weight_dtype)
    scale_dtype = _float_dtype(args.scale_dtype)
    out_dtype = _float_dtype(args.out_dtype)
    labelled = any(label is not None for label, _, _, _ in shapes)

    rows_out = []
    for label, m, n, k in shapes:
        if args.deterministic:
            input = _make_deterministic_dense(m, k, input_dtype)
            weight_dense = _make_deterministic_sparse(n, k, weight_dtype)
        else:
            input = torch.randn((m, k), dtype=torch.bfloat16, device="cuda").to(
                input_dtype
            )
            weight_dense = create_semi_structured_tensor(n, k, dtype=weight_dtype)
        input_scale = torch.rand((m,), dtype=scale_dtype, device="cuda") + 0.5
        weight_scale = torch.rand((n,), dtype=scale_dtype, device="cuda") + 0.5
        bias = torch.randn((n,), dtype=out_dtype, device="cuda") if args.bias else None
        weight, weight_meta = to_sparse_semi_structured_cutlass_sm9x_f8(weight_dense)
        call_args = (
            input,
            input_scale,
            weight,
            weight_meta,
            weight_scale,
            bias,
            out_dtype,
        )

        row = {"Model layer": label or ""} if labelled else {}
        row["M"] = m
        row["Weight NxK"] = f"{n}x{k}"
        if args.bench_only in (None, "legacy"):
            legacy_us = benchmark_function(
                rowwise_scaled_linear_sparse_cutlass_f8f8,
                *call_args,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["Legacy us"] = f"{legacy_us:.2f}"
        if args.bench_only in (None, "cutedsl"):
            cutedsl_us = benchmark_function(
                _rowwise_scaled_linear_sparse_cutedsl,
                *call_args,
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
