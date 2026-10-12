# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import torch.nn.functional as F

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.mxfp8_linear import _to_mxfp8_then_scaled_mm
from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.quantization.quantize_.common import KernelPreference

# The cast kernels each backend label selects. "legacy" is what the CuTeDSL
# kernels replace.
_BACKEND_CAST_KERNEL_CHOICES = {
    "cutedsl": (
        MXFP8Dim0CastKernelChoice.CUTEDSL,
        MXFP8Dim1CastKernelChoice.CUTEDSL,
    ),
    "legacy": (
        MXFP8Dim0CastKernelChoice.TRITON,
        MXFP8Dim1CastKernelChoice.CUDA,
    ),
}


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


def _fwd_bwd(x, weight, labels, scale_mode, wgrad_with_hp, backend):
    """One full MXFP8 linear step: forward plus the two backward gemms."""
    dim0_choice, dim1_choice = _BACKEND_CAST_KERNEL_CHOICES[backend]
    if x.grad is not None:
        x.grad.zero_()
    if weight.grad is not None:
        weight.grad.zero_()
    out = _to_mxfp8_then_scaled_mm(
        x,
        weight,
        kernel_preference=KernelPreference.AUTO,
        scale_calculation_mode=scale_mode,
        wgrad_with_hp=wgrad_with_hp,
        mxfp8_dim0_cast_kernel_choice=dim0_choice,
        mxfp8_dim1_cast_kernel_choice=dim1_choice,
    )
    F.mse_loss(out, labels).backward()
    return out


def _parse_shape(spec: str):
    """Parses `MxNxK` or `LABEL:MxNxK`."""
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
        description="Benchmark the MXFP8 linear forward + backward with the legacy "
        "(triton/cuda) vs CuTeDSL cast kernels."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        metavar="[LABEL:]MxNxK",
        help="shapes to benchmark, e.g. 4096x4096x2048 or dsv3:4096x4096x2048",
    )
    parser.add_argument("--scale-mode", choices=("floor", "rceil"), default="rceil")
    parser.add_argument("--wgrad-with-hp", action="store_true")
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

    shapes = [_parse_shape(s) for s in args.shapes]
    labelled = any(label is not None for label, _, _, _ in shapes)
    scale_mode = ScaleCalculationMode(args.scale_mode)
    rows_out = []

    for label, m, n, k in shapes:
        if args.deterministic:
            x = (
                torch.arange(m * k, device="cuda", dtype=torch.float32)
                .remainder(251)
                .reshape(m, k)
                .to(torch.bfloat16)
            )
            weight = (
                torch.arange(n * k, device="cuda", dtype=torch.float32)
                .remainder(251)
                .reshape(n, k)
                .to(torch.bfloat16)
            )
        else:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        labels = torch.ones((m, n), dtype=torch.bfloat16, device="cuda")

        row = {"Model layer": label or ""} if labelled else {}
        row["MxNxK"] = f"{m}x{n}x{k}"

        for backend in ("legacy", "cutedsl"):
            if args.bench_only not in (None, backend):
                continue
            # Fresh leaves per backend so the grad buffers are not shared.
            x_bench = x.detach().clone().requires_grad_(True)
            weight_bench = weight.detach().clone().requires_grad_(True)
            us = benchmark_function(
                _fwd_bwd,
                x_bench,
                weight_bench,
                labels,
                scale_mode,
                args.wgrad_with_hp,
                backend,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
            row["Legacy us" if backend == "legacy" else "CuTeDSL us"] = f"{us:.2f}"

        if args.bench_only is None:
            row["CuTeDSL speedup"] = (
                f"{float(row['Legacy us']) / float(row['CuTeDSL us']):.2f}"
            )
        rows_out.append(row)

    print(_markdown(rows_out, list(rows_out[0].keys())))


if __name__ == "__main__":
    main()
