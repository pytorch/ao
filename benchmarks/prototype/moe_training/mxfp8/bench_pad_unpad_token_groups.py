# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing
import sys

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    _pad_token_groups_cutedsl,
    _unpad_token_groups_cutedsl,
    fused_pad_token_groups_cuda,
    fused_unpad_token_groups_cuda,
    torch_pad_token_groups,
    torch_unpad_token_groups,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs

# fused_{pad,unpad}_token_groups_cuda can hit an illegal memory access for some
# large shapes, which poisons the CUDA context; not fixing since this legacy
# kernel will be deleted soon, so run it in a subprocess to isolate it instead.
_CUDA_WORKER_TIMEOUT_S = 300


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
    result = start.elapsed_time(end) * 1000.0 / iters
    # Tear the graph (and its private memory pool) down before the caller
    # captures another one for a different shape.
    del graph
    torch.cuda.synchronize()
    return result


def benchmark_function(f, *args, use_cuda_graph=False, graph_iters=1000, **kwargs):
    if use_cuda_graph:
        return benchmark_cuda_graph_function_in_microseconds(
            f,
            *args,
            iters=graph_iters,
            **kwargs,
        )
    return benchmark_cuda_function_in_microseconds(f, *args, **kwargs)


def _make_inputs(tokens, dim, dtype, deterministic):
    if deterministic:
        return (
            torch.arange(tokens * dim, device="cuda", dtype=torch.float32)
            .remainder(17)
            .sub(8)
            .div(16)
            .reshape(tokens, dim)
            .to(dtype)
        )
    return torch.randn(tokens, dim, dtype=dtype, device="cuda")


def _pad_cuda_worker(
    tokens,
    dim,
    groups,
    alignment_size,
    dtype_name,
    deterministic,
    use_cuda_graph,
    graph_iters,
    multiple_of,
    out_queue,
):
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    inputs = _make_inputs(tokens, dim, dtype, deterministic)
    offsets = generate_jagged_offs(
        groups, tokens, multiple_of=multiple_of, device="cuda"
    )
    us = benchmark_function(
        fused_pad_token_groups_cuda,
        inputs,
        offsets,
        alignment_size,
        use_cuda_graph=use_cuda_graph,
        graph_iters=graph_iters,
    )
    out_queue.put(us)


def _unpad_cuda_worker(
    tokens,
    dim,
    groups,
    alignment_size,
    dtype_name,
    deterministic,
    use_cuda_graph,
    graph_iters,
    multiple_of,
    out_queue,
):
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    inputs = _make_inputs(tokens, dim, dtype, deterministic)
    offsets = generate_jagged_offs(
        groups, tokens, multiple_of=multiple_of, device="cuda"
    )
    padded_inputs, padded_start_offsets, _ = torch_pad_token_groups(
        inputs, offsets, alignment_size
    )
    us = benchmark_function(
        fused_unpad_token_groups_cuda,
        padded_inputs,
        offsets,
        padded_start_offsets,
        tokens,
        alignment_size,
        use_cuda_graph=use_cuda_graph,
        graph_iters=graph_iters,
    )
    out_queue.put(us)


def _bench_cuda_isolated(
    worker,
    op_name,
    tokens,
    dim,
    groups,
    alignment_size,
    dtype_name,
    deterministic,
    use_cuda_graph,
    graph_iters,
    multiple_of,
):
    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue()
    proc = ctx.Process(
        target=worker,
        args=(
            tokens,
            dim,
            groups,
            alignment_size,
            dtype_name,
            deterministic,
            use_cuda_graph,
            graph_iters,
            multiple_of,
            out_queue,
        ),
    )
    proc.start()
    proc.join(_CUDA_WORKER_TIMEOUT_S)
    if proc.exitcode == 0 and not out_queue.empty():
        return out_queue.get()
    if proc.is_alive():
        proc.terminate()
    print(
        f"warning: cuda {op_name} crashed for shape {tokens}x{dim}x{groups}, marking N/A",
        file=sys.stderr,
    )
    return None


def _parse_shape(spec: str):
    """Parses `TOKENSxDIMxGROUPS` or `LABEL:TOKENSxDIMxGROUPS`."""
    label, _, dims = spec.rpartition(":")
    parts = dims.split("x")
    if len(parts) != 3:
        raise SystemExit(
            f"Malformed shape {spec!r}, expected [LABEL:]TOKENSxDIMxGROUPS"
        )
    tokens, dim, groups = (int(p) for p in parts)
    return (label or None, tokens, dim, groups)


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
        description="Benchmark torch / cuda / CuTeDSL MXFP8 token-group padding and unpadding."
    )
    parser.add_argument(
        "shapes",
        nargs="+",
        metavar="[LABEL:]TOKENSxDIMxGROUPS",
        help="shapes to benchmark, e.g. 16384x7168x8 or dsv3:16384x7168x8",
    )
    parser.add_argument("--mode", choices=("pad", "unpad", "both"), default="both")
    parser.add_argument("--alignment-size", type=int, default=32)
    parser.add_argument("--multiple-of", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--bench-only",
        nargs="+",
        choices=("torch", "cuda", "cutedsl"),
        default=None,
        help="restrict to one or more backends; default is all",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    shapes = [_parse_shape(s) for s in args.shapes]
    labelled = any(label is not None for label, _, _, _ in shapes)
    selected = set(args.bench_only) if args.bench_only else {"torch", "cuda", "cutedsl"}
    if args.cuda_graph_bench and "torch" in selected:
        # torch_{pad,unpad}_token_groups call .tolist(), a synchronous D2H copy
        # that CUDA graph capture never tolerates, for any shape.
        if selected == {"torch"}:
            raise SystemExit("torch cannot be benchmarked under --cuda-graph-bench")
        print(
            "warning: excluding torch from --cuda-graph-bench (unpinned D2H copy)",
            file=sys.stderr,
        )
        selected.discard("torch")

    do_pad = args.mode in ("pad", "both")
    do_unpad = args.mode in ("unpad", "both")
    rows_out = []

    for label, tokens, dim, groups in shapes:
        inputs = _make_inputs(tokens, dim, dtype, args.deterministic)
        offsets = generate_jagged_offs(
            groups, tokens, multiple_of=args.multiple_of, device="cuda"
        )

        row = {"Model layer": label or ""} if labelled else {}
        row["Tokens x Dim"] = f"{tokens}x{dim}"
        row["Groups"] = groups

        if do_pad:
            pad_cuda_us = None
            call_args = (inputs, offsets, args.alignment_size)
            if "torch" in selected:
                us = benchmark_function(
                    torch_pad_token_groups,
                    *call_args,
                    use_cuda_graph=args.cuda_graph_bench,
                    graph_iters=args.graph_iters,
                )
                row["Pad Torch us"] = f"{us:.2f}"
            if "cuda" in selected:
                pad_cuda_us = _bench_cuda_isolated(
                    _pad_cuda_worker,
                    "pad",
                    tokens,
                    dim,
                    groups,
                    args.alignment_size,
                    args.dtype,
                    args.deterministic,
                    args.cuda_graph_bench,
                    args.graph_iters,
                    args.multiple_of,
                )
                row["Pad CUDA us"] = (
                    f"{pad_cuda_us:.2f}" if pad_cuda_us is not None else "N/A"
                )
            if "cutedsl" in selected:
                out, _, _ = _pad_token_groups_cutedsl(*call_args)
                pad_cutedsl_us = benchmark_function(
                    _pad_token_groups_cutedsl,
                    *call_args,
                    use_cuda_graph=args.cuda_graph_bench,
                    graph_iters=args.graph_iters,
                )
                row["Pad CuTeDSL us"] = f"{pad_cutedsl_us:.2f}"
                moved_bytes = (inputs.numel() + out.numel()) * inputs.element_size()
                row["Pad CuTeDSL GB/s"] = (
                    f"{(moved_bytes / 1e9) / (pad_cutedsl_us / 1e6):.1f}"
                )
            if "cuda" in selected and "cutedsl" in selected:
                row["Pad CuTeDSL speedup"] = (
                    f"{pad_cuda_us / pad_cutedsl_us:.2f}"
                    if pad_cuda_us is not None
                    else "N/A"
                )

        if do_unpad:
            unpad_cuda_us = None
            padded_inputs, padded_start_offsets, _ = torch_pad_token_groups(
                inputs, offsets, args.alignment_size
            )
            call_args = (
                padded_inputs,
                offsets,
                padded_start_offsets,
                tokens,
                args.alignment_size,
            )
            if "torch" in selected:
                us = benchmark_function(
                    torch_unpad_token_groups,
                    *call_args,
                    use_cuda_graph=args.cuda_graph_bench,
                    graph_iters=args.graph_iters,
                )
                row["Unpad Torch us"] = f"{us:.2f}"
            if "cuda" in selected:
                unpad_cuda_us = _bench_cuda_isolated(
                    _unpad_cuda_worker,
                    "unpad",
                    tokens,
                    dim,
                    groups,
                    args.alignment_size,
                    args.dtype,
                    args.deterministic,
                    args.cuda_graph_bench,
                    args.graph_iters,
                    args.multiple_of,
                )
                row["Unpad CUDA us"] = (
                    f"{unpad_cuda_us:.2f}" if unpad_cuda_us is not None else "N/A"
                )
            if "cutedsl" in selected:
                out = _unpad_token_groups_cutedsl(*call_args)
                unpad_cutedsl_us = benchmark_function(
                    _unpad_token_groups_cutedsl,
                    *call_args,
                    use_cuda_graph=args.cuda_graph_bench,
                    graph_iters=args.graph_iters,
                )
                row["Unpad CuTeDSL us"] = f"{unpad_cutedsl_us:.2f}"
                moved_bytes = (
                    padded_inputs.numel() + out.numel()
                ) * inputs.element_size()
                row["Unpad CuTeDSL GB/s"] = (
                    f"{(moved_bytes / 1e9) / (unpad_cutedsl_us / 1e6):.1f}"
                )
            if "cuda" in selected and "cutedsl" in selected:
                row["Unpad CuTeDSL speedup"] = (
                    f"{unpad_cuda_us / unpad_cutedsl_us:.2f}"
                    if unpad_cuda_us is not None
                    else "N/A"
                )

        rows_out.append(row)

    print(_markdown(rows_out, list(rows_out[0].keys())))


if __name__ == "__main__":
    main()
