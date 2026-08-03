# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.ops import to_sparse_semi_structured_cutlass_sm9x_f8
from torchao.sparsity.utils import create_semi_structured_tensor
from torchao.utils import is_sm_at_least_90


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=8192)
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
    if args.cols % 8 != 0:
        raise SystemExit("cols must be divisible by 8")

    dtype = _dtype(args.dtype)
    if args.deterministic:
        weight = _make_deterministic_sparse(args.rows, args.cols, dtype)
    else:
        weight = create_semi_structured_tensor(args.rows, args.cols, dtype=dtype)
    weight = weight.contiguous()

    print(f"shape={tuple(weight.shape)} strides={weight.stride()} dtype={args.dtype}")
    if args.bench_only is None:
        legacy_data, legacy_meta = to_sparse_semi_structured_cutlass_sm9x_f8(
            weight,
            backend="legacy",
        )
        cutedsl_data, cutedsl_meta = to_sparse_semi_structured_cutlass_sm9x_f8(
            weight,
            backend="cutedsl",
        )
        torch.cuda.synchronize()

        legacy_data_u8 = legacy_data.view(torch.uint8)
        cutedsl_data_u8 = cutedsl_data.view(torch.uint8)
        data_equal = torch.equal(legacy_data_u8, cutedsl_data_u8)
        meta_equal = torch.equal(legacy_meta, cutedsl_meta)
        print(f"cutedsl_data_equal={data_equal}")
        print(f"cutedsl_meta_equal={meta_equal}")
        if not data_equal:
            diff = legacy_data_u8 != cutedsl_data_u8
            print(f"cutedsl_data_diff_count={diff.sum().item()}")
            coords = diff.nonzero()[:16]
            idx = coords[:, 0] * legacy_data.shape[1] + coords[:, 1]
            legacy_flat = legacy_data_u8.flatten()
            cutedsl_flat = cutedsl_data_u8.flatten()
            print(f"first_data_diff_coords={coords.cpu().tolist()}")
            print(f"legacy_data_values={legacy_flat[idx].cpu().tolist()}")
            print(f"cutedsl_data_values={cutedsl_flat[idx].cpu().tolist()}")
        if not meta_equal:
            diff = legacy_meta != cutedsl_meta
            print(f"cutedsl_meta_diff_count={diff.sum().item()}")
            coords = diff.nonzero()[:16]
            idx = coords[:, 0] * legacy_meta.shape[1] + coords[:, 1]
            legacy_flat = legacy_meta.flatten()
            cutedsl_flat = cutedsl_meta.flatten()
            print(f"first_meta_diff_coords={coords.cpu().tolist()}")
            print(f"legacy_meta_values={legacy_flat[idx].cpu().tolist()}")
            print(f"cutedsl_meta_values={cutedsl_flat[idx].cpu().tolist()}")

    if args.bench_only in (None, "legacy"):
        legacy_us = benchmark_function(
            to_sparse_semi_structured_cutlass_sm9x_f8,
            weight,
            backend="legacy",
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"legacy_us={legacy_us:.2f}")
    if args.bench_only in (None, "cutedsl"):
        cutedsl_us = benchmark_function(
            to_sparse_semi_structured_cutlass_sm9x_f8,
            weight,
            backend="cutedsl",
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cutedsl_us={cutedsl_us:.2f}")


if __name__ == "__main__":
    main()
