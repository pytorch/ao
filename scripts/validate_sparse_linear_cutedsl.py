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


def _sqnr(x: torch.Tensor, y: torch.Tensor) -> float:
    diff = (x.float() - y.float()).square().sum()
    signal = x.float().square().sum()
    if diff == 0:
        return float("inf")
    return (10 * torch.log10(signal / diff)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=256)
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
    parser.add_argument(
        "--cutedsl-backend",
        choices=("cutedsl",),
        default="cutedsl",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not is_sm_at_least_90():
        raise SystemExit(f"Requires SM 9.x+; got {torch.cuda.get_device_capability()}")
    if args.K % 32 != 0:
        raise SystemExit("K must be divisible by 32")
    if args.N % 8 != 0:
        raise SystemExit("N must be divisible by 8")

    input_dtype = _float8_dtype(args.input_dtype)
    weight_dtype = _float8_dtype(args.weight_dtype)
    scale_dtype = _float_dtype(args.scale_dtype)
    out_dtype = _float_dtype(args.out_dtype)

    if args.deterministic:
        input = _make_deterministic_dense(args.M, args.K, input_dtype)
        weight_dense = _make_deterministic_sparse(args.N, args.K, weight_dtype)
    else:
        input = torch.randn(
            (args.M, args.K),
            dtype=torch.bfloat16,
            device="cuda",
        ).to(input_dtype)
        weight_dense = create_semi_structured_tensor(
            args.N,
            args.K,
            dtype=weight_dtype,
        )
    input_scale = torch.rand((args.M,), dtype=scale_dtype, device="cuda") + 0.5
    weight_scale = torch.rand((args.N,), dtype=scale_dtype, device="cuda") + 0.5
    bias = torch.randn((args.N,), dtype=out_dtype, device="cuda") if args.bias else None
    weight, weight_meta = to_sparse_semi_structured_cutlass_sm9x_f8(weight_dense)
    cutedsl_backend = (
        args.bench_only if args.bench_only == "cutedsl" else args.cutedsl_backend
    )

    print(
        f"shape=({args.M}, {args.N}, {args.K}) "
        f"input_dtype={args.input_dtype} weight_dtype={args.weight_dtype} "
        f"scale_dtype={args.scale_dtype} out_dtype={args.out_dtype} "
        f"bias={args.bias} cutedsl_backend={cutedsl_backend}"
    )
    if args.bench_only is None:
        legacy = rowwise_scaled_linear_sparse_cutlass_f8f8(
            input,
            input_scale,
            weight,
            weight_meta,
            weight_scale,
            bias,
            out_dtype,
            backend="legacy",
        )
        cutedsl = rowwise_scaled_linear_sparse_cutlass_f8f8(
            input,
            input_scale,
            weight,
            weight_meta,
            weight_scale,
            bias,
            out_dtype,
            backend=cutedsl_backend,
        )
        torch.cuda.synchronize()
        close = torch.allclose(legacy, cutedsl, rtol=1e-2, atol=1e-1)
        max_abs_diff = (legacy.float() - cutedsl.float()).abs().max().item()
        print(f"cutedsl_close={close}")
        print(f"sqnr={_sqnr(legacy, cutedsl):.2f}")
        print(f"max_abs_diff={max_abs_diff:.3e}")

    if args.bench_only in (None, "legacy"):
        legacy_us = benchmark_function(
            rowwise_scaled_linear_sparse_cutlass_f8f8,
            input,
            input_scale,
            weight,
            weight_meta,
            weight_scale,
            bias,
            out_dtype,
            backend="legacy",
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"legacy_us={legacy_us:.2f}")
    if args.bench_only in (None, "cutedsl"):
        cutedsl_us = benchmark_function(
            rowwise_scaled_linear_sparse_cutlass_f8f8,
            input,
            input_scale,
            weight,
            weight_meta,
            weight_scale,
            bias,
            out_dtype,
            backend=cutedsl_backend,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cutedsl_us={cutedsl_us:.2f}")


if __name__ == "__main__":
    main()
