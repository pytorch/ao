# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mx_block_rearrange_2d_k_groups_cutedsl,
    torch_to_blocked_2d_K_groups,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--multiple-of", type=int, default=1)
    parser.add_argument(
        "--chunk-width", type=int, default=0, choices=(0, 16, 32, 64, 128)
    )
    parser.add_argument(
        "--bench-only",
        choices=("triton", "cutedsl"),
        default=None,
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if torch.cuda.get_device_capability()[0] != 10:
        raise SystemExit(f"Requires SM 10.x; got {torch.cuda.get_device_capability()}")
    if not _mxfp8_cutedsl_kernels_available:
        raise SystemExit("CuteDSL mxfp8 kernels are unavailable")

    if args.deterministic:
        scales = (
            torch.arange(args.rows * args.cols, device="cuda", dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .reshape(args.rows, args.cols)
            .view(torch.float8_e8m0fnu)
        )
    else:
        scales = torch.randint(
            0,
            255,
            (args.rows, args.cols),
            device="cuda",
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu)
    scales_ref = scales.contiguous()
    offsets = generate_jagged_offs(
        args.groups,
        args.cols,
        multiple_of=args.multiple_of,
        device="cuda",
        dtype=torch.int32,
    )

    chunk_width = None if args.chunk_width == 0 else args.chunk_width

    print(
        f"shape={tuple(scales.shape)} strides={scales.stride()} groups={args.groups} chunk_width={chunk_width or 'auto'}"
    )
    if args.bench_only is None:
        out_ref, _ = torch_to_blocked_2d_K_groups(scales_ref, offsets)
        out_triton = triton_mx_block_rearrange_2d_K_groups(scales_ref, offsets)
        out_cutedsl = mx_block_rearrange_2d_k_groups_cutedsl(
            scales,
            offsets,
            chunk_width=chunk_width,
        )
        torch.cuda.synchronize()

        print(f"triton_equal={torch.equal(out_ref, out_triton)}")
        print(f"cutedsl_equal={torch.equal(out_ref, out_cutedsl)}")
        if not torch.equal(out_ref, out_cutedsl):
            diff = out_ref.view(torch.uint8) != out_cutedsl.view(torch.uint8)
            print(f"cutedsl_diff_count={diff.sum().item()}")
            coords = diff.nonzero()[:16]
            idx = coords[:, 0] * out_ref.shape[1] + coords[:, 1]
            ref_flat = out_ref.view(torch.uint8).flatten()
            cutedsl_flat = out_cutedsl.view(torch.uint8).flatten()
            print(f"first_diff_coords_v2={coords.cpu().tolist()}")
            print(f"first_diff_flat_indices_v2={idx.cpu().tolist()}")
            print(f"ref_values={ref_flat[idx].cpu().tolist()}")
            print(f"cutedsl_values={cutedsl_flat[idx].cpu().tolist()}")
            print(f"offsets={offsets.cpu().tolist()}")

    if args.bench_only in (None, "triton"):
        triton_us = benchmark_function(
            triton_mx_block_rearrange_2d_K_groups,
            scales_ref,
            offsets,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"triton_us={triton_us:.2f}")
    if args.bench_only in (None, "cutedsl"):
        cutedsl_us = benchmark_function(
            mx_block_rearrange_2d_k_groups_cutedsl,
            scales,
            offsets,
            chunk_width,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cutedsl_us={cutedsl_us:.2f}")


if __name__ == "__main__":
    main()
