# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _missing_cutedsl_runtime_packages,
)
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_pad_token_groups import (
    pad_token_groups_cutedsl,
    unpad_token_groups_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
    fused_pad_token_groups_cuda,
    fused_unpad_token_groups_cuda,
    torch_pad_token_groups,
    torch_unpad_token_groups,
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


def maybe_benchmark_function(
    name,
    f,
    *args,
    use_cuda_graph=False,
    graph_iters=1000,
    graph_safe=True,
    **kwargs,
):
    if use_cuda_graph and not graph_safe:
        print(f"{name}=skipped_cuda_graph_unsafe")
        return None
    value = benchmark_function(
        f,
        *args,
        use_cuda_graph=use_cuda_graph,
        graph_iters=graph_iters,
        **kwargs,
    )
    print(f"{name}={value:.2f}")
    return value


def cuda_reference_supports_shape(dtype: str, dim: int) -> bool:
    return dtype == "bf16" and (dim < 8 or dim % 8 == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=7168)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--multiple-of", type=int, default=1)
    parser.add_argument("--alignment-size", type=int, default=32)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--mode", choices=("pad", "unpad", "both"), default="both")
    parser.add_argument("--bench-only", choices=("torch", "cuda", "cutedsl"))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cuda-graph-bench", action="store_true")
    parser.add_argument("--graph-iters", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if torch.cuda.get_device_capability()[0] != 10:
        raise SystemExit(f"Requires SM 10.x; got {torch.cuda.get_device_capability()}")
    if not _mxfp8_cutedsl_kernels_available:
        missing = _missing_cutedsl_runtime_packages()
        detail = (
            f"missing package(s): {', '.join(missing)}"
            if missing
            else "requires CUDA 12.8+ on SM 10.x"
        )
        raise SystemExit(f"CuteDSL mxfp8 kernels unavailable ({detail})")
    if args.bench_only in (None, "cuda") and not cuda_reference_supports_shape(
        args.dtype,
        args.dim,
    ):
        raise SystemExit(
            "CUDA pad/unpad reference requires bf16 with dim < 8 or dim % 8 == 0"
        )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.deterministic:
        inputs = (
            torch.arange(args.tokens * args.dim, device="cuda", dtype=torch.float32)
            .remainder(251)
            .reshape(args.tokens, args.dim)
            .to(dtype)
        )
    else:
        torch.manual_seed(0)
        inputs = torch.randn(args.tokens, args.dim, dtype=dtype, device="cuda")
    offsets = generate_jagged_offs(
        args.groups,
        args.tokens,
        multiple_of=args.multiple_of,
        device="cuda",
        dtype=torch.int32,
    )

    print(
        f"shape={tuple(inputs.shape)} dtype={args.dtype} groups={args.groups} alignment_size={args.alignment_size} mode={args.mode}"
    )

    if args.mode in ("pad", "both"):
        if args.bench_only is None:
            out_torch, start_torch, end_torch = torch_pad_token_groups(
                inputs,
                offsets,
                args.alignment_size,
            )
            out_cuda, start_cuda, end_cuda = fused_pad_token_groups_cuda(
                inputs,
                offsets,
                args.alignment_size,
            )
            out_cutedsl, start_cutedsl, end_cutedsl = pad_token_groups_cutedsl(
                inputs,
                offsets,
                args.alignment_size,
            )
            torch.cuda.synchronize()
            print(f"pad_cuda_equal={torch.equal(out_torch, out_cuda)}")
            print(f"pad_cutedsl_equal={torch.equal(out_torch, out_cutedsl)}")
            print(f"pad_cuda_start_equal={torch.equal(start_torch, start_cuda)}")
            print(f"pad_cutedsl_start_equal={torch.equal(start_torch, start_cutedsl)}")
            print(f"pad_cuda_end_equal={torch.equal(end_torch, end_cuda)}")
            print(f"pad_cutedsl_end_equal={torch.equal(end_torch, end_cutedsl)}")
            if not torch.equal(out_torch, out_cutedsl):
                diff = out_torch != out_cutedsl
                print(f"pad_cutedsl_diff_count={diff.sum().item()}")
                coords = diff.nonzero()[:16]
                idx = coords[:, 0] * out_torch.shape[1] + coords[:, 1]
                torch_flat = out_torch.flatten()
                cutedsl_flat = out_cutedsl.flatten()
                print(f"pad_first_diff_coords={coords.cpu().tolist()}")
                print(f"pad_first_diff_flat_indices={idx.cpu().tolist()}")
                print(f"pad_torch_values={torch_flat[idx].float().cpu().tolist()}")
                print(f"pad_cutedsl_values={cutedsl_flat[idx].float().cpu().tolist()}")
                print(f"offsets={offsets.cpu().tolist()}")
        if args.bench_only in (None, "torch"):
            maybe_benchmark_function(
                "pad_torch_us",
                torch_pad_token_groups,
                inputs,
                offsets,
                args.alignment_size,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
                graph_safe=False,
            )
        if args.bench_only in (None, "cuda"):
            maybe_benchmark_function(
                "pad_cuda_us",
                fused_pad_token_groups_cuda,
                inputs,
                offsets,
                args.alignment_size,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
        if args.bench_only in (None, "cutedsl"):
            maybe_benchmark_function(
                "pad_cutedsl_us",
                pad_token_groups_cutedsl,
                inputs,
                offsets,
                args.alignment_size,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )

    if args.mode in ("unpad", "both"):
        padded_inputs, padded_group_start_offsets, _ = torch_pad_token_groups(
            inputs,
            offsets,
            args.alignment_size,
        )
        if args.bench_only is None:
            out_torch = torch_unpad_token_groups(
                padded_inputs,
                offsets,
                padded_group_start_offsets,
                args.tokens,
                args.alignment_size,
            )
            out_cuda = fused_unpad_token_groups_cuda(
                padded_inputs,
                offsets,
                padded_group_start_offsets,
                args.tokens,
                args.alignment_size,
            )
            out_cutedsl = unpad_token_groups_cutedsl(
                padded_inputs,
                offsets,
                padded_group_start_offsets,
                args.tokens,
                args.alignment_size,
            )
            torch.cuda.synchronize()
            print(f"unpad_cuda_equal={torch.equal(out_torch, out_cuda)}")
            print(f"unpad_cutedsl_equal={torch.equal(out_torch, out_cutedsl)}")
            if not torch.equal(out_torch, out_cutedsl):
                diff = out_torch != out_cutedsl
                print(f"unpad_cutedsl_diff_count={diff.sum().item()}")
                coords = diff.nonzero()[:16]
                idx = coords[:, 0] * out_torch.shape[1] + coords[:, 1]
                torch_flat = out_torch.flatten()
                cutedsl_flat = out_cutedsl.flatten()
                print(f"unpad_first_diff_coords={coords.cpu().tolist()}")
                print(f"unpad_first_diff_flat_indices={idx.cpu().tolist()}")
                print(f"unpad_torch_values={torch_flat[idx].float().cpu().tolist()}")
                print(
                    f"unpad_cutedsl_values={cutedsl_flat[idx].float().cpu().tolist()}"
                )
                print(f"offsets={offsets.cpu().tolist()}")
        if args.bench_only in (None, "torch"):
            maybe_benchmark_function(
                "unpad_torch_us",
                torch_unpad_token_groups,
                padded_inputs,
                offsets,
                padded_group_start_offsets,
                args.tokens,
                args.alignment_size,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
                graph_safe=False,
            )
        if args.bench_only in (None, "cuda"):
            maybe_benchmark_function(
                "unpad_cuda_us",
                fused_unpad_token_groups_cuda,
                padded_inputs,
                offsets,
                padded_group_start_offsets,
                args.tokens,
                args.alignment_size,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )
        if args.bench_only in (None, "cutedsl"):
            maybe_benchmark_function(
                "unpad_cutedsl_us",
                unpad_token_groups_cutedsl,
                padded_inputs,
                offsets,
                padded_group_start_offsets,
                args.tokens,
                args.alignment_size,
                use_cuda_graph=args.cuda_graph_bench,
                graph_iters=args.graph_iters,
            )


if __name__ == "__main__":
    main()
