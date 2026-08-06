# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_32x1_cutedsl,
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _missing_cutedsl_runtime_packages,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common import KernelPreference


def quantize_dim1_cuda(x, block_size, scale_mode):
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


def quantize_dim1_cutedsl(x, block_size, scale_mode):
    qdata, scales = mxfp8_quantize_2d_32x1_cutedsl(
        x,
        block_size=block_size,
        scaling_mode=scale_mode.value.lower(),
        blocked_scale_output=False,
    )
    return qdata.t(), scales


def run(grad_output, input_act, offs, block_size, scale_mode, choice):
    if choice == MXFP8Dim1CastKernelChoice.CUTEDSL:
        quantize_dim1 = quantize_dim1_cutedsl
    else:
        quantize_dim1 = quantize_dim1_cuda

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", "--m", dest="m", type=int, default=4096)
    parser.add_argument("--N", "--n", dest="n", type=int, default=4096)
    parser.add_argument("--K", "--k", dest="k", type=int, default=2048)
    parser.add_argument("--num-experts", "--groups", dest="groups", type=int, default=8)
    parser.add_argument("--multiple-of", type=int, default=128)
    parser.add_argument("--scale-mode", choices=("floor", "rceil"), default="rceil")
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

    block_size = 32
    scale_mode = ScaleCalculationMode(args.scale_mode)

    if args.deterministic:
        grad_output = (
            torch.arange(args.m * args.n, device="cuda", dtype=torch.float32)
            .remainder(251)
            .reshape(args.m, args.n)
            .to(torch.bfloat16)
        )
        input_act = (
            torch.arange(args.m * args.k, device="cuda", dtype=torch.float32)
            .remainder(251)
            .reshape(args.m, args.k)
            .to(torch.bfloat16)
        )
    else:
        torch.manual_seed(0)
        grad_output = torch.randn(args.m, args.n, dtype=torch.bfloat16, device="cuda")
        input_act = torch.randn(args.m, args.k, dtype=torch.bfloat16, device="cuda")

    offsets = generate_jagged_offs(
        args.groups,
        args.m,
        multiple_of=args.multiple_of,
        device="cuda",
        dtype=torch.int32,
    )

    print(
        f"shape=({args.m}, {args.n}, {args.k}) groups={args.groups} multiple_of={args.multiple_of} scale_mode={args.scale_mode}"
    )
    if args.bench_only is None:
        out_cuda = run(
            grad_output,
            input_act,
            offsets,
            block_size,
            scale_mode,
            MXFP8Dim1CastKernelChoice.CUDA,
        )
        out_cutedsl = run(
            grad_output,
            input_act,
            offsets,
            block_size,
            scale_mode,
            MXFP8Dim1CastKernelChoice.CUTEDSL,
        )
        torch.cuda.synchronize()

        print(f"cutedsl_equal={torch.equal(out_cuda, out_cutedsl)}")
        if not torch.equal(out_cuda, out_cutedsl):
            diff = out_cuda != out_cutedsl
            print(f"cutedsl_diff_count={diff.sum().item()}")
            coords = diff.nonzero()[:16]
            cuda_flat = out_cuda.flatten()
            cutedsl_flat = out_cutedsl.flatten()
            idx = (
                coords[:, 0] * out_cuda.shape[1] * out_cuda.shape[2]
                + coords[:, 1] * out_cuda.shape[2]
                + coords[:, 2]
            )
            print(f"first_diff_coords={coords.cpu().tolist()}")
            print(f"first_diff_flat_indices={idx.cpu().tolist()}")
            print(f"cuda_values={cuda_flat[idx].float().cpu().tolist()}")
            print(f"cutedsl_values={cutedsl_flat[idx].float().cpu().tolist()}")
            print(f"sqnr={compute_error(out_cuda, out_cutedsl).item():.2f}")
            print(f"offsets={offsets.cpu().tolist()}")

    if args.bench_only in (None, "cuda"):
        cuda_us = benchmark_function(
            run,
            grad_output,
            input_act,
            offsets,
            block_size,
            scale_mode,
            MXFP8Dim1CastKernelChoice.CUDA,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cuda_us={cuda_us:.2f}")
    if args.bench_only in (None, "cutedsl"):
        cutedsl_us = benchmark_function(
            run,
            grad_output,
            input_act,
            offsets,
            block_size,
            scale_mode,
            MXFP8Dim1CastKernelChoice.CUTEDSL,
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cutedsl_us={cutedsl_us:.2f}")


if __name__ == "__main__":
    main()
