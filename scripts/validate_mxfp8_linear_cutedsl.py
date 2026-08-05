# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import torch.nn.functional as F

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels.mxfp8.cute_utils import (
    _missing_cutedsl_runtime_packages,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
)
from torchao.prototype.moe_training.mxfp8_linear import (
    _to_mxfp8_then_scaled_mm,
    set_mxfp8_linear_backend,
)
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference


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


def run(x, weight, labels, scale_mode, wgrad_with_hp, backend):
    set_mxfp8_linear_backend(backend)
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
        backend=backend,
    )
    F.mse_loss(out, labels).backward()
    return out, x.grad, weight.grad


def benchmark_backend(
    x,
    weight,
    labels,
    scale_mode,
    wgrad_with_hp,
    backend,
    use_cuda_graph=False,
    graph_iters=1000,
):
    x_bench = x.detach().clone().requires_grad_(True)
    weight_bench = weight.detach().clone().requires_grad_(True)
    return benchmark_function(
        run,
        x_bench,
        weight_bench,
        labels,
        scale_mode,
        wgrad_with_hp,
        backend,
        use_cuda_graph=use_cuda_graph,
        graph_iters=graph_iters,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", "--m", dest="m", type=int, default=4096)
    parser.add_argument("--N", "--n", dest="n", type=int, default=4096)
    parser.add_argument("--K", "--k", dest="k", type=int, default=2048)
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

    scale_mode = ScaleCalculationMode(args.scale_mode)
    if args.deterministic:
        x = (
            torch.arange(args.m * args.k, device="cuda", dtype=torch.float32)
            .remainder(251)
            .reshape(args.m, args.k)
            .to(torch.bfloat16)
        )
        weight = (
            torch.arange(args.n * args.k, device="cuda", dtype=torch.float32)
            .remainder(251)
            .reshape(args.n, args.k)
            .to(torch.bfloat16)
        )
    else:
        torch.manual_seed(0)
        x = torch.randn(args.m, args.k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(args.n, args.k, dtype=torch.bfloat16, device="cuda")
    labels = torch.ones((args.m, args.n), dtype=torch.bfloat16, device="cuda")

    print(
        f"shape=({args.m}, {args.n}, {args.k}) wgrad_with_hp={args.wgrad_with_hp} scale_mode={args.scale_mode}"
    )
    if args.bench_only is None:
        x_legacy = x.detach().clone().requires_grad_(True)
        weight_legacy = weight.detach().clone().requires_grad_(True)
        out_legacy, x_grad_legacy, weight_grad_legacy = run(
            x_legacy,
            weight_legacy,
            labels,
            scale_mode,
            args.wgrad_with_hp,
            "legacy",
        )
        x_cutedsl = x.detach().clone().requires_grad_(True)
        weight_cutedsl = weight.detach().clone().requires_grad_(True)
        out_cutedsl, x_grad_cutedsl, weight_grad_cutedsl = run(
            x_cutedsl,
            weight_cutedsl,
            labels,
            scale_mode,
            args.wgrad_with_hp,
            "cutedsl",
        )
        torch.cuda.synchronize()

        output_sqnr = compute_error(out_legacy, out_cutedsl)
        input_grad_sqnr = compute_error(x_grad_legacy, x_grad_cutedsl)
        weight_grad_sqnr = compute_error(weight_grad_legacy, weight_grad_cutedsl)
        print(f"output_sqnr={output_sqnr:.2f}")
        print(f"input_grad_sqnr={input_grad_sqnr:.2f}")
        print(f"weight_grad_sqnr={weight_grad_sqnr:.2f}")

    if args.bench_only in (None, "legacy"):
        legacy_us = benchmark_backend(
            x,
            weight,
            labels,
            scale_mode,
            args.wgrad_with_hp,
            "legacy",
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"legacy_us={legacy_us:.2f}")
    if args.bench_only in (None, "cutedsl"):
        cutedsl_us = benchmark_backend(
            x,
            weight,
            labels,
            scale_mode,
            args.wgrad_with_hp,
            "cutedsl",
            use_cuda_graph=args.cuda_graph_bench,
            graph_iters=args.graph_iters,
        )
        print(f"cutedsl_us={cutedsl_us:.2f}")


if __name__ == "__main__":
    main()
