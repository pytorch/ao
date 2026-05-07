# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass

import torch
from tabulate import tabulate
from triton.testing import do_bench

from torchao.prototype.moe_training.config import (
    Float8TrainingOpConfig,
    Float8TrainingRecipe,
)
from torchao.prototype.moe_training.fp8_tensorwise_grouped_mm import (
    _to_fp8_tensorwise_then_scaled_grouped_mm,
)

try:
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig as TurboFloat8QuantConfig,
        Format as TurboFormat,
        ScalingGranularity as TurboScalingGranularity,
    )
    from primus_turbo.pytorch.ops import grouped_gemm_fp8 as turbo_grouped_gemm_fp8
except ImportError:
    TurboFloat8QuantConfig = None
    TurboFormat = None
    TurboScalingGranularity = None
    turbo_grouped_gemm_fp8 = None


@dataclass(frozen=True)
class BenchmarkShape:
    experts: int
    tokens_per_expert: int
    k: int
    n: int


def make_inputs(shape: BenchmarkShape):
    total_tokens = shape.experts * shape.tokens_per_expert
    a = torch.randn(
        total_tokens,
        shape.k,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    b = torch.randn(
        shape.experts,
        shape.n,
        shape.k,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = b.requires_grad_(True)
    b_t = b.transpose(-2, -1)
    offs = torch.arange(
        shape.tokens_per_expert,
        total_tokens + 1,
        shape.tokens_per_expert,
        device="cuda",
        dtype=torch.int32,
    )
    group_lens = torch.full(
        (shape.experts,),
        shape.tokens_per_expert,
        device="cuda",
        dtype=torch.int64,
    )
    return a, b, b_t, offs, group_lens


def benchmark_forward(fn):
    return do_bench(fn, return_mode="median") * 1e3


def benchmark_forward_backward(fn):
    out = fn()
    grad = torch.randn_like(out)

    def fwd_bwd():
        out = fn()
        out.backward(grad, retain_graph=True)

    return do_bench(fwd_bwd, return_mode="median") * 1e3


def benchmark_backward(fn):
    out = fn()
    grad = torch.randn_like(out)

    def bwd():
        out.backward(grad, retain_graph=True)

    return do_bench(bwd, return_mode="median") * 1e3


def run_shape(shape: BenchmarkShape, args: argparse.Namespace):
    config = Float8TrainingOpConfig.from_recipe(Float8TrainingRecipe.FP8_TENSORWISE)
    a, b, b_t, offs, group_lens = make_inputs(shape)

    def tensorwise_grouped_mm():
        return _to_fp8_tensorwise_then_scaled_grouped_mm(
            a,
            b_t,
            offs,
            config.out_dtype,
            config.float8_dtype,
        )

    def bf16_grouped_mm():
        return torch._grouped_mm(
            a,
            b_t,
            offs,
            out_dtype=config.out_dtype,
        )

    turbo_config = None
    if args.compare_turbo:
        if turbo_grouped_gemm_fp8 is None:
            raise RuntimeError("primus_turbo is not importable in this environment")
        turbo_config = TurboFloat8QuantConfig(
            format=TurboFormat.E4M3,
            granularity=TurboScalingGranularity.TENSORWISE,
        )

        def turbo_tensorwise_grouped_mm():
            return turbo_grouped_gemm_fp8(
                a,
                b,
                group_lens,
                trans_b=True,
                config=turbo_config,
            )

    # Compile/autotune once before measuring.
    tensorwise_grouped_mm()
    bf16_grouped_mm()
    if args.compare_turbo:
        turbo_tensorwise_grouped_mm()
    torch.cuda.synchronize()

    row = {
        "experts": shape.experts,
        "tokens_per_expert": shape.tokens_per_expert,
        "M": shape.experts * shape.tokens_per_expert,
        "K": shape.k,
        "N": shape.n,
        "fp8_dtype": str(config.float8_dtype).replace("torch.", ""),
        "bf16_fwd_us": benchmark_forward(bf16_grouped_mm),
        "tensorwise_fwd_us": benchmark_forward(tensorwise_grouped_mm),
    }

    if args.backward:
        row["tensorwise_bwd_us"] = benchmark_backward(tensorwise_grouped_mm)
        row["tensorwise_fwd_bwd_us"] = benchmark_forward_backward(tensorwise_grouped_mm)

    if args.compare_turbo:
        row["turbo_fwd_us"] = benchmark_forward(turbo_tensorwise_grouped_mm)
        if args.backward:
            row["turbo_bwd_us"] = benchmark_backward(turbo_tensorwise_grouped_mm)
            row["turbo_fwd_bwd_us"] = benchmark_forward_backward(
                turbo_tensorwise_grouped_mm
            )

    return row


def parse_shape(raw_shape: str) -> BenchmarkShape:
    experts, tokens_per_expert, k, n = (int(part) for part in raw_shape.split(","))
    return BenchmarkShape(experts, tokens_per_expert, k, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Shape as experts,tokens_per_expert,k,n. May be repeated.",
    )
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--compare-turbo",
        action="store_true",
        help="Also benchmark Primus Turbo tensorwise FP8 grouped GEMM.",
    )
    args = parser.parse_args()

    shapes = (
        [parse_shape(raw_shape) for raw_shape in args.shape]
        if args.shape
        else [
            BenchmarkShape(1, 256, 1024, 1024),
            BenchmarkShape(2, 256, 1024, 1024),
            BenchmarkShape(4, 256, 1024, 1024),
            BenchmarkShape(8, 256, 2048, 2048),
        ]
    )

    torch.manual_seed(0)
    print(f"torch: {torch.__version__}")
    print(f"hip: {torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    rows = [run_shape(shape, args) for shape in shapes]
    print(tabulate(rows, headers="keys", floatfmt=".2f"))


if __name__ == "__main__":
    main()
