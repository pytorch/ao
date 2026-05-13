# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark forward and backward of FP8 grouped GEMM for an MoE layer using a
DeepSeek-V3 sized config.

Compares:
  - bf16 grouped_mm
  - FP8 rowwise grouped_mm
  - FP8 tensorwise grouped_mm

Example:
  python bench_fp8_grouped_mm_deepseek_v3.py \
      --batch-size 1 --seq-len 4096 --ep 8 --backward \
      --trace-dir benchmark_results/deepseek_grouped_mm_trace
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
from tabulate import tabulate
from torch.profiler import ProfilerActivity, profile, record_function
from triton.testing import do_bench

from torchao.prototype.moe_training.config import (
    Float8TrainingOpConfig,
    Float8TrainingRecipe,
)
from torchao.prototype.moe_training.fp8_grouped_mm import (
    _to_fp8_rowwise_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.fp8_tensorwise_grouped_mm import (
    _to_fp8_tensorwise_then_scaled_grouped_mm,
)


DSV3_NUM_ROUTED_EXPERTS = 256
DSV3_TOP_K = 8
DSV3_HIDDEN_DIM = 7168
DSV3_INTERMEDIATE_DIM = 2048


@dataclass(frozen=True)
class GemmShape:
    """Shape of a single grouped GEMM A (M,K) @ B_t (E,K,N) -> (M,N)."""

    name: str
    experts: int
    tokens_per_expert: int
    k: int
    n: int

    @property
    def M(self) -> int:
        return self.experts * self.tokens_per_expert


def make_inputs(shape: GemmShape):
    total_tokens = shape.M
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
    ).requires_grad_(True)
    b_t = b.transpose(-2, -1)
    offs = torch.arange(
        shape.tokens_per_expert,
        total_tokens + 1,
        shape.tokens_per_expert,
        device="cuda",
        dtype=torch.int32,
    )
    return a, b, b_t, offs


def bench_us(fn: Callable[[], torch.Tensor]) -> float:
    return do_bench(fn, return_mode="median") * 1e3


def bench_backward(fn: Callable[[], torch.Tensor]) -> float:
    out = fn()
    grad = torch.randn_like(out)

    def bwd():
        out.backward(grad, retain_graph=True)

    return bench_us(bwd)


def bench_forward_backward(fn: Callable[[], torch.Tensor]) -> float:
    out = fn()
    grad = torch.randn_like(out)

    def fwd_bwd():
        out = fn()
        out.backward(grad, retain_graph=True)

    return bench_us(fwd_bwd)


def make_fns(
    shape: GemmShape,
) -> Dict[str, Callable[[], torch.Tensor]]:
    config = Float8TrainingOpConfig.from_recipe(Float8TrainingRecipe.FP8_TENSORWISE)
    a, _b, b_t, offs = make_inputs(shape)

    def bf16_fn():
        return torch._grouped_mm(a, b_t, offs, out_dtype=config.out_dtype)

    def rowwise_fn():
        return _to_fp8_rowwise_then_scaled_grouped_mm(
            a,
            b_t,
            offs,
            config.out_dtype,
            config.float8_dtype,
            pad_token_groups_for_grouped_mm=False,
        )

    def tensorwise_fn():
        return _to_fp8_tensorwise_then_scaled_grouped_mm(
            a,
            b_t,
            offs,
            config.out_dtype,
            config.float8_dtype,
        )

    return {
        "bf16": bf16_fn,
        "rowwise": rowwise_fn,
        "tensorwise": tensorwise_fn,
    }


def run_shape(shape: GemmShape, args: argparse.Namespace) -> Dict[str, object]:
    fns = make_fns(shape)

    for fn in fns.values():
        fn()
    torch.cuda.synchronize()

    row: Dict[str, object] = {
        "gemm": shape.name,
        "E": shape.experts,
        "tokens_per_expert": shape.tokens_per_expert,
        "M": shape.M,
        "K": shape.k,
        "N": shape.n,
        "bf16_fwd_us": bench_us(fns["bf16"]),
        "rowwise_fwd_us": bench_us(fns["rowwise"]),
        "tensorwise_fwd_us": bench_us(fns["tensorwise"]),
    }
    row["rowwise_fwd_speedup"] = row["bf16_fwd_us"] / row["rowwise_fwd_us"]
    row["tensorwise_fwd_speedup"] = row["bf16_fwd_us"] / row["tensorwise_fwd_us"]

    if args.backward:
        row["bf16_bwd_us"] = bench_backward(fns["bf16"])
        row["rowwise_bwd_us"] = bench_backward(fns["rowwise"])
        row["tensorwise_bwd_us"] = bench_backward(fns["tensorwise"])
        row["bf16_fwd_bwd_us"] = bench_forward_backward(fns["bf16"])
        row["rowwise_fwd_bwd_us"] = bench_forward_backward(fns["rowwise"])
        row["tensorwise_fwd_bwd_us"] = bench_forward_backward(fns["tensorwise"])
        row["rowwise_fwd_bwd_speedup"] = (
            row["bf16_fwd_bwd_us"] / row["rowwise_fwd_bwd_us"]
        )
        row["tensorwise_fwd_bwd_speedup"] = (
            row["bf16_fwd_bwd_us"] / row["tensorwise_fwd_bwd_us"]
        )

    return row


def trace_shape(shape: GemmShape, args: argparse.Namespace) -> None:
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    fns = make_fns(shape)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    shape_label = shape.name.split()[0]
    trace_path = trace_dir / f"{shape_label}_M{shape.M}_K{shape.k}_N{shape.n}.json"

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        for _ in range(args.trace_iters):
            for impl, fn in fns.items():
                with record_function(f"{shape.name}:{impl}:forward"):
                    out = fn()
                if args.backward:
                    grad = torch.randn_like(out)
                    with record_function(f"{shape.name}:{impl}:backward"):
                        out.backward(grad, retain_graph=True)
                torch.cuda.synchronize()
                prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"Wrote trace: {trace_path}")


def deepseek_v3_shapes(args: argparse.Namespace) -> List[GemmShape]:
    assert DSV3_NUM_ROUTED_EXPERTS % args.ep == 0, (
        f"num_routed_experts ({DSV3_NUM_ROUTED_EXPERTS}) must be divisible by "
        f"EP ({args.ep})"
    )
    num_local_experts = DSV3_NUM_ROUTED_EXPERTS // args.ep
    global_token_expert_assignments = args.batch_size * args.seq_len * DSV3_TOP_K
    per_rank_token_expert_assignments = global_token_expert_assignments // args.ep
    tokens_per_expert = per_rank_token_expert_assignments // num_local_experts

    if tokens_per_expert % 16 != 0:
        tokens_per_expert = ((tokens_per_expert + 15) // 16) * 16

    return [
        GemmShape(
            name="gate (w1)",
            experts=num_local_experts,
            tokens_per_expert=tokens_per_expert,
            k=DSV3_HIDDEN_DIM,
            n=DSV3_INTERMEDIATE_DIM,
        ),
        GemmShape(
            name="up (w3)",
            experts=num_local_experts,
            tokens_per_expert=tokens_per_expert,
            k=DSV3_HIDDEN_DIM,
            n=DSV3_INTERMEDIATE_DIM,
        ),
        GemmShape(
            name="down (w2)",
            experts=num_local_experts,
            tokens_per_expert=tokens_per_expert,
            k=DSV3_INTERMEDIATE_DIM,
            n=DSV3_HIDDEN_DIM,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek-V3 grouped GEMMs for bf16, rowwise, tensorwise."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="If set, write torch.profiler Chrome traces for each GEMM shape.",
    )
    parser.add_argument("--trace-iters", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"torch: {torch.__version__}")
    print(f"hip: {torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    shapes = deepseek_v3_shapes(args)
    print(
        f"\nDeepSeek-V3 MoE layer, EP={args.ep}: "
        f"num_local_experts={DSV3_NUM_ROUTED_EXPERTS // args.ep}, "
        f"global batch={args.batch_size}, seq_len={args.seq_len}, "
        f"top_k={DSV3_TOP_K}, tokens_per_expert={shapes[0].tokens_per_expert} "
        f"(M_per_rank={shapes[0].M})\n"
    )

    rows = [run_shape(shape, args) for shape in shapes]
    print(tabulate(rows, headers="keys", floatfmt=".2f"))

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(0),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "ep": args.ep,
                "backward": args.backward,
            },
            "results": rows,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results: {output_path}")

    if args.trace_dir:
        for shape in shapes:
            trace_shape(shape, args)


if __name__ == "__main__":
    main()
