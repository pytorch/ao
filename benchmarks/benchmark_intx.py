# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark calibration-free quantization methods: INT8-INT4, INT8-INT4-HQQ

HQQ doesn't require calibration flow and can be applied directly without element-wise operations.

Usage:
    python benchmarks/benchmark_intx_methods.py --model_id meta-llama/Llama-3.1-8B --limit 100
"""

import argparse
import csv
import gc
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchao._models._eval import TransformerEvalWrapper
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    ModuleFqnToConfig,
    quantize_,
)
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.utils import benchmark_model, get_model_size_in_bytes


@dataclass
class Result:
    method: str
    size_gb: float
    comp_ratio: float
    quant_time_s: float
    fwd_ms: float
    tok_per_s: float
    peak_mem_gb: float
    accuracy: dict = None


def get_config(method: str):
    """Get quantization config for method."""
    configs = {
        "INT8-INT4": (
            IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(32)
            ),
        ),
        "INT8-INT4-HQQ": (
            IntxWeightOnlyConfig(
                weight_dtype=torch.int8,
                granularity=PerAxis(0),
                intx_choose_qparams_algorithm="hqq_scale_only",
            ),
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4,
                weight_granularity=PerGroup(32),
                intx_choose_qparams_algorithm="hqq_scale_only",
            ),
        ),
    }
    emb_cfg, lin_cfg = configs[method]
    return ModuleFqnToConfig({"_default": lin_cfg, "model.embed_tokens": emb_cfg})


def benchmark_method(
    model_id, method, baseline_size=None, tasks=None, limit=None, device="cuda"
):
    """Benchmark a single method."""
    print(f"\n{'=' * 60}\n{method}\n{'=' * 60}")

    # Ensure clean CUDA state before loading
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Load and quantize
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    quantize_(model, get_config(method), filter_fn=None)
    quant_time = time.time() - t0

    # Metrics
    size = get_model_size_in_bytes(model)
    size_gb = size / 1e9
    comp_ratio = baseline_size / size if baseline_size else 1.0

    # Benchmark forward pass
    inputs = torch.randint(0, 32000, (1, 512), device=device)
    for _ in range(3):  # warmup
        model(inputs)
    fwd_ms = benchmark_model(model, 10, (inputs,), device_type=device)

    # Benchmark generation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=100, do_sample=False)
    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0
    tok_per_s = (out.shape[1] - inputs.shape[1]) / gen_time

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Eval
    accuracy = None
    eval_results = TransformerEvalWrapper(
        model, tokenizer, 2048, device=device
    ).run_eval(tasks, limit)
    results_dict = eval_results.get("results", {})
    first_task = list(results_dict.keys())[0]
    accuracy = results_dict[first_task].get("exact_match,flexible-extract", None)

    result = Result(
        method, size_gb, comp_ratio, quant_time, fwd_ms, tok_per_s, peak_mem, accuracy
    )

    # Cleanup with proper synchronization
    del model
    del tokenizer
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print(
        f"Size: {size_gb:.3f} GB ({comp_ratio:.2f}x) | Quant: {quant_time:.1f}s | "
        f"Fwd: {fwd_ms:.2f}ms | Throughput: {tok_per_s:.1f} tok/s | Mem: {peak_mem:.2f} GB"
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--methods", nargs="+", default=["INT8-INT4", "INT8-INT4-HQQ"])
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"], help="lm_eval tasks")
    parser.add_argument("--limit", type=int, default=50, help="lm_eval limit per task")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"Benchmarking {args.model_id} on {args.methods}")

    # Baseline
    print("\nMeasuring baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    baseline = get_model_size_in_bytes(model)
    print(f"Baseline: {baseline / 1e9:.3f} GB")
    del model
    torch.cuda.empty_cache()

    # Benchmark
    results = []
    for method in args.methods:
        results.append(
            benchmark_method(
                args.model_id, method, baseline, args.tasks, args.limit, args.device
            )
        )
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(
        f"{'Method':<18} {'Size(GB)':<10} {'Comp':<8} {'Quant(s)':<10} {'Fwd(ms)':<10} {'Tok/s':<10} {'Mem(GB)':<10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r.method:<18} {r.size_gb:<10.3f} {r.comp_ratio:<8.2f} {r.quant_time_s:<10.1f} "
            f"{r.fwd_ms:<10.2f} {r.tok_per_s:<10.1f} {r.peak_mem_gb:<10.2f} {r.accuracy}"
        )

    # Save CSV
    with open(args.output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "size_gb",
                "compression",
                "quant_time_s",
                "fwd_ms",
                "tok_per_s",
                "peak_mem_gb",
                "accuracy",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.method,
                    f"{r.size_gb:.2f}",
                    f"{r.comp_ratio:.2f}",
                    f"{r.quant_time_s:.1f}",
                    f"{r.fwd_ms:.2f}",
                    f"{r.tok_per_s:.1f}",
                    f"{r.peak_mem_gb:.2f}",
                    f"{r.accuracy:.4f}",
                ]
            )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
