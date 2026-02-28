# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark any two attention backends against each other for a single layer.

Sweeps over sequence lengths from 1024 to 131072, measuring runtime and
SQNR (Signal-to-Quantization-Noise Ratio) for each configuration.

Available backends:
    fa2      - BF16 SDPA with FlashAttention 2 (PyTorch default)
    fa3      - BF16 SDPA with FlashAttention 3
    fa3_fp8  - FP8 SDPA with FlashAttention 3 (includes quantization kernels)

Usage:
    # Default: FA2 vs FA3+FP8
    python benchmarks/prototype/attention/benchmark_sdpa.py

    # FA3 bf16 vs FA3 fp8
    python benchmarks/prototype/attention/benchmark_sdpa.py --baseline fa3 --test fa3_fp8

    # With causal masking
    python benchmarks/prototype/attention/benchmark_sdpa.py --baseline fa3 --test fa3_fp8 --causal
"""

import argparse
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention import (
    SDPBackend,
    activate_flash_attention_impl,
    restore_flash_attention_impl,
    sdpa_kernel,
)

from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

BACKENDS = ["fa2", "fa3", "fa3_fp8"]

BACKEND_LABELS = {
    "fa2": "FA2 BF16",
    "fa3": "FA3 BF16",
    "fa3_fp8": "FA3 FP8",
}


@contextmanager
def _activate_backend(backend: str):
    """Context manager that activates the appropriate flash attention impl."""
    if backend in ("fa3", "fa3_fp8"):
        activate_flash_attention_impl("FA3")
    else:
        # fa2 is the default, no activation needed
        pass
    try:
        yield
    finally:
        if backend in ("fa3", "fa3_fp8"):
            restore_flash_attention_impl()


def _run_attention(backend: str, q, k, v, is_causal: bool):
    """Run a single attention call for the given backend."""
    if backend == "fa3_fp8":
        return fp8_fa3_sdpa(q, k, v, is_causal=is_causal)
    else:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def compute_sqnr(reference: torch.Tensor, approximate: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = reference.float().pow(2).mean()
    noise_power = (reference.float() - approximate.float()).pow(2).mean()
    if noise_power == 0:
        return float("inf")
    return (10 * torch.log10(signal_power / noise_power)).item()


def benchmark_fn(fn, num_warmup, num_iters):
    """Benchmark a function, returning median runtime in ms."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[num_iters // 2]  # median


@torch.inference_mode()
def run_benchmark(
    baseline: str = "fa2",
    test: str = "fa3_fp8",
    is_causal: bool = False,
    num_warmup: int = 5,
    num_iters: int = 20,
):
    B = 1
    H = 32
    D = 128
    SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    device = "cuda"
    dtype = torch.bfloat16

    baseline_label = BACKEND_LABELS[baseline]
    test_label = BACKEND_LABELS[test]

    print("=" * 90)
    print(f"Benchmark: {baseline_label} vs {test_label} — Single Attention Layer")
    print(f"  Shape: (B={B}, H={H}, S=variable, D={D})")
    print(f"  Causal: {is_causal}")
    print(f"  Warmup: {num_warmup}, Iters: {num_iters}")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print("=" * 90)

    col_baseline = f"{baseline_label} (ms)"
    col_test = f"{test_label} (ms)"
    col_w = max(len(col_baseline), len(col_test), 12)

    header = (
        f"{'SeqLen':>8} | "
        f"{col_baseline:>{col_w}} | "
        f"{col_test:>{col_w}} | "
        f"{'Speedup':>8} | "
        f"{'SQNR (dB)':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for S in SEQ_LENGTHS:
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # --- Baseline ---
        with _activate_backend(baseline):
            baseline_fn = partial(_run_attention, baseline, q, k, v, is_causal)
            baseline_time = benchmark_fn(baseline_fn, num_warmup, num_iters)
            ref_out = _run_attention(baseline, q, k, v, is_causal)

        # --- Test ---
        with _activate_backend(test):
            test_fn = partial(_run_attention, test, q, k, v, is_causal)
            test_time = benchmark_fn(test_fn, num_warmup, num_iters)
            test_out = _run_attention(test, q, k, v, is_causal)

        sqnr = compute_sqnr(ref_out, test_out)
        speedup = baseline_time / test_time

        print(
            f"{S:>8} | "
            f"{baseline_time:>{col_w}.3f} | "
            f"{test_time:>{col_w}.3f} | "
            f"{speedup:>7.2f}x | "
            f"{sqnr:>10.2f}"
        )

        results.append(
            {
                "seq_len": S,
                "baseline_ms": baseline_time,
                "test_ms": test_time,
                "speedup": speedup,
                "sqnr_db": sqnr,
            }
        )

        del q, k, v, ref_out, test_out
        torch.cuda.empty_cache()

    print("-" * len(header))
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark any two attention backends for a single layer"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="fa2",
        choices=BACKENDS,
        help="Baseline attention backend (default: fa2)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="fa3_fp8",
        choices=BACKENDS,
        help="Test attention backend to compare against baseline (default: fa3_fp8)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal attention masking",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=20,
        help="Number of timed iterations",
    )
    args = parser.parse_args()

    run_benchmark(
        baseline=args.baseline,
        test=args.test,
        is_causal=args.causal,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )


if __name__ == "__main__":
    main()
