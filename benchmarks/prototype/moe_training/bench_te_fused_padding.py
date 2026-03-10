# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Microbenchmark for TE's fused_multi_row_padding / fused_multi_row_unpadding kernels.

Measures latency, achieved memory bandwidth (GB/s), and HBM bandwidth utilization %.
Compares against a naive PyTorch reference (torch.cat with F.pad per expert).

Usage:
    cd /path/to/ao
    PYTHONPATH=. python -m benchmarks.prototype.moe_training.bench_te_fused_padding
    PYTHONPATH=. python -m benchmarks.prototype.moe_training.bench_te_fused_padding --profile
"""

import argparse
import itertools
import logging
import random
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from triton.testing import do_bench

try:
    import transformer_engine  # noqa: F401 — loads native libs first
    import transformer_engine_torch as tex
except ImportError:
    raise ImportError(
        "TransformerEngine is required for this benchmark. "
        "Install from: https://github.com/NVIDIA/TransformerEngine"
    )

device = torch.device("cuda")

_MXFP8_BLOCK = 32


def _ceil_to_block(n: int) -> int:
    return (n + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK * _MXFP8_BLOCK


def _get_peak_hbm_bandwidth_gb_s() -> float:
    """Return peak HBM bandwidth in GB/s for the current GPU."""
    sm_major = torch.cuda.get_device_capability()[0]
    name = torch.cuda.get_device_name().lower()
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if "b200" in name or "gb200" in name:
        return 8000.0
    if "b300" in name or "gb300" in name:
        return 8000.0
    if sm_major >= 10:
        logging.warning(
            f"Unknown Blackwell GPU '{name}' (sm_{sm_major}) – "
            "assuming 8000 GB/s; override with --peak-bw if incorrect."
        )
        return 8000.0
    elif "h100" in name and "sxm" in name:
        return 3350.0
    elif "h100" in name:
        return 2000.0  # PCIe variant
    if "a100" in name:
        if mem_gb > 60:  # treat as 80 GB SKU
            if "sxm" in name:
                return 2039.0  # A100 80 GB SXM4
            else:
                return 2000.0  # A100 80 GB PCIe (rounded)
        else:
            return 1555.0
    logging.warning(
        f"Unrecognised GPU '{name}' – assuming 2000 GB/s; "
        "override with --peak-bw if incorrect."
    )
    return 2000.0



# ──────────────────────────────────────────────────────────────────────────────
# Padding implementations
# ──────────────────────────────────────────────────────────────────────────────


def te_pad(tensor: torch.Tensor, m_splits: List[int], padded_splits: List[int]) -> torch.Tensor:
    K = tensor.shape[-1]
    padded_total = sum(padded_splits)
    padded = torch.empty(padded_total, K, dtype=tensor.dtype, device=tensor.device)
    tex.fused_multi_row_padding(tensor.view(-1, K), padded, m_splits, padded_splits)
    return padded


def te_unpad(
    padded: torch.Tensor, m_splits: List[int], padded_splits: List[int]
) -> torch.Tensor:
    K = padded.shape[-1]
    total_tokens = sum(m_splits)
    out = torch.empty(total_tokens, K, dtype=padded.dtype, device=padded.device)
    tex.fused_multi_row_unpadding(padded, out, padded_splits, m_splits)
    return out


def naive_pad(tensor: torch.Tensor, m_splits: List[int], padded_splits: List[int]) -> torch.Tensor:
    """Reference: per-expert slice + F.pad + cat."""
    tensor = tensor.contiguous()
    K = tensor.shape[-1]
    chunks = tensor.split(m_splits)
    padded_chunks = []
    for chunk, ps in zip(chunks, padded_splits):
        pad_rows = ps - chunk.shape[0]
        if pad_rows > 0:
            padded_chunks.append(
                torch.nn.functional.pad(chunk, (0, 0, 0, pad_rows))
            )
        else:
            padded_chunks.append(chunk)
    return torch.cat(padded_chunks, dim=0)


def naive_unpad(
    padded: torch.Tensor, m_splits: List[int], padded_splits: List[int]
) -> torch.Tensor:
    """Reference: per-expert slice + narrow + cat."""
    chunks = padded.split(padded_splits)
    return torch.cat([c[:m] for c, m in zip(chunks, m_splits)], dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PadConfig:
    total_M: int
    K: int
    num_experts: int
    alignment: str  # "aligned" or "random"


@dataclass(frozen=True)
class PadResult:
    te_pad_us: float
    naive_pad_us: float
    pad_speedup: float
    te_unpad_us: float
    naive_unpad_us: float
    unpad_speedup: float
    pad_bw_gb_s: float
    unpad_bw_gb_s: float
    pad_bw_util_pct: float
    unpad_bw_util_pct: float


def generate_m_splits(total_M: int, num_experts: int, alignment: str) -> List[int]:
    """Generate per-expert token counts with controlled alignment.

    total_M controls the approximate problem scale (per-expert base = total_M // num_experts).
    alignment controls how token counts relate to _MXFP8_BLOCK (32):
      - "aligned": every expert's count is a multiple of 32 (best case, minimal padding)
      - "random":  each expert gets a random count around the base (worst case)
    """
    base = total_M // max(num_experts, 1)
    random.seed(42)

    if alignment == "aligned":
        return [_ceil_to_block(base)] * num_experts

    splits = []
    for _ in range(num_experts):
        m = max(base + random.randint(-base // 4, base // 4), 1)
        if m % _MXFP8_BLOCK == 0:
            m -= 1
        splits.append(m)
    return splits


def get_configs() -> List[PadConfig]:
    M_list = [16384, 65536, 128000]
    K_list = [1536, 2048, 5120, 7168]
    expert_list = [1, 4, 8, 16]
    alignment_list = ["random", "aligned"]

    configs = []
    for M, K, G, alignment in itertools.product(M_list, K_list, expert_list, alignment_list):
        configs.append(PadConfig(total_M=M, K=K, num_experts=G, alignment=alignment))
    return configs


def bench_us(fn, *args, **kwargs) -> float:
    return do_bench(lambda: fn(*args, **kwargs), return_mode="median") * 1e3


def run_experiment(config: PadConfig, peak_bw: float) -> PadResult:
    m_splits = generate_m_splits(config.total_M, config.num_experts, config.alignment)
    padded_splits = [_ceil_to_block(m) for m in m_splits]

    total_tokens = sum(m_splits)
    padded_total = sum(padded_splits)

    tensor = torch.randn(total_tokens, config.K, dtype=torch.bfloat16, device=device)

    element_bytes = tensor.element_size()

    # ── Pad benchmarks ──
    te_pad_us = bench_us(te_pad, tensor, m_splits, padded_splits)
    naive_pad_us = bench_us(naive_pad, tensor, m_splits, padded_splits)

    # ── Unpad benchmarks ──
    padded_tensor = torch.randn(padded_total, config.K, dtype=torch.bfloat16, device=device)

    te_unpad_us = bench_us(te_unpad, padded_tensor, m_splits, padded_splits)
    naive_unpad_us = bench_us(naive_unpad, padded_tensor, m_splits, padded_splits)

    # ── Bandwidth calculation ──
    # Pad: read total_tokens*K, write padded_total*K
    pad_bytes = (total_tokens + padded_total) * config.K * element_bytes
    # Unpad: read padded_total*K, write total_tokens*K
    unpad_bytes = (padded_total + total_tokens) * config.K * element_bytes

    pad_bw_gb_s = (pad_bytes / 1e9) / (te_pad_us / 1e6) if te_pad_us > 0 else 0.0
    unpad_bw_gb_s = (unpad_bytes / 1e9) / (te_unpad_us / 1e6) if te_unpad_us > 0 else 0.0

    pad_bw_util = (pad_bw_gb_s / peak_bw) * 100 if peak_bw > 0 else 0.0
    unpad_bw_util = (unpad_bw_gb_s / peak_bw) * 100 if peak_bw > 0 else 0.0

    pad_speedup = naive_pad_us / te_pad_us if te_pad_us > 0 else float("inf")
    unpad_speedup = naive_unpad_us / te_unpad_us if te_unpad_us > 0 else float("inf")

    return PadResult(
        te_pad_us=round(te_pad_us, 2),
        naive_pad_us=round(naive_pad_us, 2),
        pad_speedup=round(pad_speedup, 2),
        te_unpad_us=round(te_unpad_us, 2),
        naive_unpad_us=round(naive_unpad_us, 2),
        unpad_speedup=round(unpad_speedup, 2),
        pad_bw_gb_s=round(pad_bw_gb_s, 1),
        unpad_bw_gb_s=round(unpad_bw_gb_s, 1),
        pad_bw_util_pct=round(pad_bw_util, 1),
        unpad_bw_util_pct=round(unpad_bw_util, 1),
    )


def print_results(configs: List[PadConfig], results: List[PadResult], peak_bw: float):
    gpu_name = torch.cuda.get_device_name()
    print(f"\nGPU: {gpu_name}  |  Peak HBM BW: {peak_bw:.0f} GB/s\n")

    headers = [
        "M",
        "K",
        "experts",
        "align",
        "te_pad_us",
        "naive_pad_us",
        "pad_spdup",
        "pad_BW(GB/s)",
        "pad_BW%",
        "te_unpad_us",
        "naive_unpad_us",
        "unpad_spdup",
        "unpad_BW(GB/s)",
        "unpad_BW%",
    ]
    rows = []
    for cfg, res in zip(configs, results):
        rows.append([
            cfg.total_M,
            cfg.K,
            cfg.num_experts,
            cfg.alignment,
            res.te_pad_us,
            res.naive_pad_us,
            f"{res.pad_speedup}x",
            res.pad_bw_gb_s,
            f"{res.pad_bw_util_pct}%",
            res.te_unpad_us,
            res.naive_unpad_us,
            f"{res.unpad_speedup}x",
            res.unpad_bw_gb_s,
            f"{res.unpad_bw_util_pct}%",
        ])
    print(tabulate(rows, headers=headers, floatfmt=".2f"))


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark for TE fused padding ops")
    parser.add_argument("--profile", action="store_true", help="Save a chrome trace")
    parser.add_argument(
        "--peak-bw", type=float, default=0,
        help="Override peak HBM bandwidth in GB/s (auto-detected if 0)",
    )
    args = parser.parse_args()

    peak_bw = args.peak_bw if args.peak_bw > 0 else _get_peak_hbm_bandwidth_gb_s()

    torch.manual_seed(42)
    configs = get_configs()
    results = []

    for cfg in configs:
        results.append(run_experiment(cfg, peak_bw))

    print_results(configs, results, peak_bw)

    if args.profile:
        # Profile the largest unaligned case for detailed trace
        cfg = PadConfig(total_M=128000, K=7168, num_experts=8, alignment="random")
        m_splits = generate_m_splits(cfg.total_M, cfg.num_experts, cfg.alignment)
        padded_splits = [_ceil_to_block(m) for m in m_splits]
        tensor = torch.randn(sum(m_splits), cfg.K, dtype=torch.bfloat16, device=device)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                te_pad(tensor, m_splits, padded_splits)
                prof.step()

        prof.export_chrome_trace("te_fused_padding_profile.json")
        print("\nSaved: te_fused_padding_profile.json")


if __name__ == "__main__":
    main()
