# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Microbenchmark for silu_mul_fw and silu_mul_bw kernels.

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.ep.syncless.silu_mul_kernel import (
    silu_mul_bw,
    silu_mul_fw,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    hidden_dim: int


@dataclass(frozen=True)
class ExperimentResult:
    # Forward
    fw_us: float
    fw_gbps: float
    # Backward
    bw_us: float
    bw_gbps: float
    # PyTorch eager baseline (forward only)
    eager_fw_us: float
    eager_fw_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    configs = []
    for num_tokens in [1 * 8192, 2 * 8192, 4 * 8192, 8 * 8192]:
        for hidden_dim in [2048, 4096]:
            configs.append(
                ExperimentConfig(num_tokens=num_tokens, hidden_dim=hidden_dim)
            )
    return configs


def _eager_silu_mul(h13: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """PyTorch eager SwiGLU: silu(h1) * h3."""
    h1 = h13[:, :hidden_dim]
    h3 = h13[:, hidden_dim:]
    return F.silu(h1) * h3


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M = config.num_tokens
    hidden_dim = config.hidden_dim

    # Simulate the saved-activations buffer layout: (M, 2*hidden_dim) bf16
    h13_buffer = torch.randn(M, 2 * hidden_dim, dtype=torch.bfloat16, device=device)

    # GPU-resident offset and num_tokens (zero-sync)
    offset = torch.tensor(0, dtype=torch.int64, device=device)
    num_tokens = torch.tensor(M, dtype=torch.int64, device=device)
    sym_mem_buffer_rows = M

    # Pre-allocate output buffers
    fw_output = torch.empty(M, hidden_dim, dtype=torch.bfloat16, device=device)
    h_out = torch.empty(M, hidden_dim, dtype=torch.bfloat16, device=device)
    grad_h13_out = torch.empty(M, 2 * hidden_dim, dtype=torch.bfloat16, device=device)

    # ---- Forward benchmark ----
    # Warmup
    silu_mul_fw(h13_buffer, offset, num_tokens, sym_mem_buffer_rows, fw_output)

    fw_us = benchmark_cuda_function_in_microseconds(
        silu_mul_fw,
        h13_buffer,
        offset,
        num_tokens,
        sym_mem_buffer_rows,
        fw_output,
    )

    # Forward mem bw: reads h13 (M * 2*hidden_dim * 2B), writes h (M * hidden_dim * 2B)
    fw_read_bytes = M * 2 * hidden_dim * 2  # bf16
    fw_write_bytes = M * hidden_dim * 2
    fw_total_bytes = fw_read_bytes + fw_write_bytes
    fw_gbps = (fw_total_bytes / 1e9) / (fw_us / 1e6)

    # ---- Backward benchmark ----
    grad_h = torch.randn(M, hidden_dim, dtype=torch.bfloat16, device=device)

    # Warmup
    silu_mul_bw(h13_buffer, grad_h, offset, num_tokens, h_out, grad_h13_out)

    bw_us = benchmark_cuda_function_in_microseconds(
        silu_mul_bw,
        h13_buffer,
        grad_h,
        offset,
        num_tokens,
        h_out,
        grad_h13_out,
    )

    # Backward mem bw:
    #   reads: h13 (M * 2*hidden_dim * 2B) + grad_h (M * hidden_dim * 2B)
    #   writes: h_out (M * hidden_dim * 2B) + grad_h13 (M * 2*hidden_dim * 2B)
    bw_read_bytes = M * 2 * hidden_dim * 2 + M * hidden_dim * 2
    bw_write_bytes = M * hidden_dim * 2 + M * 2 * hidden_dim * 2
    bw_total_bytes = bw_read_bytes + bw_write_bytes
    bw_gbps = (bw_total_bytes / 1e9) / (bw_us / 1e6)

    # ---- Eager baseline (forward only) ----
    _eager_silu_mul(h13_buffer, hidden_dim)

    eager_fw_us = benchmark_cuda_function_in_microseconds(
        _eager_silu_mul,
        h13_buffer,
        hidden_dim,
    )
    eager_fw_gbps = (fw_total_bytes / 1e9) / (eager_fw_us / 1e6)

    return ExperimentResult(
        fw_us=fw_us,
        fw_gbps=fw_gbps,
        bw_us=bw_us,
        bw_gbps=bw_gbps,
        eager_fw_us=eager_fw_us,
        eager_fw_gbps=eager_fw_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "shape (M, hdim)",
        "fw_us",
        "eager_fw_us",
        "fw_speedup",
        "bw_us",
        "fw_gbps",
        "bw_gbps",
        "eager_fw_gbps",
    ]
    rows = []
    for exp in experiments:
        c, r = exp.config, exp.result
        fw_speedup = round(r.eager_fw_us / r.fw_us, 3) if r.fw_us > 0 else 0
        rows.append(
            [
                f"({c.num_tokens}, {c.hidden_dim})",
                round(r.fw_us, 3),
                round(r.eager_fw_us, 3),
                f"{fw_speedup}x",
                round(r.bw_us, 3),
                round(r.fw_gbps, 1),
                round(r.bw_gbps, 1),
                round(r.eager_fw_gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print_results(results)


if __name__ == "__main__":
    main()
