# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# Microbenchmark for fused silu_mul_fw_mxfp8 kernel vs. the 2-step approach
# (silu_mul_fw → triton_to_mxfp8_dim0 + scale rearrangement).

from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.ep.syncless.mxfp8_kernels import (
    triton_mx_block_rearrange_input_sym_mem_buffer,
)
from torchao.prototype.moe_training.ep.syncless.silu_mul_kernel import (
    silu_mul_fw,
    silu_mul_fw_mxfp8,
)
from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_dim0,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

SCALE_BLOCK_SIZE = 32


@dataclass(frozen=True)
class ExperimentConfig:
    num_tokens: int
    hidden_dim: int


@dataclass(frozen=True)
class ExperimentResult:
    # Fused kernel
    fused_us: float
    fused_gbps: float
    # 2-step: silu_mul_fw + triton_to_mxfp8_dim0 + scale rearrange
    two_step_us: float
    two_step_gbps: float


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


def _two_step_silu_quant(
    h13_buffer,
    offset,
    num_tokens,
    sym_mem_buffer_rows,
):
    """2-step baseline: silu_mul_fw (BF16) → triton_to_mxfp8_dim0 + scale rearrange."""
    h = silu_mul_fw(h13_buffer, offset, num_tokens, sym_mem_buffer_rows)
    h_e4m3, h_scales = triton_to_mxfp8_dim0(
        h, inner_block_size=SCALE_BLOCK_SIZE, scaling_mode="rceil"
    )
    h_scales_blocked = triton_mx_block_rearrange_input_sym_mem_buffer(
        h_scales, num_tokens
    )
    return h_e4m3, h_scales_blocked


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M = config.num_tokens
    hidden_dim = config.hidden_dim

    # Simulate the saved-activations buffer layout: (M, 2*hidden_dim) bf16
    h13_buffer = torch.randn(M, 2 * hidden_dim, dtype=torch.bfloat16, device=device)

    # GPU-resident offset and num_tokens (zero-sync)
    offset = torch.tensor(0, dtype=torch.int64, device=device)
    num_tokens = torch.tensor(M, dtype=torch.int64, device=device)
    sym_mem_buffer_rows = M

    # ---- Fused kernel benchmark ----
    # Warmup
    silu_mul_fw_mxfp8(h13_buffer, offset, num_tokens, sym_mem_buffer_rows)

    fused_us = benchmark_cuda_function_in_microseconds(
        silu_mul_fw_mxfp8,
        h13_buffer,
        offset,
        num_tokens,
        sym_mem_buffer_rows,
    )

    # ---- 2-step baseline benchmark ----
    # Warmup
    _two_step_silu_quant(h13_buffer, offset, num_tokens, sym_mem_buffer_rows)

    two_step_us = benchmark_cuda_function_in_microseconds(
        _two_step_silu_quant,
        h13_buffer,
        offset,
        num_tokens,
        sym_mem_buffer_rows,
    )

    # ---- Memory bandwidth calculation ----
    # Fused kernel:
    #   reads: h13 (M * 2*hidden_dim * 2B bf16)
    #   writes: h_e4m3 (M * hidden_dim * 1B fp8) + scales (padded, ~M * hidden_dim/32 * 1B)
    read_bytes = M * 2 * hidden_dim * 2
    scale_cols = hidden_dim // SCALE_BLOCK_SIZE
    write_data_bytes = M * hidden_dim * 1
    write_scale_bytes = M * scale_cols * 1
    total_bytes = read_bytes + write_data_bytes + write_scale_bytes
    fused_gbps = (total_bytes / 1e9) / (fused_us / 1e6)

    # 2-step has same logical I/O plus the BF16 intermediate
    # (written by silu_mul_fw, read by triton_to_mxfp8_dim0)
    intermediate_bytes = M * hidden_dim * 2 * 2  # write + read
    two_step_total_bytes = total_bytes + intermediate_bytes
    two_step_gbps = (two_step_total_bytes / 1e9) / (two_step_us / 1e6)

    return ExperimentResult(
        fused_us=fused_us,
        fused_gbps=fused_gbps,
        two_step_us=two_step_us,
        two_step_gbps=two_step_gbps,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "shape (M, hdim)",
        "fused_us",
        "2step_us",
        "fused_gbps",
        "2step_gbps",
        "fused_speedup",
    ]
    rows = []
    for exp in experiments:
        c, r = exp.config, exp.result
        speedup = round(r.two_step_us / r.fused_us, 3) if r.fused_us > 0 else 0
        rows.append(
            [
                f"({c.num_tokens}, {c.hidden_dim})",
                round(r.fused_us, 3),
                round(r.two_step_us, 3),
                round(r.fused_gbps, 1),
                round(r.two_step_gbps, 1),
                f"{speedup}x",
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
