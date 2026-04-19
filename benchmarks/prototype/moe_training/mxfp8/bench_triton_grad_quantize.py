# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark: fused ``triton_mxfp8_quantize_dim0_dim1`` vs. today's 4-kernel
backward-pass sequence (dim0 quantize + dim0 rearrange + dim1 quantize + dim1
rearrange).

Target (review feedback on ao#4293): close the gap to the B200 bf16->fp8
memcpy ceiling at the DeepSeek-V3-like backward-pass shape
``(num_groups=4, M_per_group=4096, N=2048)`` -> ``(total_M=16384, N=2048)``
and the adjacent sweep. Measured B200 bf16 memcpy is ~5 TB/s on this rig
(the ``_memcpy_bf16_bw_gbps`` helper below measures it live), so
``% memcpy BW`` is the relevant metric.

In this problem the single-pass kernel is bottlenecked by the ``(N, M)``
scattered-stride fp8 store on the dim1 output: every lane of every warp
stores a 1-byte fp8 value at stride-``M`` into HBM, which thrashes the
row buffer of HBM3e on B200. On the flagged shape the existing
``triton_to_mxfp8_dim1`` reference kernel already peaks around 1 TB/s
for the same reason; fusing dim0 + dim1 + blocked rearrange into one
kernel lets us overlap the coalesced (M,N) dim0 store with the scattered
(N,M) dim1 store (disjoint output tensors so the memory controller can
issue both concurrently) and still win 1.5-8x over the 4-kernel baseline.
"""

import argparse
from dataclasses import dataclass
from typing import List

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_2d_M_groups,
    triton_mxfp8_quantize_dim0_dim1,
)
from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
)
from torchao.utils import ceil_div

device = torch.device("cuda")

torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    M: int
    N: int


@dataclass(frozen=True)
class ExperimentResult:
    four_kernel_us: float
    fused_us: float
    four_kernel_gbps: float
    fused_gbps: float
    speedup: float
    memcpy_bw_pct: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # Daniel's target shape first, then a sweep covering realistic backward
    # pass sizes: (num_groups, M_per_group, N) in {(4, 4096, 2048), (4, 4096,
    # 5120), (8, 4096, 2048), (4, 8192, 2048), ...} as flat (total_M, N).
    pairs = [
        # (total_M, N)  - Daniel's primary landing target
        (16384, 2048),
        # DeepSeek-V3-like sweeps
        (4096, 2048),
        (8192, 2048),
        (32768, 2048),
        (16384, 5120),
        (16384, 7168),
        (8192, 5120),
        (8192, 7168),
        (32768, 5120),
    ]
    return [ExperimentConfig(M=m, N=n) for m, n in pairs]


def _four_kernel_reference(x: torch.Tensor):
    """Today's backward-pass path: dim0 quant + dim0 rearrange + dim1 quant +
    dim1 rearrange. Runs the same four Triton kernels that the production
    ``_MXFP8GroupedMM.backward`` invokes today (minus the cross-group
    bookkeeping, which is the same constant overhead on either side)."""
    M, N = x.shape
    qdata0, scales0_rm = triton_to_mxfp8_dim0(
        x, inner_block_size=32, scaling_mode="rceil"
    )
    qdata1_t, scales1_rm = triton_to_mxfp8_dim1(
        x, inner_block_size=32, scaling_mode="rceil"
    )
    # We pass a single-group offset so the rearrange kernel still runs its
    # standard launch, matching the production pad-and-blocked-scale path.
    one_group_m = torch.tensor([M], dtype=torch.int32, device=x.device)
    one_group_n = torch.tensor([N], dtype=torch.int32, device=x.device)
    scales0_b = triton_mx_block_rearrange_2d_M_groups(scales0_rm, one_group_m)
    scales1_b = triton_mx_block_rearrange_2d_M_groups(scales1_rm, one_group_n)
    return qdata0, qdata1_t, scales0_b, scales1_b


def _bytes_touched(M: int, N: int) -> int:
    """Bytes of HBM that MUST flow for this problem: one bf16 read of the
    input, two fp8 writes (row-major + transposed), and two e8m0 scale
    writes in blocked layout. This is the memcpy lower bound."""
    scale_cols_n = ceil_div(N // 32, 4) * 4
    scale_cols_m = ceil_div(M // 32, 4) * 4
    return (
        M * N * 2  # bf16 read
        + M * N * 1  # fp8 dim0 write
        + N * M * 1  # fp8 dim1_t write
        + M * scale_cols_n * 1  # e8m0 dim0 scales (blocked)
        + N * scale_cols_m * 1  # e8m0 dim1 scales (blocked)
    )


def _memcpy_bf16_bw_gbps(M: int, N: int) -> float:
    """Rough B200 bf16 memcpy bandwidth estimate (read + write of an (M, N)
    bf16 tensor, sustained). Measured in-process so the number reflects the
    current machine, not a datasheet constant."""
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    def memcpy():
        return x.clone()

    for _ in range(5):
        memcpy()
    us = benchmark_cuda_function_in_microseconds(memcpy)
    # clone reads and writes the full tensor -> 2 * M * N * 2 bytes.
    bytes_touched = 2 * M * N * 2
    return (bytes_touched / 1e9) / (us / 1e6)


def run_experiment(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentResult:
    M, N = config.M, config.N

    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    def run_4kernel():
        return _four_kernel_reference(x)

    def run_fused():
        return triton_mxfp8_quantize_dim0_dim1(x, scaling_mode="rceil")

    for _ in range(5):
        run_4kernel()
        run_fused()

    four_kernel_us = benchmark_cuda_function_in_microseconds(run_4kernel)
    fused_us = benchmark_cuda_function_in_microseconds(run_fused)

    bytes_total = _bytes_touched(M, N)
    four_kernel_gbps = (bytes_total / 1e9) / (four_kernel_us / 1e6)
    fused_gbps = (bytes_total / 1e9) / (fused_us / 1e6)

    # Express fused BW as a % of the bf16 memcpy ceiling at the same input
    # size. This is the "90%+ memcpy BW" metric Daniel asked for.
    memcpy_gbps = _memcpy_bf16_bw_gbps(M, N)
    memcpy_bw_pct = 100.0 * fused_gbps / memcpy_gbps if memcpy_gbps > 0 else 0.0

    return ExperimentResult(
        four_kernel_us=four_kernel_us,
        fused_us=fused_us,
        four_kernel_gbps=four_kernel_gbps,
        fused_gbps=fused_gbps,
        speedup=four_kernel_us / fused_us,
        memcpy_bw_pct=memcpy_bw_pct,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N",
        "4k_us",
        "fused_us",
        "4k_GB/s",
        "fused_GB/s",
        "speedup",
        "% memcpy BW",
    ]
    rows = []
    for exp in experiments:
        rows.append(
            [
                exp.config.M,
                exp.config.N,
                f"{exp.result.four_kernel_us:.1f}",
                f"{exp.result.fused_us:.1f}",
                f"{exp.result.four_kernel_gbps:.0f}",
                f"{exp.result.fused_gbps:.0f}",
                f"{exp.result.speedup:.2f}x",
                f"{exp.result.memcpy_bw_pct:.1f}%",
            ]
        )
    print(tabulate(rows, headers=headers))


def main(args: argparse.Namespace):
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config, args)
        results.append(Experiment(config=config, result=result))
    print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling with PyTorch profiler"
    )
    args = parser.parse_args()
    main(args)
