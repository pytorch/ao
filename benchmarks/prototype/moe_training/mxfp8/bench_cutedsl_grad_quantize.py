# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL ceiling benchmark for MXFP8 grad_out backward quantization.

This is the first benchmark for the "can we get close to memcpy?" path from
ao#4293.  It keeps the same logical output contract as the requested backward
kernel:

    * read bf16 grad_out shaped (M, N)
    * emit row-major MXFP8 grad_out with blocked e8m0 scales
    * emit row-major MXFP8 grad_out.T with blocked e8m0 scales

The CuTeDSL variants below intentionally use the existing 1x32 and 32x1
single-output kernels instead of a new fused kernel.  That means they read the
bf16 input twice, but they give us a useful ceiling for the low-level CuTe path
before investing in a same-contract, single-read kernel.
"""

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_grad_quantize import (
    cutedsl_mxfp8_quantize_dim0_dim1_single_read,
    cutedsl_mxfp8_quantize_dim0_dim1_streams,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.triton_grad_quantize import (
    triton_mxfp8_quantize_dim0_dim1,
)
from torchao.utils import ceil_div

device = torch.device("cuda")


@dataclass(frozen=True)
class ExperimentConfig:
    M: int
    N: int
    scaling_mode: str


@dataclass(frozen=True)
class ExperimentResult:
    memcpy_gbps: float
    cutedsl_dim0_us: float
    cutedsl_dim0_gbps: float
    cutedsl_dim0_memcpy_pct: float
    cutedsl_dim1_us: float
    cutedsl_dim1_gbps: float
    cutedsl_dim1_memcpy_pct: float
    triton_us: float
    triton_gbps: float
    triton_memcpy_pct: float
    cutedsl_serial_us: float
    cutedsl_serial_gbps: float
    cutedsl_serial_memcpy_pct: float
    cutedsl_streams_us: float
    cutedsl_streams_gbps: float
    cutedsl_streams_memcpy_pct: float
    cutedsl_single_read_us: float
    cutedsl_single_read_gbps: float
    cutedsl_single_read_memcpy_pct: float


def get_configs(quick: bool) -> list[ExperimentConfig]:
    shapes = (
        [(16384, 2048)]
        if quick
        else [
            # Primary review target: (num_groups=4, M_per_group=4096, N=2048).
            (16384, 2048),
            # Neighboring backward-pass sizes.
            (4096, 2048),
            (8192, 2048),
            (32768, 2048),
            (16384, 5120),
            (16384, 7168),
        ]
    )
    return [ExperimentConfig(M=m, N=n, scaling_mode="rceil") for m, n in shapes]


def _requested_contract_bytes(M: int, N: int) -> int:
    """Minimum bytes for the requested single-read dual-output contract."""
    scale_cols_n = ceil_div(N // 32, 4) * 4
    scale_cols_m = ceil_div(M // 32, 4) * 4
    return M * N * 2 + M * N + N * M + M * scale_cols_n + N * scale_cols_m


def _dim0_bytes(M: int, N: int) -> int:
    scale_cols_n = ceil_div(N // 32, 4) * 4
    return M * N * 2 + M * N + M * scale_cols_n


def _dim1_bytes(M: int, N: int) -> int:
    scale_cols_m = ceil_div(M // 32, 4) * 4
    return M * N * 2 + M * N + N * scale_cols_m


def _gbps(bytes_touched: int, time_us: float) -> float:
    return (bytes_touched / 1e9) / (time_us / 1e6)


def _bench_us(fn: Callable[[], object]) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return benchmark_cuda_function_in_microseconds(fn)


def _memcpy_bf16_gbps(x: torch.Tensor) -> float:
    def memcpy():
        return x.clone()

    us = _bench_us(memcpy)
    return _gbps(x.numel() * 2 * 2, us)


def _cutedsl_serial(x: torch.Tensor, scaling_mode: str):
    q_dim0, scales_dim0 = _cutedsl_dim0(x, scaling_mode)
    q_dim1_t, scales_dim1 = _cutedsl_dim1(x, scaling_mode)
    return q_dim0, q_dim1_t, scales_dim0, scales_dim1


def _cutedsl_dim0(x: torch.Tensor, scaling_mode: str):
    return mxfp8_quantize_2d_1x32_cutedsl(
        x,
        scaling_mode=scaling_mode,
        stage_count=2,
    )


def _cutedsl_dim1(x: torch.Tensor, scaling_mode: str):
    q_dim1_col_major, scales_dim1 = mxfp8_quantize_2d_32x1_cutedsl(
        x,
        scaling_mode=scaling_mode,
        stage_count=2,
        blocked_scale_output=True,
    )
    return q_dim1_col_major.t(), scales_dim1


def _cutedsl_two_streams(x: torch.Tensor, scaling_mode: str):
    return cutedsl_mxfp8_quantize_dim0_dim1_streams(x, scaling_mode)


def _cutedsl_single_read(x: torch.Tensor, scaling_mode: str):
    return cutedsl_mxfp8_quantize_dim0_dim1_single_read(x, scaling_mode)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    torch.manual_seed(123)
    x = torch.randn(config.M, config.N, dtype=torch.bfloat16, device=device)
    requested_bytes = _requested_contract_bytes(config.M, config.N)
    dim0_bytes = _dim0_bytes(config.M, config.N)
    dim1_bytes = _dim1_bytes(config.M, config.N)
    memcpy_gbps = _memcpy_bf16_gbps(x)

    def run_triton():
        return triton_mxfp8_quantize_dim0_dim1(x, scaling_mode=config.scaling_mode)

    def run_cutedsl_dim0():
        return _cutedsl_dim0(x, config.scaling_mode)

    def run_cutedsl_dim1():
        return _cutedsl_dim1(x, config.scaling_mode)

    def run_cutedsl_serial():
        return _cutedsl_serial(x, config.scaling_mode)

    def run_cutedsl_streams():
        return _cutedsl_two_streams(x, config.scaling_mode)

    def run_cutedsl_single_read():
        return _cutedsl_single_read(x, config.scaling_mode)

    cutedsl_dim0_us = _bench_us(run_cutedsl_dim0)
    cutedsl_dim1_us = _bench_us(run_cutedsl_dim1)
    triton_us = _bench_us(run_triton)
    cutedsl_serial_us = _bench_us(run_cutedsl_serial)
    cutedsl_streams_us = _bench_us(run_cutedsl_streams)
    cutedsl_single_read_us = _bench_us(run_cutedsl_single_read)

    cutedsl_dim0_gbps = _gbps(dim0_bytes, cutedsl_dim0_us)
    cutedsl_dim1_gbps = _gbps(dim1_bytes, cutedsl_dim1_us)
    triton_gbps = _gbps(requested_bytes, triton_us)
    cutedsl_serial_gbps = _gbps(requested_bytes, cutedsl_serial_us)
    cutedsl_streams_gbps = _gbps(requested_bytes, cutedsl_streams_us)
    cutedsl_single_read_gbps = _gbps(requested_bytes, cutedsl_single_read_us)

    return ExperimentResult(
        memcpy_gbps=memcpy_gbps,
        cutedsl_dim0_us=cutedsl_dim0_us,
        cutedsl_dim0_gbps=cutedsl_dim0_gbps,
        cutedsl_dim0_memcpy_pct=100.0 * cutedsl_dim0_gbps / memcpy_gbps,
        cutedsl_dim1_us=cutedsl_dim1_us,
        cutedsl_dim1_gbps=cutedsl_dim1_gbps,
        cutedsl_dim1_memcpy_pct=100.0 * cutedsl_dim1_gbps / memcpy_gbps,
        triton_us=triton_us,
        triton_gbps=triton_gbps,
        triton_memcpy_pct=100.0 * triton_gbps / memcpy_gbps,
        cutedsl_serial_us=cutedsl_serial_us,
        cutedsl_serial_gbps=cutedsl_serial_gbps,
        cutedsl_serial_memcpy_pct=100.0 * cutedsl_serial_gbps / memcpy_gbps,
        cutedsl_streams_us=cutedsl_streams_us,
        cutedsl_streams_gbps=cutedsl_streams_gbps,
        cutedsl_streams_memcpy_pct=100.0 * cutedsl_streams_gbps / memcpy_gbps,
        cutedsl_single_read_us=cutedsl_single_read_us,
        cutedsl_single_read_gbps=cutedsl_single_read_gbps,
        cutedsl_single_read_memcpy_pct=100.0 * cutedsl_single_read_gbps / memcpy_gbps,
    )


def print_results(results: list[tuple[ExperimentConfig, ExperimentResult]]) -> None:
    rows = []
    for config, result in results:
        rows.append(
            [
                config.M,
                config.N,
                f"{result.memcpy_gbps:.0f}",
                f"{result.cutedsl_dim0_us:.1f}",
                f"{result.cutedsl_dim0_memcpy_pct:.1f}%",
                f"{result.cutedsl_dim1_us:.1f}",
                f"{result.cutedsl_dim1_memcpy_pct:.1f}%",
                f"{result.triton_us:.1f}",
                f"{result.triton_memcpy_pct:.1f}%",
                f"{result.cutedsl_serial_us:.1f}",
                f"{result.cutedsl_serial_memcpy_pct:.1f}%",
                f"{result.cutedsl_streams_us:.1f}",
                f"{result.cutedsl_streams_memcpy_pct:.1f}%",
                f"{result.cutedsl_single_read_us:.1f}",
                f"{result.cutedsl_single_read_memcpy_pct:.1f}%",
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "M",
                "N",
                "memcpy GB/s",
                "cute dim0 us",
                "cute dim0 %memcpy",
                "cute dim1 us",
                "cute dim1 %memcpy",
                "triton us",
                "triton %memcpy",
                "cute serial us",
                "cute serial %memcpy",
                "cute streams us",
                "cute streams %memcpy",
                "cute single-read us",
                "cute single-read %memcpy",
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shape", action="append", default=[])
    args = parser.parse_args()

    configs = get_configs(args.quick)
    if args.shape:
        wanted = {
            tuple(int(part) for part in shape.lower().split("x"))
            for shape in args.shape
        }
        configs = [config for config in configs if (config.M, config.N) in wanted]
        if not configs:
            raise ValueError(f"no benchmark configs matched --shape={args.shape}")

    results = []
    for config in tqdm(configs):
        results.append((config, run_experiment(config)))
    print_results(results)


if __name__ == "__main__":
    main()
