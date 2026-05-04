# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Sweeps (BLOCK_SIZE, BLOCK_SIZE_ITER, num_warps) for the colwise FP8 kernel
# on representative DeepSeek-MoE-16B training shapes (MI300X).
#
# Usage:
#   cd ~/ao
#   python benchmarks/prototype/moe_training/fp8_rowwise/bench_colwise_block_configs.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
import triton
import triton.language as tl
from tabulate import tabulate
from triton.testing import do_bench

from torchao.prototype.moe_training.utils import generate_jagged_offs

EPS = 1e-12
FP8_DTYPE_MAP = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
}

device = torch.device("cuda")


# ---------------------------------------------------------------------------
# Standalone colwise kernel (no autotune wrapper) so we can inject any config.
# Mirrors the production _triton_fp8_per_group_colwise_scales_kernel exactly.
# Input shape: (K, N) where K=token rows (jagged dim), N=hidden cols.
# BLOCK_SIZE tiles N; BLOCK_SIZE_ITER tiles K in the inner loop.
# ---------------------------------------------------------------------------
@triton.jit
def _colwise_kernel(
    input_ptr,
    offsets_ptr,
    out_ptr,
    scales_ptr,
    K: tl.int64,
    N: tl.int64,
    N_GROUPS: tl.int64,
    str_ir: tl.int64,
    str_ic: tl.int64,
    str_or: tl.int64,
    str_oc: tl.int64,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    in_dtype: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_ITER: tl.constexpr,
    EPS: tl.constexpr,
):
    bcol = tl.program_id(0)
    gidx = tl.program_id(1)
    rs = tl.load(offsets_ptr + gidx - 1, mask=gidx > 0, other=0)
    re = tl.load(offsets_ptr + gidx)
    col_offs = (bcol * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    amax = tl.zeros((BLOCK_SIZE,), dtype=in_dtype)
    for r0 in range(rs, re, BLOCK_SIZE_ITER):
        row_offs = (r0 + tl.arange(0, BLOCK_SIZE_ITER)).to(tl.int64)
        offs = row_offs[:, None] * str_ir + col_offs[None, :] * str_ic
        mask = (row_offs[:, None] < re) & (col_offs[None, :] < N)
        d = tl.load(input_ptr + offs, mask=mask, other=0.0).to(in_dtype)
        amax = tl.maximum(amax, tl.max(tl.abs(d), axis=0)).to(in_dtype)
    amax = amax.to(tl.float64)
    s = (fp8_max / tl.clamp(amax, min=EPS, max=float("inf"))).to(tl.float32)
    s = tl.exp2(tl.floor(tl.log2(s)))
    sc_offs = col_offs + N * gidx
    sc_mask = tl.arange(0, BLOCK_SIZE) < N
    tl.store(scales_ptr + sc_offs, s, mask=sc_mask)
    for r0 in range(rs, re, BLOCK_SIZE_ITER):
        row_offs = (r0 + tl.arange(0, BLOCK_SIZE_ITER)).to(tl.int64)
        offs = row_offs[:, None] * str_ir + col_offs[None, :] * str_ic
        mask = (row_offs[:, None] < re) & (col_offs[None, :] < N)
        d = tl.load(input_ptr + offs, mask=mask, other=0.0).to(in_dtype)
        sd = d * s[None, :]
        fp8d = tl.clamp(sd, min=fp8_min, max=fp8_max).to(out_dtype)
        o_offs = row_offs[:, None] * str_or + col_offs[None, :] * str_oc
        tl.store(out_ptr + o_offs, fp8d, mask=mask)


@dataclass(frozen=True)
class ExperimentConfig:
    M: int  # total token rows (jagged dim, K_triton)
    K: int  # hidden cols (N_triton)
    n_groups: int
    block_size: int
    block_size_iter: int
    num_warps: int
    fp8_dtype: torch.dtype


@dataclass(frozen=True)
class ExperimentResult:
    time_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


SHAPES = [
    (16640, 2048, 64),
    (16640, 5120, 64),
    (16640, 2048, 128),
    (16640, 5120, 128),
]

BLOCK_SIZES = [32, 64, 128, 256]
BLOCK_SIZE_ITERS = [32, 64, 128, 256]
NUM_WARPS_LIST = [4, 8]


def get_configs(fp8_dtype: torch.dtype) -> List[ExperimentConfig]:
    configs = []
    for (M, K, n_groups), bs, bsi, nw in itertools.product(
        SHAPES, BLOCK_SIZES, BLOCK_SIZE_ITERS, NUM_WARPS_LIST
    ):
        configs.append(
            ExperimentConfig(
                M=M,
                K=K,
                n_groups=n_groups,
                block_size=bs,
                block_size_iter=bsi,
                num_warps=nw,
                fp8_dtype=fp8_dtype,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M, K, n_groups = config.M, config.K, config.n_groups
    bs, bsi, nw = config.block_size, config.block_size_iter, config.num_warps
    fp8_dtype = config.fp8_dtype
    tl_fp8_dtype = FP8_DTYPE_MAP[fp8_dtype]

    inp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    offs = generate_jagged_offs(n_groups, M, multiple_of=16)
    fp8_min = torch.finfo(fp8_dtype).min
    fp8_max = torch.finfo(fp8_dtype).max

    out = torch.empty_like(inp, dtype=fp8_dtype).as_strided(inp.size(), (1, M))
    sc = torch.empty(K * n_groups, dtype=torch.float32, device=device)
    grid = (triton.cdiv(K, bs), n_groups)

    def run():
        _colwise_kernel[grid](
            inp,
            offs,
            out,
            sc,
            M,
            K,
            n_groups,
            inp.stride(0),
            inp.stride(1),
            out.stride(0),
            out.stride(1),
            fp8_min,
            fp8_max,
            tl.bfloat16,
            tl_fp8_dtype,
            BLOCK_SIZE=bs,
            BLOCK_SIZE_ITER=bsi,
            EPS=EPS,
            num_warps=nw,
            num_stages=2,
        )

    t_us = do_bench(run, return_mode="median") * 1e3
    return ExperimentResult(time_us=t_us)


def print_results(experiments: List[Experiment]):
    # Group by (M, K, n_groups) and print best configs
    from collections import defaultdict

    groups = defaultdict(list)
    for e in experiments:
        key = (e.config.M, e.config.K, e.config.n_groups)
        groups[key].append(e)

    summary_rows = []
    for key, exps in groups.items():
        M, K, n_groups = key
        best = min(exps, key=lambda e: e.result.time_us)
        summary_rows.append(
            [
                M,
                K,
                n_groups,
                best.config.block_size,
                best.config.block_size_iter,
                best.config.num_warps,
                f"{best.result.time_us:.1f}",
            ]
        )

    print()
    print("=" * 70)
    print("Colwise FP8 Block Config Sweep — Best per Shape")
    print("=" * 70)
    headers = [
        "M",
        "K",
        "n_groups",
        "BLOCK_SIZE",
        "BLOCK_SIZE_ITER",
        "num_warps",
        "time (us)",
    ]
    print(tabulate(summary_rows, headers=headers))

    # Full results table
    print()
    print("=" * 70)
    print("Full Results")
    print("=" * 70)
    all_rows = []
    for e in experiments:
        c, r = e.config, e.result
        all_rows.append(
            [
                c.M,
                c.K,
                c.n_groups,
                c.block_size,
                c.block_size_iter,
                c.num_warps,
                f"{r.time_us:.1f}",
            ]
        )
    print(tabulate(all_rows, headers=headers))


def main():
    fp8_dtype = (
        torch.float8_e4m3fnuz if torch.version.hip is not None else torch.float8_e4m3fn
    )
    print(f"GPU  : {torch.cuda.get_device_name()}")
    print(f"ROCm : {torch.version.hip}")
    print(f"FP8  : {fp8_dtype}")
    print()

    configs = get_configs(fp8_dtype)
    experiments = []
    for config in configs:
        result = run_experiment(config)
        experiments.append(Experiment(config=config, result=result))

    print_results(experiments)


if __name__ == "__main__":
    main()
