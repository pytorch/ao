# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import fire
import torch
from tabulate import tabulate
from torch._inductor.utils import do_bench_using_profiling
from tqdm import tqdm

from torchao.float8.float8_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
    _hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import _pad_tensor_for_matmul

# estimating TOPs for matmuls in fp32, fp16, fp8
# assuming A * B = C, with A being M * K, B being K * N, C being M * N

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 1979e12
h100_peak_tops_float8_tc = 3958e12

dtype_to_peak_tops = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}


def benchmark_fn_in_usec(f, *args, **kwargs):
    no_args = lambda: f(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


def get_tops_info(tops, time, peak_tops):
    time_sec = time / 1e6
    tops_sec = float(tops) / time_sec
    pct_top_peak = tops_sec / peak_tops
    return tops_sec, pct_top_peak


def do_fp8_matmul(A, B, fp8_dtype, out_dtype):
    scale_a = torch.tensor([1], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1], device="cuda", dtype=torch.float32)

    a_config = ScaledMMConfig(
        emulate=False, use_fast_accum=True, fp8_output=True, pad_inner_dim=True
    )
    b_config = ScaledMMConfig(
        emulate=False, use_fast_accum=True, fp8_output=True, pad_inner_dim=True
    )
    a_config = LinearMMConfig(a_config, a_config, a_config)
    b_config = LinearMMConfig(b_config, b_config, b_config)

    a_fp8 = _hp_tensor_and_scale_to_float8(
        A,
        scale_a,
        fp8_dtype,
        a_config,
        GemmInputRole.INPUT,
    )
    b_fp8 = _hp_tensor_and_scale_to_float8(
        B,
        scale_b,
        fp8_dtype,
        b_config,
        GemmInputRole.WEIGHT,
    )

    return a_fp8 @ b_fp8


def do_fp8_pad_first_matmul(A, B, fp8_dtype, out_dtype):
    # Breaks with compile due to trying to pad on fp8 dtype
    # return do_fp8_matmul(A, B, fp8_dtype, out_dtype)
    A_pad = _pad_tensor_for_matmul(A, dims=1)  # mem copy
    B_pad = _pad_tensor_for_matmul(B, dims=0)  # mem copy

    scale_a = torch.tensor([1], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1], device="cuda", dtype=torch.float32)

    A_pad = A_pad.to(fp8_dtype)  # mem copy
    B_pad = B_pad.to(fp8_dtype)  # mem copy

    B_pad = B_pad.t().contiguous().t()  # mem copy

    return torch._scaled_mm(
        A_pad, B_pad, scale_a, scale_b, out_dtype=out_dtype, use_fast_accum=True
    )


def do_hp_matmul(A, B):
    return torch.matmul(A, B)


def do_aligned_bf16_matmul(A, B):
    A_pad = _pad_tensor_for_matmul(A, dims=1)
    B_pad = _pad_tensor_for_matmul(B, dims=0)
    return torch.matmul(A_pad, B_pad)


@dataclass
class Experiment_config:
    M: int
    K: int
    N: int
    output_dtype: torch.dtype
    fp8_dtype: torch.dtype

    def __iter__(self):
        return iter((self.M, self.K, self.N, self.output_dtype, self.fp8_dtype))


def gen_configs():
    shapes = shapes = [
        (8193, 2501, 5008),
        (65, 253, 4096),
        (1023, 1029, 2512),
        (4095, 511, 10000),
        (2047, 3073, 8192),
        (511, 769, 7504),
        (127, 4097, 12288),
        (32769, 15, 15024),
        (9217, 8191, 20480),
        (16385, 1025, 25008),
    ]
    output_dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn
    return [Experiment_config(*shape, output_dtype, fp8_dtype) for shape in shapes]


@torch.no_grad()
def run(compile: bool = False, n_limit: Optional[int] = None):
    device = "cuda"
    experiments = gen_configs()
    results = []
    tops_table = []
    tops_headers = [
        "Shape",
        "Ref Dtype",
        "Ref Tops",
        "Aligned BF16 Tops",
        "FP8 Tops",
        "Ref % Peak",
        "Aligned BF16 % Peak",
        "FP8 % Peak",
    ]

    for experiment in tqdm(experiments):
        M, K, N, output_dtype, fp8_dtype = experiment
        tops = 2 * M * N * K

        A_base = torch.rand(M, K, device=device, dtype=output_dtype)
        B_base = torch.rand(K, N, device=device, dtype=output_dtype)

        hp_func = torch.compile(do_hp_matmul) if compile else do_hp_matmul
        aligned_bf16_func = (
            torch.compile(do_aligned_bf16_matmul) if compile else do_aligned_bf16_matmul
        )
        fp8_func = torch.compile(do_fp8_pad_first_matmul) if compile else do_fp8_matmul

        ref_time = benchmark_fn_in_usec(hp_func, A_base, B_base)
        aligned_bf16_time = benchmark_fn_in_usec(aligned_bf16_func, A_base, B_base)
        fp8_time = benchmark_fn_in_usec(
            fp8_func, A_base, B_base, fp8_dtype, output_dtype
        )

        ref_tops_sec, ref_pct_top_peak = get_tops_info(
            tops, ref_time, dtype_to_peak_tops[output_dtype]
        )
        aligned_bf16_tops_sec, aligned_bf16_pct_top_peak = get_tops_info(
            tops, aligned_bf16_time, dtype_to_peak_tops[torch.bfloat16]
        )
        fp8_tops_sec, fp8_pct_top_peak = get_tops_info(
            tops, fp8_time, dtype_to_peak_tops[fp8_dtype]
        )
        tops_table.append(
            [
                f"({M}x{K}x{N})",
                f"{output_dtype}",
                f"{ref_tops_sec:.2E}",
                f"{aligned_bf16_tops_sec:.2E}",
                f"{fp8_tops_sec:.2E}",
                f"{ref_pct_top_peak:.3f}",
                f"{aligned_bf16_pct_top_peak:.3f}",
                f"{fp8_pct_top_peak:.3f}",
            ]
        )
        results.append(
            [
                (M, K, N),
                output_dtype,
                ref_time,
                aligned_bf16_time,
                fp8_time,
                ref_time / aligned_bf16_time,
                ref_time / fp8_time,
            ]
        )

    print("TOPs".center(80, "*"))
    print(tabulate(tops_table, headers=tops_headers))
    print("Speed Results".center(80, "*"))
    headers = [
        "Shape",
        "Ref Dtype",
        "Ref Time",
        "Aligned BF16 Time",
        "FP8 Time",
        "Aligned BF16 Speedup",
        "FP8 Speedup",
    ]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    fire.Fire(run)
