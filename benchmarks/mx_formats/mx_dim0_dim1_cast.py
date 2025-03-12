# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Starting with https://github.com/vkuzo/pytorch_scripts/blob/main/mx_cast_poc/20250305_mx_dim0_dim1_cast.py
and making it nice.
"""

from typing import Callable

import fire
import torch
import triton
from torch._inductor.utils import do_bench_using_profiling

from torchao.prototype.mx_formats.custom_cast import (
    to_mxfp8_across_dim0_and_dim1,
    to_mxfp8_across_dim0_and_dim1_reference,
)

torch.manual_seed(0)


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


def compute_error(x, y):
    Ps = torch.linalg.norm(x)
    Pn = torch.linalg.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


def run(
    M: int = 4096,
    K: int = 2048,
    BLOCK_SIZE: int = 32,
):
    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"triton version: {triton.__version__}")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 1000

    to_mxfp8_across_dim0_and_dim1_reference_c = torch.compile(
        to_mxfp8_across_dim0_and_dim1_reference
    )

    # reference implementation (plain PyTorch + torch.compile)
    x_d0, x_d1, scale_e8m0_d0, scale_e8m0_d1 = (
        to_mxfp8_across_dim0_and_dim1_reference_c(x, BLOCK_SIZE)
    )
    x_d0, x_d1 = x_d0.bfloat16(), x_d1.bfloat16()
    scale_fp_d0 = scale_e8m0_d0.float()
    scale_fp_d1 = scale_e8m0_d1.float()
    x_d0_and_back = (x_d0.reshape(-1, BLOCK_SIZE) * scale_fp_d0).reshape(x_d0.shape)
    x_d1_and_back = (
        (x_d1.t().reshape(-1, BLOCK_SIZE) * scale_fp_d1).reshape(x_d1.t().shape).t()
    )

    sqnr_bf16_vs_dim0_ref = compute_error(x, x_d0_and_back)
    sqnr_bf16_vs_dim1_ref = compute_error(x, x_d1_and_back)
    print(
        f"bf16 vs normalized reference sqnrs: dim0 {sqnr_bf16_vs_dim0_ref}, dim1 {sqnr_bf16_vs_dim1_ref}"
    )
    assert (
        sqnr_bf16_vs_dim0_ref > 28 and sqnr_bf16_vs_dim1_ref > 28
    ), "reference mx numerics are incorrect"

    # basic triton kernel
    x_d0_t, x_d1_t, scale_e8m0_d0_t, scale_e8m0_d1_t = to_mxfp8_across_dim0_and_dim1(
        x, tile_size=BLOCK_SIZE
    )
    x_d0_t, x_d1_t = x_d0_t.bfloat16(), x_d1_t.bfloat16()

    # ensure bitwise equivalency of outputs with reference
    torch.testing.assert_close(x_d0, x_d0_t, atol=0, rtol=0)
    torch.testing.assert_close(x_d1, x_d1_t, atol=0, rtol=0)
    torch.testing.assert_close(scale_e8m0_d0, scale_e8m0_d0_t, atol=0, rtol=0)
    torch.testing.assert_close(scale_e8m0_d1, scale_e8m0_d1_t, atol=0, rtol=0)
    print("normalized reference vs normalized triton are bitwise equivalent")

    # now, measure performance

    # warm up
    for _ in range(2):
        __ = to_mxfp8_across_dim0_and_dim1_reference_c(x, BLOCK_SIZE)
    time_reference_compile_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_across_dim0_and_dim1_reference_c(x, b), x, BLOCK_SIZE
    )

    # warm up
    for _ in range(2):
        __ = to_mxfp8_across_dim0_and_dim1(x, tile_size=BLOCK_SIZE)
    time_triton_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_across_dim0_and_dim1(x, tile_size=BLOCK_SIZE),
        x,
        BLOCK_SIZE,
    )

    # calculate bytes read/written
    bytes_per_el_bf16 = 2
    bytes_per_el_fp8 = 1
    triton_bytes_read = x.numel() * bytes_per_el_bf16
    triton_bytes_written = (
        sum(x.numel() for x in (x_d0_t, x_d1_t, scale_e8m0_d0_t, scale_e8m0_d1_t))
        * bytes_per_el_fp8
    )
    triton_achieved_mem_bw_gbps = (triton_bytes_read + triton_bytes_written) / (
        time_triton_us / 1e6
    )
    # TODO(future PR): read 8.0 TB/s number from roofline_utils.py instead of hardcoding
    triton_pct_peak_mem_bw = triton_achieved_mem_bw_gbps / 8.0e12

    print("time_reference_compile_us", time_reference_compile_us)
    print("time_triton_us", time_triton_us)
    print("triton_achieved_mem_bw_gbps", triton_achieved_mem_bw_gbps)
    # Note: as of 2025-03-11, inductor code for adding 1.0 to a large bf16 tensor
    # can achieve around 50-70% of B200 peak mem bw
    print("triton_pct_peak_mem_bw", triton_pct_peak_mem_bw)
    print("speedup", time_reference_compile_us / time_triton_us)


if __name__ == "__main__":
    fire.Fire(run)
