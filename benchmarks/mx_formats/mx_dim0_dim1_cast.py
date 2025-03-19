# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import fire
import pandas as pd
import torch
import triton
from torch._inductor.utils import do_bench_using_profiling

from torchao.prototype.mx_formats.custom_cast import (
    to_mxfp8_across_dim0_and_dim1,
    to_mxfp8_across_dim0_and_dim1_reference,
    to_mxfp8_dim0_reference,
    to_mxfp8_dim1,
    to_mxfp8_dim1_reference,
)
from torchao.quantization.utils import compute_error
from torchao.testing.float8.roofline_utils import get_specs

torch.manual_seed(0)

bytes_per_el_bf16 = 2
bytes_per_el_fp8 = 1


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


def run(
    M: int = 4096,
    K: int = 2048,
    BLOCK_SIZE: int = 32,
    check_accuracy: bool = True,
):
    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"triton version: {triton.__version__}")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 1000

    to_mxfp8_across_dim0_and_dim1_reference_c = torch.compile(
        to_mxfp8_across_dim0_and_dim1_reference
    )
    to_mxfp8_dim0_reference_c = torch.compile(to_mxfp8_dim0_reference)
    to_mxfp8_dim1_reference_c = torch.compile(to_mxfp8_dim1_reference)

    # reference implementation (plain PyTorch + torch.compile)
    x_d0, x_d1, scale_e8m0_d0, scale_e8m0_d1 = (
        to_mxfp8_across_dim0_and_dim1_reference_c(x, BLOCK_SIZE)
    )

    # verify reference dim0_dim1 matches dim0 and dim1 separately
    x_d0_separate, scale_e8m0_d0_separate = to_mxfp8_dim0_reference_c(x, BLOCK_SIZE)
    x_d1_separate, scale_e8m0_d1_separate = to_mxfp8_dim1_reference_c(x, BLOCK_SIZE)
    torch.testing.assert_close(x_d0, x_d0_separate, atol=0, rtol=0)
    torch.testing.assert_close(x_d1, x_d1_separate, atol=0, rtol=0)
    torch.testing.assert_close(scale_e8m0_d0, scale_e8m0_d0_separate, atol=0, rtol=0)
    torch.testing.assert_close(scale_e8m0_d1, scale_e8m0_d1_separate, atol=0, rtol=0)

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

    # triton kernel for dim1 only
    x_d1_only_t, scale_e8m0_d1_only_t = to_mxfp8_dim1(x, row_tile_size=BLOCK_SIZE)

    # triton kernel for dim0 and dim1
    x_d0_t, x_d1_t, scale_e8m0_d0_t, scale_e8m0_d1_t = to_mxfp8_across_dim0_and_dim1(
        x, tile_size=BLOCK_SIZE
    )
    x_d0_t, x_d1_t, x_d1_only_t = (
        x_d0_t.bfloat16(),
        x_d1_t.bfloat16(),
        x_d1_only_t.bfloat16(),
    )

    # ensure bitwise equivalency of outputs with reference
    if check_accuracy:
        torch.testing.assert_close(x_d0, x_d0_t, atol=0, rtol=0)
        torch.testing.assert_close(x_d1, x_d1_t, atol=0, rtol=0)
        torch.testing.assert_close(scale_e8m0_d0, scale_e8m0_d0_t, atol=0, rtol=0)
        torch.testing.assert_close(scale_e8m0_d1, scale_e8m0_d1_t, atol=0, rtol=0)
        print("reference vs triton dim0_dim1 are bitwise equivalent")
        torch.testing.assert_close(x_d1, x_d1_only_t, atol=0, rtol=0)
        torch.testing.assert_close(scale_e8m0_d1, scale_e8m0_d1_only_t, atol=0, rtol=0)
        print("reference vs triton dim1 are bitwise equivalent")
    else:
        print("accuracy checking skipped")

    # now, measure performance

    # define a speed-of-light torch.compile kernel to get a sense of
    # achievable mem bandwidth
    def add_one(x):
        x = x + 1
        return x

    add_one_c = torch.compile(add_one)

    for _ in range(2):
        __ = add_one_c(x)
    time_add_one_compile_us = benchmark_cuda_function_in_microseconds(
        lambda x: add_one(x), x
    )

    for _ in range(2):
        __ = to_mxfp8_across_dim0_and_dim1_reference_c(x, BLOCK_SIZE)
    time_ref_dim0_dim1_compile_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_across_dim0_and_dim1_reference_c(x, b), x, BLOCK_SIZE
    )

    for _ in range(2):
        __ = to_mxfp8_dim0_reference_c(x, BLOCK_SIZE)
    time_ref_dim0_compile_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_dim0_reference_c(x, b), x, BLOCK_SIZE
    )

    for _ in range(2):
        __ = to_mxfp8_dim1_reference_c(x, BLOCK_SIZE)
    time_ref_dim1_compile_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_dim1_reference_c(x, b), x, BLOCK_SIZE
    )

    for _ in range(2):
        __ = to_mxfp8_dim1(x, row_tile_size=BLOCK_SIZE)
    time_triton_dim1_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_dim1(x, row_tile_size=BLOCK_SIZE),
        x,
        BLOCK_SIZE,
    )

    # warm up
    for _ in range(2):
        __ = to_mxfp8_across_dim0_and_dim1(x, tile_size=BLOCK_SIZE)
    time_triton_dim0_dim1_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_across_dim0_and_dim1(x, tile_size=BLOCK_SIZE),
        x,
        BLOCK_SIZE,
    )

    # calculate memory bandwidth
    peak_mem_bw = get_specs()["peak_mem_bw_bytes_sec"]

    # add_one kernel
    add_one_bps = x.numel() * bytes_per_el_bf16 * 2 / (time_add_one_compile_us / 1e6)

    # dim0 or dim1 kernel
    dim0_bytes_read = x.numel() * bytes_per_el_bf16
    dim0_bytes_written = (x_d0_t.numel() + scale_e8m0_d0_t.numel()) * bytes_per_el_fp8
    dim0_bytes_rw = dim0_bytes_read + dim0_bytes_written
    ref_dim0_bps = dim0_bytes_rw / (time_ref_dim0_compile_us / 1e6)
    ref_dim1_bps = dim0_bytes_rw / (time_ref_dim1_compile_us / 1e6)
    triton_dim1_bps = dim0_bytes_rw / (time_triton_dim1_us / 1e6)

    # triton dim0_dim1 kernel
    triton_dim0_dim1_bytes_read = x.numel() * bytes_per_el_bf16
    triton_dim0_dim1_bytes_written = (
        sum(x.numel() for x in (x_d0_t, x_d1_t, scale_e8m0_d0_t, scale_e8m0_d1_t))
        * bytes_per_el_fp8
    )
    triton_dim0_dim1_bps = (
        triton_dim0_dim1_bytes_read + triton_dim0_dim1_bytes_written
    ) / (time_triton_dim0_dim1_us / 1e6)

    results = [
        ["add_one", time_add_one_compile_us, add_one_bps / 1e9],
        ["compile_dim0", time_ref_dim0_compile_us, ref_dim0_bps / 1e9],
        ["compile_dim1", time_ref_dim1_compile_us, ref_dim1_bps / 1e9],
        [
            "compile_dim0_dim1",
            time_ref_dim0_dim1_compile_us,
            triton_dim0_dim1_bps / 1e9,
        ],
        ["triton_dim1", time_triton_dim1_us, triton_dim1_bps / 1e9],
        ["triton_dim0_dim1", time_triton_dim0_dim1_us, triton_dim0_dim1_bps / 1e9],
    ]
    df = pd.DataFrame(results, columns=["experiment", "time_us", "mem_bw_gbps"])
    df["mem_bw_pct_peak"] = df["mem_bw_gbps"] * 1e9 / peak_mem_bw
    print("\n", df)


if __name__ == "__main__":
    fire.Fire(run)
