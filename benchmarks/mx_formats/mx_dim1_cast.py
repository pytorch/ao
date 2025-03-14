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
    to_mxfp8_dim1,
)

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
):
    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"triton version: {triton.__version__}")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 1000

    x_d1_only_t, scale_e8m0_d1_only_t = to_mxfp8_dim1(x, row_tile_size=BLOCK_SIZE)

    for _ in range(2):
        __ = to_mxfp8_dim1(x, row_tile_size=BLOCK_SIZE)
    time_triton_dim1_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: to_mxfp8_dim1(x, row_tile_size=BLOCK_SIZE),
        x,
        BLOCK_SIZE,
    )

    dim0_bytes_read = x.numel() * bytes_per_el_bf16
    dim0_bytes_written = (
        x_d1_only_t.numel() + scale_e8m0_d1_only_t.numel()
    ) * bytes_per_el_fp8
    dim0_bytes_rw = dim0_bytes_read + dim0_bytes_written
    triton_dim1_bps = dim0_bytes_rw / (time_triton_dim1_us / 1e6)

    print("time_triton_dim1_us", time_triton_dim1_us)
    print("triton_dim1_mem_bw_gbps", triton_dim1_bps / 1e9)


if __name__ == "__main__":
    fire.Fire(run)
