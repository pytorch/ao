# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import fire
import torch
import triton
from triton.testing import do_bench

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_dim0,
    triton_to_mxfp8_dim1,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx

torch.manual_seed(0)

bytes_per_el_bf16 = 2
bytes_per_el_fp8 = 1


def scale_dim0_reference(x_hp, block_size) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_hp.is_contiguous()
    x_hp_d0_block = x_hp.reshape(-1, block_size)
    x_hp_d0_block_abs = x_hp_d0_block.abs()
    amax_dim0 = torch.amax(x_hp_d0_block_abs, dim=1).unsqueeze(1)
    x_hp_d0_block_normalized = x_hp_d0_block / amax_dim0
    x_hp_d0_normalized = x_hp_d0_block_normalized.reshape(x_hp.shape)
    return x_hp_d0_normalized, amax_dim0


def scale_dim1_reference(x_hp, block_size) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_hp.is_contiguous()
    x_hp_d1 = x_hp.t().contiguous()
    x_hp_d1_block = x_hp_d1.reshape(-1, block_size)
    x_hp_d1_block_abs = x_hp_d1_block.abs()
    amax_dim1 = torch.amax(x_hp_d1_block_abs, dim=1).unsqueeze(1)
    x_hp_d1_block_normalized = x_hp_d1_block / amax_dim1
    x_hp_d1_normalized = x_hp_d1_block_normalized.reshape(x_hp_d1.shape)
    return x_hp_d1_normalized, amax_dim1


def scale_dim0_dim1_reference(
    x_hp: torch.Tensor, block_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    # normalize across dim0
    x_hp_d0_normalized, amax_dim0 = scale_dim0_reference(x_hp, block_size)
    # normalize across dim1
    x_hp_d1_normalized, amax_dim1 = scale_dim1_reference(x_hp, block_size)
    return x_hp_d0_normalized, x_hp_d1_normalized.t(), amax_dim0, amax_dim1


def to_mx_dim0_reference(
    x_hp,
    block_size,
    scaling_mode=ScaleCalculationMode.FLOOR,
    target_dtype=torch.float8_e4m3fn,
):
    scale_d0, data_d0 = to_mx(x_hp, target_dtype, block_size, scaling_mode=scaling_mode)
    return data_d0, scale_d0


def to_mx_dim1_reference(
    x_hp,
    block_size,
    scaling_mode=ScaleCalculationMode.FLOOR,
    target_dtype=torch.float8_e4m3fn,
):
    x_hp = x_hp.t().contiguous()
    scale_d1, data_d1 = to_mx(x_hp, target_dtype, block_size, scaling_mode=scaling_mode)
    return data_d1.t(), scale_d1


def benchmark_cuda_function_in_microseconds(f, *args):
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def run(
    M: int = 16384,
    K: int = 16384,
    BLOCK_SIZE: int = 32,
    mode: str = "dim0",
):
    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"triton version: {triton.__version__}")
    print(f"mode: {mode}")
    assert mode in (
        "dim0",
        "dim1",
        "dim0_dim1",
        "dim0_mxfp8_floor",
        "dim0_mxfp4_floor",
        "dim0_mxfp8_rceil",
        "dim0_mxfp8_triton_floor",
        "dim1_mxfp8_floor",
        "dim1_mxfp8_rceil",
        "dim1_mxfp8_triton_floor",
        "dim1_mxfp8_cuda_floor",
        "dim1_mxfp8_cuda_rceil",
    )

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 1000

    if mode == "dim0":
        scale_dim0_reference_c = torch.compile(scale_dim0_reference)
        y_d0, s_d0 = scale_dim0_reference_c(x, BLOCK_SIZE)

        for _ in range(2):
            __ = scale_dim0_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: scale_dim0_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d0.dtype == torch.bfloat16
        assert s_d0.dtype == torch.bfloat16
        bytes_rw = sum(t.numel() for t in [x, y_d0, s_d0]) * bytes_per_el_bf16
        bps = bytes_rw / (time_us / 1e6)

    elif mode == "dim1":
        scale_dim1_reference_c = torch.compile(scale_dim1_reference)
        y_d1, s_d1 = scale_dim1_reference_c(x, BLOCK_SIZE)

        for _ in range(2):
            __ = scale_dim1_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: scale_dim1_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d1.dtype == torch.bfloat16
        assert s_d1.dtype == torch.bfloat16
        bytes_rw = sum(t.numel() for t in [x, y_d1, s_d1]) * bytes_per_el_bf16
        bps = bytes_rw / (time_us / 1e6)

    elif mode == "dim0_dim1":
        scale_dim0_dim1_reference_c = torch.compile(scale_dim0_dim1_reference)
        y_d0, y_d1, s_d0, s_d1 = scale_dim0_dim1_reference_c(x, BLOCK_SIZE)

        for _ in range(2):
            __ = scale_dim0_dim1_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: scale_dim0_dim1_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d0.dtype == torch.bfloat16
        assert s_d0.dtype == torch.bfloat16
        assert y_d1.dtype == torch.bfloat16
        assert s_d1.dtype == torch.bfloat16
        bytes_rw = (
            sum(t.numel() for t in [x, y_d0, y_d1, s_d0, s_d1]) * bytes_per_el_bf16
        )
        bps = bytes_rw / (time_us / 1e6)

    elif mode == "dim0_mxfp8_floor":
        to_mx_dim0_reference_c = torch.compile(to_mx_dim0_reference)
        y_d0, s_d0 = to_mx_dim0_reference_c(x, BLOCK_SIZE)

        for _ in range(2):
            __ = to_mx_dim0_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: to_mx_dim0_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d0.dtype == torch.float8_e4m3fn
        assert s_d0.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d0.numel() + s_d0.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim0_mxfp4_floor":
        to_mx_dim0_reference_c = torch.compile(to_mx_dim0_reference)
        y_d0, s_d0 = to_mx_dim0_reference_c(
            x, BLOCK_SIZE, target_dtype=torch.float4_e2m1fn_x2
        )

        for _ in range(2):
            __ = to_mx_dim0_reference_c(
                x, BLOCK_SIZE, target_dtype=torch.float4_e2m1fn_x2
            )
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: to_mx_dim0_reference_c(
                x, BLOCK_SIZE, target_dtype=torch.float4_e2m1fn_x2
            ),
            x,
            BLOCK_SIZE,
        )

        # TODO(future PR): make to_mx return float4 directly
        assert y_d0.dtype == torch.uint8
        assert s_d0.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d0.numel() + s_d0.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim0_mxfp8_rceil":
        to_mx_dim0_reference_c = torch.compile(to_mx_dim0_reference)
        y_d0, s_d0 = to_mx_dim0_reference_c(x, BLOCK_SIZE, ScaleCalculationMode.RCEIL)

        for _ in range(2):
            __ = to_mx_dim0_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: to_mx_dim0_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d0.dtype == torch.float8_e4m3fn
        assert s_d0.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d0.numel() + s_d0.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim0_mxfp8_triton_floor":
        y_d0, s_d0 = triton_to_mxfp8_dim0(x, inner_block_size=BLOCK_SIZE)

        for _ in range(2):
            __ = triton_to_mxfp8_dim0(x, inner_block_size=BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: triton_to_mxfp8_dim0(x, inner_block_size=BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )
        assert y_d0.dtype == torch.float8_e4m3fn
        assert s_d0.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d0.numel() + s_d0.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim1_mxfp8_floor":
        to_mx_dim1_reference_c = torch.compile(to_mx_dim1_reference)
        y_d1, s_d1 = to_mx_dim1_reference_c(x, BLOCK_SIZE)

        for _ in range(2):
            __ = to_mx_dim1_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: to_mx_dim1_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d1.dtype == torch.float8_e4m3fn
        assert s_d1.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d1.numel() + s_d1.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim1_mxfp8_rceil":
        to_mx_dim1_reference_c = torch.compile(to_mx_dim1_reference)
        y_d1, s_d1 = to_mx_dim1_reference_c(x, BLOCK_SIZE, ScaleCalculationMode.RCEIL)

        for _ in range(2):
            __ = to_mx_dim1_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: to_mx_dim1_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d1.dtype == torch.float8_e4m3fn
        assert s_d1.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d1.numel() + s_d1.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim1_mxfp8_triton_floor":
        y_d1, s_d1 = triton_to_mxfp8_dim1(x, inner_block_size=BLOCK_SIZE)

        for _ in range(2):
            __ = triton_to_mxfp8_dim1(x, inner_block_size=BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: triton_to_mxfp8_dim1(x, inner_block_size=BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        assert y_d1.dtype == torch.float8_e4m3fn
        assert s_d1.dtype == torch.float8_e8m0fnu
        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d1.numel() + s_d1.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim1_mxfp8_cuda_floor":
        from torchao.prototype import mxfp8_cuda

        _, y_d1, _, s_d1 = mxfp8_cuda.quantize(
            x, rowwise=False, colwise=True, scaling_mode="floor"
        )

        for _ in range(2):
            __ = mxfp8_cuda.quantize(
                x, rowwise=False, colwise=True, scaling_mode="floor"
            )

        time_us = benchmark_cuda_function_in_microseconds(
            lambda x: mxfp8_cuda.quantize(
                x, rowwise=False, colwise=True, scaling_mode="floor"
            ),
            x,
        )

        assert y_d1.dtype == torch.float8_e4m3fn
        assert s_d1.dtype == torch.float8_e8m0fnu

        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d1.numel() + s_d1.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    elif mode == "dim1_mxfp8_cuda_rceil":
        from torchao.prototype import mxfp8_cuda

        _, y_d1, _, s_d1 = mxfp8_cuda.quantize(
            x, rowwise=False, colwise=True, scaling_mode="rceil"
        )

        for _ in range(2):
            __ = mxfp8_cuda.quantize(
                x, rowwise=False, colwise=True, scaling_mode="rceil"
            )

        time_us = benchmark_cuda_function_in_microseconds(
            lambda x: mxfp8_cuda.quantize(
                x, rowwise=False, colwise=True, scaling_mode="rceil"
            ),
            x,
        )

        assert y_d1.dtype == torch.float8_e4m3fn
        assert s_d1.dtype == torch.float8_e8m0fnu

        bytes_r = x.numel() * bytes_per_el_bf16
        bytes_w = (y_d1.numel() + s_d1.numel()) * bytes_per_el_fp8
        bps = (bytes_r + bytes_w) / (time_us / 1e6)

    else:
        raise AssertionError(f"unknown mode {mode}")

    print("time_us", time_us)
    print("mem_bw_gbps", bps / 1e9)


if __name__ == "__main__":
    fire.Fire(run)
