from typing import Callable, Tuple

import fire
import torch
import triton
from torch._inductor.utils import do_bench_using_profiling

torch.manual_seed(0)

bytes_per_el_bf16 = 2


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


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


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
    assert mode in ("dim0", "dim1", "dim0_dim1")

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

        bytes_rw = sum(t.numel() for t in [x, y_d1, s_d1]) * bytes_per_el_bf16
        bps = bytes_rw / (time_us / 1e6)

    else:
        assert mode == "dim0_dim1"
        scale_dim0_dim1_reference_c = torch.compile(scale_dim0_dim1_reference)
        y_d0, y_d1, s_d0, s_d1 = scale_dim0_dim1_reference_c(x, BLOCK_SIZE)

        for _ in range(2):
            __ = scale_dim0_dim1_reference_c(x, BLOCK_SIZE)
        time_us = benchmark_cuda_function_in_microseconds(
            lambda x, b: scale_dim0_dim1_reference_c(x, BLOCK_SIZE),
            x,
            BLOCK_SIZE,
        )

        bytes_rw = (
            sum(t.numel() for t in [x, y_d0, y_d1, s_d0, s_d1]) * bytes_per_el_bf16
        )
        bps = bytes_rw / (time_us / 1e6)

    print("time_us", time_us)
    print("mem_bw_gbps", bps / 1e9)


if __name__ == "__main__":
    fire.Fire(run)
