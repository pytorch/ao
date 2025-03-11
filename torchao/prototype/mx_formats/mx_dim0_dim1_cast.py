# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Starting with https://github.com/vkuzo/pytorch_scripts/blob/main/mx_cast_poc/20250305_mx_dim0_dim1_cast.py
and making it nice.
"""

from typing import Callable, Tuple

import fire
import torch
import triton
import triton.language as tl
from torch._inductor.utils import do_bench_using_profiling

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


def scale_dim0_dim1_reference(
    x_hp: torch.Tensor, block_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    # normalize across dim0
    x_hp_d0_block = x_hp.reshape(-1, block_size)
    x_hp_d0_block_abs = x_hp_d0_block.abs()
    amax_dim0 = torch.amax(x_hp_d0_block_abs, dim=1).unsqueeze(1)
    x_hp_d0_block_normalized = x_hp_d0_block / amax_dim0
    x_hp_d0_normalized = x_hp_d0_block_normalized.reshape(x_hp.shape)

    # normalize across dim1
    x_hp_d1 = x_hp.t().contiguous()
    x_hp_d1_block = x_hp_d1.reshape(-1, block_size)
    x_hp_d1_block_abs = x_hp_d1_block.abs()
    amax_dim1 = torch.amax(x_hp_d1_block_abs, dim=1).unsqueeze(1)
    x_hp_d1_block_normalized = x_hp_d1_block / amax_dim1
    x_hp_d1_normalized = x_hp_d1_block_normalized.reshape(x_hp_d1.shape)

    return x_hp_d0_normalized, x_hp_d1_normalized.t(), amax_dim0, amax_dim1


@triton.jit
def normalization_kernel(
    x_ptr,  # pointer to input tensor
    output_row_major_ptr,  # pointer to row-major output tensor (row-normalized)
    output_col_major_ptr,  # pointer to column-major output tensor (column-normalized)
    row_max_abs_ptr,  # pointer to store row-wise maximum absolute values
    col_max_abs_ptr,  # pointer to store column-wise maximum absolute values
    n_rows,  # number of rows in the tensor
    n_cols,  # number of columns in the tensor
    TILE_SIZE: tl.constexpr,  # tile size as a compile-time constant
):
    """
    credit: mostly Claude, some Vasiliy
    """

    # Get program ID
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Calculate starting row and column for this tile
    start_row = pid_row * TILE_SIZE
    start_col = pid_col * TILE_SIZE

    # Create offsets for the block
    row_offsets = tl.arange(0, TILE_SIZE)
    col_offsets = tl.arange(0, TILE_SIZE)

    # Compute global row/col positions
    rows = start_row + row_offsets[:, None]  # Convert to 2D for proper broadcasting
    cols = start_col + col_offsets[None, :]

    # Create masks for out-of-bounds accesses
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask & col_mask

    # Compute memory offsets for row-major layout (rows, cols)
    row_major_offsets = (rows * n_cols + cols).to(tl.int32)

    # Compute memory offsets for column-major layout (cols, rows)
    col_major_offsets = (cols * n_rows + rows).to(tl.int32)

    # Load the entire block in a single operation
    x_block = tl.load(x_ptr + row_major_offsets, mask=mask)

    # ----------------------------------------------------
    # Row-wise normalization
    # ----------------------------------------------------
    # Calculate the absolute values of elements in the block
    x_block_abs = tl.abs(x_block)

    # Find the maximum absolute value for each row
    # We use a small epsilon to avoid division by zero
    epsilon = 1e-10
    row_max_abs = tl.max(x_block_abs, axis=1) + epsilon

    # Normalize each row by its maximum absolute value
    # Broadcasting row_max_abs to match x_block's shape
    row_normalized = x_block / row_max_abs[:, None]

    # ----------------------------------------------------
    # Column-wise normalization
    # ----------------------------------------------------
    # Find the maximum absolute value for each column
    col_max_abs = tl.max(x_block_abs, axis=0) + epsilon

    # Normalize each column by its maximum absolute value
    # Broadcasting col_max_abs to match x_block's shape
    col_normalized = x_block / col_max_abs[None, :]

    # Store the row-normalized result in row-major format
    tl.store(output_row_major_ptr + row_major_offsets, row_normalized, mask=mask)

    # Store the column-normalized result in column-major format
    tl.store(output_col_major_ptr + col_major_offsets, col_normalized, mask=mask)

    # Create 1D ranges for storing row and column max values
    row_indices = start_row + tl.arange(0, TILE_SIZE)
    col_indices = start_col + tl.arange(0, TILE_SIZE)

    # Create masks for valid rows and columns
    row_mask = row_indices < n_rows
    col_mask = col_indices < n_cols

    # Vasiliy - deviating from Claude here for much simpler code
    row_scale_start_ptr = row_max_abs_ptr + (pid_row * n_cols) + pid_col
    row_scale_indices = tl.arange(0, TILE_SIZE) * (n_cols // TILE_SIZE)
    # TODO(future): mask
    tl.store(row_scale_start_ptr + row_scale_indices, row_max_abs)

    # Vasiliy - deviating from Claude here for much simpler code
    col_scale_start_ptr = col_max_abs_ptr + (pid_col * n_rows) + pid_row
    col_scale_indices = tl.arange(0, TILE_SIZE) * (n_rows // TILE_SIZE)
    # TODO(future): mask
    tl.store(col_scale_start_ptr + col_scale_indices, col_max_abs)


# Function to launch the kernel
def normalize_tiled(x, tile_size=32):
    # Get tensor shape
    n_rows, n_cols = x.shape

    # Create output tensors (both row-major and column-major)
    output_row_major = torch.empty_like(x)
    output_col_major = torch.empty((n_cols, n_rows), dtype=x.dtype, device=x.device)

    # Create tensors for row-wise and column-wise maximum absolute values
    row_max_abs = torch.empty(
        n_rows, n_cols // tile_size, dtype=x.dtype, device=x.device
    )
    col_max_abs = torch.empty(
        n_cols, n_rows // tile_size, dtype=x.dtype, device=x.device
    )

    # Calculate grid dimensions based on tile size
    grid_rows = triton.cdiv(n_rows, tile_size)
    grid_cols = triton.cdiv(n_cols, tile_size)

    # Launch the kernel
    normalization_kernel[(grid_rows, grid_cols)](
        x_ptr=x,
        output_row_major_ptr=output_row_major,
        output_col_major_ptr=output_col_major,
        row_max_abs_ptr=row_max_abs,
        col_max_abs_ptr=col_max_abs,
        n_rows=n_rows,
        n_cols=n_cols,
        TILE_SIZE=tile_size,
    )

    return (
        output_row_major,
        output_col_major.t(),
        row_max_abs.reshape(-1, 1),
        col_max_abs.reshape(-1, 1),
    )


def run(
    M: int = 4096,
    K: int = 2048,
    BLOCK_SIZE: int = 32,
):
    print(f"M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"triton version: {triton.__version__}")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    scale_dim0_dim1_c = torch.compile(scale_dim0_dim1_reference)

    # reference implementation (plain PyTorch + torch.compile)
    x_d0, x_d1, amax_d0, amax_d1 = scale_dim0_dim1_c(x, BLOCK_SIZE)
    x_d0_and_back = (x_d0.reshape(-1, BLOCK_SIZE) * amax_d0).reshape(x_d0.shape)
    x_d1_and_back = (
        (x_d1.t().reshape(-1, BLOCK_SIZE) * amax_d1).reshape(x_d1.t().shape).t()
    )

    sqnr_bf16_vs_dim0_ref = compute_error(x, x_d0_and_back)
    sqnr_bf16_vs_dim1_ref = compute_error(x, x_d1_and_back)
    print(
        f"bf16 vs normalized reference sqnrs: dim0 {sqnr_bf16_vs_dim0_ref}, dim1 {sqnr_bf16_vs_dim1_ref}"
    )
    assert (
        sqnr_bf16_vs_dim0_ref > 50 and sqnr_bf16_vs_dim1_ref > 50
    ), "reference normlization numerics are incorrect"

    # basic triton kernel
    x_d0_t, x_d1_t, amax_d0_t, amax_d1_t = normalize_tiled(x, tile_size=BLOCK_SIZE)

    # ensure bitwise equivalency of outputs with reference
    torch.testing.assert_close(x_d0, x_d0_t, atol=0, rtol=0)
    torch.testing.assert_close(x_d1, x_d1_t, atol=0, rtol=0)
    torch.testing.assert_close(amax_d0, amax_d0_t, atol=0, rtol=0)
    torch.testing.assert_close(amax_d1, amax_d1_t, atol=0, rtol=0)
    print("normalized reference vs normalized triton are bitwise equivalent")

    if False:
        # for debugging
        sqnr_x_d0_ref_vs_t = compute_error(x_d0, x_d0_t)
        print("sqnr_x_d0_t", sqnr_x_d0_ref_vs_t)
        sqnr_amax_d0_vs_t = compute_error(amax_d0, amax_d0_t)
        print("sqnr_amax_d0_t", sqnr_amax_d0_vs_t)
        sqnr_x_d1_ref_vs_t = compute_error(x_d1, x_d1_t)
        print("sqnr_x_d1_t", sqnr_x_d1_ref_vs_t)
        sqnr_amax_d1_vs_t = compute_error(amax_d1, amax_d1_t)
        print("sqnr_amax_d1_t", sqnr_amax_d1_vs_t)

    # now, measure performance

    # warm up
    for _ in range(2):
        __ = scale_dim0_dim1_reference(x, BLOCK_SIZE)
    time_reference_compile_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: scale_dim0_dim1_c(x, b), x, BLOCK_SIZE
    )

    # warm up
    for _ in range(2):
        __ = normalize_tiled(x, tile_size=BLOCK_SIZE)
    time_triton_us = benchmark_cuda_function_in_microseconds(
        lambda x, b: normalize_tiled(x, tile_size=BLOCK_SIZE), x, BLOCK_SIZE
    )

    print("time_reference_compile_us", time_reference_compile_us)
    print("time_triton_us", time_triton_us)
    print("speedup", time_reference_compile_us / time_triton_us)


if __name__ == "__main__":
    fire.Fire(run)
