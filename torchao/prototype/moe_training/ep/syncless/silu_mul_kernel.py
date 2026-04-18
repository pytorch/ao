# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Fused SwiGLU (silu_mul) forward and backward Triton kernels.

These kernels accept GPU-resident ``offset`` and ``num_tokens`` tensors,
eliminating D2H synchronisation in the MoE expert forward/backward.

``silu_mul_fw``
    Reads ``num_tokens`` rows of ``[h1 | h3]`` from a shared saved-activations
    buffer at a GPU-determined offset, computes ``silu(h1) * h3``, and writes
    the result (zero-padded to ``sym_mem_buffer_rows``) into a new tensor.

``silu_mul_bw``
    Re-reads the saved ``[h1 | h3]``, recomputes the forward, and produces
    both the recomputed ``h`` (for w2 wgrad) and the SwiGLU gradient
    ``grad_h13`` (for w13 dgrad/wgrad).  Writes directly into pre-allocated
    output buffers.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torchao.prototype.moe_training.ep.syncless.mxfp8_kernels import (
    _calculate_scale_rceil,
)


@triton.jit
def _silu_mul_fw_kernel(
    input_ptr,
    output_ptr,
    offset_ptr,
    num_tokens_ptr,
    hidden_dim: tl.constexpr,
    input_stride_row: tl.constexpr,
    output_stride_row: tl.constexpr,
    sym_mem_buffer_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute silu(h1) * h3 for valid rows; zero-fill padding rows."""
    pid = tl.program_id(0)
    offset = tl.load(offset_ptr).to(tl.int64)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)

    elem_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = elem_ids // hidden_dim
    col = elem_ids % hidden_dim

    out_mask = (row < sym_mem_buffer_rows) & (col < hidden_dim)
    valid_mask = row < num_tokens

    # Load h1 and h3 from input_buffer[offset + row, col] and [offset + row, hidden_dim + col]
    input_row = offset + row
    h1_off = input_row * input_stride_row + col
    h3_off = input_row * input_stride_row + hidden_dim + col

    h1 = tl.load(input_ptr + h1_off, mask=valid_mask & out_mask, other=0.0)
    h3 = tl.load(input_ptr + h3_off, mask=valid_mask & out_mask, other=0.0)

    # silu(h1) = h1 * sigmoid(h1), keep everything in float32 for precision
    h1_f32 = h1.to(tl.float32)
    h3_f32 = h3.to(tl.float32)
    silu_h1_f32 = h1_f32 * tl.sigmoid(h1_f32)
    result = (silu_h1_f32 * h3_f32).to(h1.dtype)

    out_off = row * output_stride_row + col
    tl.store(output_ptr + out_off, result, mask=out_mask)


@triton_op("torchao::silu_mul_fw", mutates_args={})
def silu_mul_fw(
    input_buffer: torch.Tensor,
    saved_activation_buffer_offset: torch.Tensor,
    num_tokens: torch.Tensor,
    sym_mem_buffer_rows: int,
) -> torch.Tensor:
    """Fused SwiGLU forward: ``silu(h1) * h3`` with GPU-resident offset.

    Args:
        input_buffer: ``(saved_activations_buffer_rows, 2*hidden_dim)`` BF16
            shared saved-activations buffer.
        saved_activation_buffer_offset: Scalar int64 GPU tensor — start row in ``input_buffer``.
        num_tokens: Scalar int64 GPU tensor — number of valid rows.
        sym_mem_buffer_rows: Total output rows (Python int).

    Returns:
        ``(sym_mem_buffer_rows, hidden_dim)`` BF16 tensor.
    """
    hidden_dim = input_buffer.shape[1] // 2
    output = torch.empty(
        sym_mem_buffer_rows,
        hidden_dim,
        device=input_buffer.device,
        dtype=input_buffer.dtype,
    )

    total_elems = sym_mem_buffer_rows * hidden_dim
    BLOCK_SIZE = 1024
    grid = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    wrap_triton(_silu_mul_fw_kernel)[grid](
        input_buffer,
        output,
        saved_activation_buffer_offset,
        num_tokens,
        hidden_dim=hidden_dim,
        input_stride_row=input_buffer.stride(0),
        output_stride_row=output.stride(0),
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@triton.jit
def _silu_mul_fw_mxfp8_kernel(
    input_ptr,
    output_data_ptr,
    output_scales_ptr,
    offset_ptr,
    num_tokens_ptr,
    hidden_dim: tl.constexpr,
    input_stride_row: tl.constexpr,
    output_data_stride_row: tl.constexpr,
    sym_mem_buffer_rows,
    n_scale_col_blocks: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU forward + MXFP8 quantization with blocked scale layout.

    Computes silu(h1) * h3, quantizes to MXFP8, and writes FP8 data + blocked
    E8M0 scales directly. Eliminates the BF16 intermediate and separate scale
    rearrangement kernel.

    Grid: (cdiv(sym_mem_buffer_rows, TILE_M), cdiv(hidden_dim, TILE_K))
    """
    SCALE_BLOCKS_PER_TILE: tl.constexpr = TILE_K // SCALE_BLOCK_SIZE
    BLOCK_ROWS: tl.constexpr = TILE_M
    BLOCK_COLS: tl.constexpr = SCALE_BLOCKS_PER_TILE
    OUT_STRIDE_PER_BLOCK: tl.constexpr = BLOCK_ROWS * BLOCK_COLS
    OUT_STRIDE_PER_ROW_OF_BLOCKS: tl.constexpr = (
        BLOCK_ROWS * BLOCK_COLS * n_scale_col_blocks
    )

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offset = tl.load(offset_ptr).to(tl.int64)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)

    row_offs = pid_row * TILE_M + tl.arange(0, TILE_M)
    col_offs = pid_col * TILE_K + tl.arange(0, TILE_K)

    valid_mask = row_offs < num_tokens
    col_mask = col_offs < hidden_dim

    # Load h1 and h3 from input_buffer[offset + row, col] and [offset + row, hidden_dim + col]
    input_rows = offset + row_offs
    h1_offs = input_rows[:, None] * input_stride_row + col_offs[None, :]
    h3_offs = input_rows[:, None] * input_stride_row + hidden_dim + col_offs[None, :]

    load_mask = valid_mask[:, None] & col_mask[None, :]
    h1 = tl.load(input_ptr + h1_offs, mask=load_mask, other=0.0)
    h3 = tl.load(input_ptr + h3_offs, mask=load_mask, other=0.0)

    # Compute silu(h1) * h3 in f32, then cast to BF16 to match the
    # 2-step path (silu_mul_fw outputs BF16, then triton_to_mxfp8_dim0
    # quantizes from BF16). The BF16 round-trip ensures identical numerics.
    h1_f32 = h1.to(tl.float32)
    h3_f32 = h3.to(tl.float32)
    silu_h1_f32 = h1_f32 * tl.sigmoid(h1_f32)
    result_bf16 = (silu_h1_f32 * h3_f32).to(tl.bfloat16)

    # Zero out invalid rows
    result_bf16 = tl.where(valid_mask[:, None], result_bf16, 0.0)

    # Quantize to MXFP8: reshape to groups of SCALE_BLOCK_SIZE (32)
    # Cast back to f32 for scale computation (matches triton_to_mxfp8_dim0)
    result_f32 = result_bf16.to(tl.float32)
    result_r = tl.reshape(
        result_f32, (TILE_M * SCALE_BLOCKS_PER_TILE, SCALE_BLOCK_SIZE)
    )
    abs_result_r = tl.abs(result_r)

    # Compute E8M0 scale per block using rceil PTX path
    descale_fp, scale_e8m0 = _calculate_scale_rceil(abs_result_r, 1, True)

    # Multiply by reciprocal scale and cast to FP8
    quantized_r = result_r * descale_fp[:, None]
    quantized = tl.reshape(quantized_r, (TILE_M, TILE_K))
    quantized_fp8 = quantized.to(tl.float8e4nv)

    # Store FP8 data in row-major layout
    data_offs = row_offs[:, None] * output_data_stride_row + col_offs[None, :]
    store_mask = (row_offs[:, None] < sym_mem_buffer_rows) & (
        col_offs[None, :] < hidden_dim
    )
    tl.store(output_data_ptr + data_offs, quantized_fp8, mask=store_mask)

    # Apply blocked layout swizzle to scales and store directly
    scale_2d = tl.reshape(scale_e8m0, (BLOCK_ROWS, BLOCK_COLS))

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    r_div_32 = rows // 32
    r_mod_32 = rows % 32
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS,))
    scales_flat = tl.reshape(scale_2d, (BLOCK_ROWS * BLOCK_COLS,))

    block_offset = (
        pid_row * OUT_STRIDE_PER_ROW_OF_BLOCKS + pid_col * OUT_STRIDE_PER_BLOCK
    )
    tl.store(output_scales_ptr + block_offset + dest_indices_flat, scales_flat)


@triton_op("torchao::silu_mul_fw_mxfp8", mutates_args={})
def silu_mul_fw_mxfp8(
    input_buffer: torch.Tensor,
    saved_activation_buffer_offset: torch.Tensor,
    num_tokens: torch.Tensor,
    sym_mem_buffer_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU forward + MXFP8 quantization with blocked scale layout.

    Computes ``silu(h1) * h3`` and quantizes directly to MXFP8 with blocked
    E8M0 scales, eliminating the BF16 intermediate tensor and separate scale
    rearrangement kernel.

    Args:
        input_buffer: ``(saved_activations_buffer_rows, 2*hidden_dim)`` BF16
            shared saved-activations buffer.
        saved_activation_buffer_offset: Scalar int64 GPU tensor — start row in ``input_buffer``.
        num_tokens: Scalar int64 GPU tensor — number of valid rows.
        sym_mem_buffer_rows: Total output rows (Python int).

    Returns:
        output_e4m3: ``(sym_mem_buffer_rows, hidden_dim)`` float8_e4m3fn row-major.
        output_scales_blocked: ``(padded_rows * padded_scale_cols,)`` float8_e8m0fnu
            in tcgen05 blocked layout.
    """
    hidden_dim = input_buffer.shape[1] // 2
    block_size = 32

    # Allocate FP8 data output
    output_data = torch.empty(
        sym_mem_buffer_rows,
        hidden_dim,
        device=input_buffer.device,
        dtype=torch.float8_e4m3fn,
    )

    # Compute blocked scale layout dimensions
    scale_cols = hidden_dim // block_size
    BLOCK_ROWS, BLOCK_COLS = 128, 4
    n_row_blocks = (sym_mem_buffer_rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    n_col_blocks = (scale_cols + BLOCK_COLS - 1) // BLOCK_COLS
    padded_rows = n_row_blocks * BLOCK_ROWS
    padded_scale_cols = n_col_blocks * BLOCK_COLS

    # Allocate blocked scale output
    output_scales = torch.empty(
        padded_rows * padded_scale_cols,
        device=input_buffer.device,
        dtype=torch.uint8,
    )

    TILE_M = 128
    TILE_K = 128

    grid = (
        (sym_mem_buffer_rows + TILE_M - 1) // TILE_M,
        (hidden_dim + TILE_K - 1) // TILE_K,
    )

    wrap_triton(_silu_mul_fw_mxfp8_kernel)[grid](
        input_buffer,
        output_data,
        output_scales,
        saved_activation_buffer_offset,
        num_tokens,
        hidden_dim=hidden_dim,
        input_stride_row=input_buffer.stride(0),
        output_data_stride_row=output_data.stride(0),
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        n_scale_col_blocks=n_col_blocks,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        SCALE_BLOCK_SIZE=block_size,
    )

    return output_data, output_scales.view(torch.float8_e8m0fnu)


@triton.jit
def _silu_mul_bw_kernel(
    h13_buffer_ptr,
    grad_h_ptr,
    h_out_ptr,
    grad_h13_out_ptr,
    offset_ptr,
    num_tokens_ptr,
    hidden_dim: tl.constexpr,
    h13_stride_row: tl.constexpr,
    grad_h_stride_row: tl.constexpr,
    h_out_stride_row: tl.constexpr,
    grad_h13_stride_row: tl.constexpr,
    sym_mem_buffer_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Recompute SwiGLU forward + compute backward for valid rows; zero-fill padding."""
    pid = tl.program_id(0)
    offset = tl.load(offset_ptr).to(tl.int64)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)

    elem_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = elem_ids // hidden_dim
    col = elem_ids % hidden_dim

    out_mask = (row < sym_mem_buffer_rows) & (col < hidden_dim)
    valid_mask = row < num_tokens

    # Load h1, h3 from h13_buffer[offset + row, :]
    input_row = offset + row
    h1_off = input_row * h13_stride_row + col
    h3_off = input_row * h13_stride_row + hidden_dim + col

    h1 = tl.load(h13_buffer_ptr + h1_off, mask=valid_mask & out_mask, other=0.0)
    h3 = tl.load(h13_buffer_ptr + h3_off, mask=valid_mask & out_mask, other=0.0)

    # Recompute forward — must match _silu_mul_fw_kernel exactly so that
    # h_out is bit-identical to the forward's h (used for grad_w2 wgrad).
    h1_f32 = h1.to(tl.float32)
    h3_f32 = h3.to(tl.float32)
    sig_h1 = tl.sigmoid(h1_f32)
    silu_h1_f32 = h1_f32 * sig_h1
    # Keep everything in float32 until final cast for precision
    h = (silu_h1_f32 * h3_f32).to(h1.dtype)

    # Load grad_h[row, col]
    grad_h_off = row * grad_h_stride_row + col
    grad_h_val = tl.load(grad_h_ptr + grad_h_off, mask=valid_mask & out_mask, other=0.0)

    # SwiGLU backward — keep in float32 for gradient precision
    grad_h_f32 = grad_h_val.to(tl.float32)
    dsilu = sig_h1 + h1_f32 * sig_h1 * (1.0 - sig_h1)
    grad_h1 = (grad_h_f32 * h3_f32 * dsilu).to(h1.dtype)
    grad_h3 = (grad_h_f32 * silu_h1_f32).to(h1.dtype)

    # Write outputs - explicitly zero-fill rows beyond num_tokens
    h_out_off = row * h_out_stride_row + col
    tl.store(h_out_ptr + h_out_off, tl.where(valid_mask, h, 0.0), mask=out_mask)

    grad_h1_off = row * grad_h13_stride_row + col
    grad_h3_off = row * grad_h13_stride_row + hidden_dim + col
    tl.store(
        grad_h13_out_ptr + grad_h1_off,
        tl.where(valid_mask, grad_h1, 0.0),
        mask=out_mask,
    )
    tl.store(
        grad_h13_out_ptr + grad_h3_off,
        tl.where(valid_mask, grad_h3, 0.0),
        mask=out_mask,
    )


@triton_op("torchao::silu_mul_bw", mutates_args={})
def silu_mul_bw(
    swiglu_input: torch.Tensor,  # h13
    grad_h: torch.Tensor,
    saved_activation_buffer_offset: torch.Tensor,
    num_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU backward: recompute forward + compute gradients.

    Reads `num_tokens` rows of `[h1 | h3]` from `swiglu_input` at
    `buffer_offset`, recomputes `h = silu(h1) * h3`, and computes
    grad_h13.

    Rows beyond ``num_tokens`` are zero-filled by the kernel.

    Args:
        swiglu_input: `(saved_activations_buffer_rows, 2*hidden_dim)` BF16.
        grad_h: `(sym_mem_buffer_rows, hidden_dim)`` BF16 — w2 dgrad output.
        saved_activation_buffer_offset: Scalar int64 GPU tensor — row offset into `h13_buffer`.
        num_tokens: Scalar int64 GPU tensor — valid row count.

    Returns:
        h: `(sym_mem_buffer_rows, hidden_dim)`  - recomputed forward 'h = silu(h1) * h3'
        grad_h13: `(sym_mem_buffer_rows, 2*hidden_dim)`
    """
    hidden_dim = swiglu_input.shape[1] // 2
    sym_mem_buffer_rows = grad_h.shape[0]

    # preallocate outputs
    h_out = torch.empty(
        sym_mem_buffer_rows, hidden_dim, device=grad_h.device, dtype=torch.bfloat16
    )
    grad_h13_out = torch.empty(
        sym_mem_buffer_rows,
        2 * hidden_dim,
        device=grad_h.device,
        dtype=torch.bfloat16,
    )

    # Only process num_tokens rows; upper-bound with sym_mem_buffer_rows
    total_elems = sym_mem_buffer_rows * hidden_dim
    BLOCK_SIZE = 1024
    grid = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    wrap_triton(_silu_mul_bw_kernel)[grid](
        swiglu_input,
        grad_h,
        h_out,
        grad_h13_out,
        saved_activation_buffer_offset,
        num_tokens,
        hidden_dim=hidden_dim,
        h13_stride_row=swiglu_input.stride(0),
        grad_h_stride_row=grad_h.stride(0),
        h_out_stride_row=h_out.stride(0),
        grad_h13_stride_row=grad_h13_out.stride(0),
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return h_out, grad_h13_out
