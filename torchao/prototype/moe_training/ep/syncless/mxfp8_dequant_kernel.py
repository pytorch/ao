# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
MXFP8 dequantization kernel with GPU-resident offset support.

Eliminates D2H synchronisation by accepting GPU tensors for buffer offset
and token count, avoiding .item() calls in the MoE backward pass.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _e8m0_to_fp32(scale_e8m0):
    """Convert E8M0 scale to FP32 (copied from mx_formats/kernels.py)."""
    e8m0_nan_val = 255
    e8m0_exponent_bias = 127
    s_offset = scale_e8m0.to(tl.int16) - e8m0_exponent_bias
    s_fp = tl.exp2(s_offset.to(tl.float32))
    s_fp = tl.where(scale_e8m0 != e8m0_nan_val, s_fp, float("nan"))
    return s_fp.to(tl.float32)


@triton.jit
def _mxfp8_dequant_buffer_kernel(
    e4m3_data_buffer,
    e8m0_scales_buffer,
    out_buffer,
    offset_ptr,
    num_tokens_ptr,
    buffer_stride_row: tl.constexpr,
    scale_stride_row: tl.constexpr,
    out_stride_row: tl.constexpr,
    hidden_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    sym_mem_buffer_rows: tl.constexpr,
    out_dtype: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    """MXFP8 dequantization kernel with GPU-resident offset/num_tokens."""
    offset = tl.load(offset_ptr).to(tl.int64)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    SCALE_BLOCKS_PER_COL_TILE: tl.constexpr = COL_TILE_SIZE // SCALE_BLOCK_SIZE

    # Load block of e4m3 data from buffer[offset:offset+num_tokens]
    row_offs = pid_row * ROW_TILE_SIZE + tl.arange(0, ROW_TILE_SIZE)
    col_offs = pid_col * COL_TILE_SIZE + tl.arange(0, COL_TILE_SIZE)

    # Output masking
    out_mask = (row_offs < sym_mem_buffer_rows) & (col_offs < hidden_dim)
    valid_mask = row_offs < num_tokens

    # Add offset to read from correct buffer position
    buffer_row = offset + row_offs
    block_offs = buffer_row[:, None] * buffer_stride_row + col_offs[None, :]

    # Load e4m3 data only for valid tokens
    e4m3_data_block = tl.load(
        e4m3_data_buffer + block_offs,
        mask=valid_mask[:, None] & (col_offs[None, :] < hidden_dim),
        other=0.0,
    )

    # Load block of e8m0 scales from scales_buffer[offset:offset+num_tokens]
    scale_col_offs = pid_col * SCALE_BLOCKS_PER_COL_TILE + tl.arange(
        0, SCALE_BLOCKS_PER_COL_TILE
    )
    scale_block_offs = buffer_row[:, None] * scale_stride_row + scale_col_offs[None, :]

    e8m0_scale_block = tl.load(
        e8m0_scales_buffer + scale_block_offs,
        mask=valid_mask[:, None] & (scale_col_offs[None, :] < scale_dim),
        other=0,
    )

    # Dequantize: convert e8m0 scales to fp32 and multiply with e4m3 data
    e4m3_data_block_r = e4m3_data_block.reshape(
        ROW_TILE_SIZE * SCALE_BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE
    )
    e8m0_scale_block_r = e8m0_scale_block.reshape(
        ROW_TILE_SIZE * SCALE_BLOCKS_PER_COL_TILE, 1
    )
    fp32_scale = _e8m0_to_fp32(e8m0_scale_block_r)
    data_hp = e4m3_data_block_r.to(tl.float32) * fp32_scale

    # Convert to output dtype and reshape
    out_buffer_block = data_hp.to(out_dtype)
    out_buffer_block = out_buffer_block.reshape(ROW_TILE_SIZE, COL_TILE_SIZE)

    # Write to output buffer - explicit zero-fill rows beyond num_tokens
    out_block_offs = row_offs[:, None] * out_stride_row + col_offs[None, :]
    tl.store(
        out_buffer + out_block_offs,
        tl.where(valid_mask[:, None], out_buffer_block, 0.0),
        mask=out_mask,
    )


@triton_op("torchao::mxfp8_dequant_buffer", mutates_args={})
def mxfp8_dequant_buffer(
    e4m3_data_buffer: torch.Tensor,
    e8m0_scales_buffer: torch.Tensor,
    buffer_offset: torch.Tensor,
    num_tokens: torch.Tensor,
    sym_mem_buffer_rows: int,
    out_dtype: torch.dtype,
    scale_block_size: int = 32,
) -> torch.Tensor:
    """MXFP8 dequantization with GPU-resident offset/num_tokens.

    Args:
        e4m3_data_buffer: (buffer_rows, hidden_dim) FP8 E4M3 data buffer
        e8m0_scales_buffer: (buffer_rows, scale_dim) E8M0 scales buffer
        buffer_offset: Scalar int64 GPU tensor - start row in buffers
        num_tokens: Scalar int64 GPU tensor - number of rows to process
        sym_mem_buffer_rows: Total output rows (Python int)
        out_dtype: Output dtype (torch.bfloat16 or torch.float32)
        scale_block_size: Scale block size (must be 32)

    Returns:
        (sym_mem_buffer_rows, hidden_dim) tensor with first num_tokens rows
        containing dequantized data and remaining rows zero-filled
    """
    assert scale_block_size == 32, "scale_block_size must be 32 for now"
    assert out_dtype in (torch.bfloat16, torch.float32), (
        "out_dtype must be bf16 or fp32"
    )

    # Get dimensions
    hidden_dim = e4m3_data_buffer.shape[1]
    scale_dim = e8m0_scales_buffer.shape[1]

    # Create output buffer with sym_mem_buffer_rows (no .item() sync!)
    out_buffer = torch.empty(
        sym_mem_buffer_rows, hidden_dim, device=e4m3_data_buffer.device, dtype=out_dtype
    )

    # Triton dtype
    out_dtype_tl = tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32

    # Launch kernel
    ROW_TILE_SIZE = 256
    COL_TILE_SIZE = 256

    grid = (
        triton.cdiv(sym_mem_buffer_rows, ROW_TILE_SIZE),
        triton.cdiv(hidden_dim, COL_TILE_SIZE),
    )

    wrap_triton(_mxfp8_dequant_buffer_kernel)[grid](
        e4m3_data_buffer,
        e8m0_scales_buffer.to(torch.uint8),
        out_buffer,
        buffer_offset,
        num_tokens,
        buffer_stride_row=e4m3_data_buffer.stride(0),
        scale_stride_row=e8m0_scales_buffer.stride(0),
        out_stride_row=out_buffer.stride(0),
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        out_dtype=out_dtype_tl,
        SCALE_BLOCK_SIZE=scale_block_size,
        ROW_TILE_SIZE=ROW_TILE_SIZE,
        COL_TILE_SIZE=COL_TILE_SIZE,
    )

    return out_buffer
