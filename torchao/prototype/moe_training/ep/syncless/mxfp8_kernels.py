# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Fused MXFP8 dequant-requant kernel for wgrad GEMMs.

Reads FP8 E4M3 data with 1×32 row-major scaling from a saved-activations
buffer, dequantizes to FP32, then requantizes with 32×1 column scaling to
produce column-major FP8 output suitable for MXFP8 wgrad GEMMs.

This fused kernel avoids materializing a BF16 intermediate tensor, saving
memory bandwidth and enabling the wgrad GEMM to run at FP8 throughput
(~2× faster than BF16 on B200).
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
def _calculate_reciprocal_scale(scale_e8m0_biased):
    """Calculate reciprocal of E8M0 scale (copied from mx_formats/kernels.py)."""
    FP32_MANTISSA_BITS: tl.constexpr = 23
    descale_fp = tl.where(
        scale_e8m0_biased == 255,
        float("nan"),
        tl.where(
            scale_e8m0_biased == 254,
            2**-127,
            tl.where(
                scale_e8m0_biased == 0,
                1.0,
                ((254 - scale_e8m0_biased).to(tl.int32) << FP32_MANTISSA_BITS).to(
                    tl.float32, bitcast=True
                ),
            ),
        ),
    )
    return descale_fp


@triton.jit
def _calculate_scale_rceil(x, axis, USE_PTX: tl.constexpr):
    """Calculate E8M0 scale and reciprocal (copied from mx_formats/kernels.py)."""
    e8m0_exponent_bias = 127
    max_abs = tl.max(x, axis=axis)
    nan_mask = x != x
    has_nan_per_axis = tl.max(nan_mask, axis=axis)
    max_abs = tl.where(has_nan_per_axis > 0, float("nan"), max_abs)
    F8E4M3_MAX_RCP: tl.constexpr = 1.0 / 448.0
    scale_input = max_abs * F8E4M3_MAX_RCP
    if USE_PTX:
        scale_e8m0_biased = tl.inline_asm_elementwise(
            asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            constraints="=h,r",
            args=[scale_input.to(tl.float32, bitcast=False)],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)
    else:
        scale_e8m0_unbiased = tl.clamp(
            tl.ceil(tl.log2(scale_input)),
            min=-1 * e8m0_exponent_bias,
            max=e8m0_exponent_bias,
        )
        scale_e8m0_biased = (scale_e8m0_unbiased + 127).to(tl.uint8)
    descale_fp = _calculate_reciprocal_scale(scale_e8m0_biased)
    return descale_fp, scale_e8m0_biased


def _get_dequant_requant_autotune_configs():
    results = []
    for TILE_M in (64, 128, 256):
        for TILE_K in (64, 128, 256):
            for num_warps in (4, 8):
                for num_stages in (2,):
                    results.append(
                        triton.Config(
                            {"TILE_M": TILE_M, "TILE_K": TILE_K},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return results


@triton.autotune(
    configs=_get_dequant_requant_autotune_configs(),
    key=["dim", "sym_mem_buffer_rows"],
)
@triton.jit
def _mxfp8_dequant_requant_col_major_kernel(
    # Input pointers
    e4m3_data_ptr,
    e8m0_scales_ptr,
    # Output pointers
    out_data_ptr,
    out_scales_ptr,
    # GPU-resident count and output offset
    num_tokens_ptr,
    out_offset_ptr,
    # Strides
    input_stride_row: tl.constexpr,
    scale_input_stride_row: tl.constexpr,
    out_data_stride_row: tl.constexpr,
    out_scales_stride_row: tl.constexpr,
    # Dimensions
    dim: tl.constexpr,
    scale_dim: tl.constexpr,
    sym_mem_buffer_rows: tl.constexpr,
    out_scale_cols: tl.constexpr,
    # Constants
    SCALE_BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Fused dequant 1×32 -> requant 32×1 kernel.

    Reads FP8 data from a contiguous input tensor with 1×32 row-major
    scaling, dequantizes to FP32, requantizes with 32×1 column scaling,
    and writes column-major FP8 output at out_offset position in the
    output buffer.

    Early-exits for tile blocks beyond ``num_tokens``.

    Grid: (cdiv(sym_mem_buffer_rows, TILE_M), cdiv(dim, TILE_K))
    """
    BLOCKS_PER_M_TILE: tl.constexpr = TILE_M // SCALE_BLOCK_SIZE
    SCALE_BLOCKS_PER_K_TILE: tl.constexpr = TILE_K // SCALE_BLOCK_SIZE

    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)
    out_offset = tl.load(out_offset_ptr).to(tl.int64)

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Early-exit: skip blocks entirely beyond valid tokens.
    if pid_row * TILE_M >= num_tokens:
        return

    row_offs = pid_row * TILE_M + tl.arange(0, TILE_M)
    col_offs = pid_col * TILE_K + tl.arange(0, TILE_K)

    valid_token_mask = row_offs < num_tokens
    col_mask = col_offs < dim

    # Load e4m3 data from input[row, col]
    data_offsets = row_offs[:, None] * input_stride_row + col_offs[None, :]
    e4m3_block = tl.load(
        e4m3_data_ptr + data_offsets,
        mask=valid_token_mask[:, None] & col_mask[None, :],
        other=0.0,
    )

    # Load e8m0 scales from input[row, col // 32]
    scale_col_offs = pid_col * SCALE_BLOCKS_PER_K_TILE + tl.arange(
        0, SCALE_BLOCKS_PER_K_TILE
    )
    scale_offsets = row_offs[:, None] * scale_input_stride_row + scale_col_offs[None, :]
    e8m0_block = tl.load(
        e8m0_scales_ptr + scale_offsets,
        mask=valid_token_mask[:, None] & (scale_col_offs[None, :] < scale_dim),
        other=0,
    )

    # Dequant to FP32
    e4m3_r = e4m3_block.reshape(TILE_M * SCALE_BLOCKS_PER_K_TILE, SCALE_BLOCK_SIZE)
    e8m0_r = e8m0_block.reshape(TILE_M * SCALE_BLOCKS_PER_K_TILE, 1)
    fp32_scale = _e8m0_to_fp32(e8m0_r)
    fp32_block = e4m3_r.to(tl.float32) * fp32_scale
    fp32_block = fp32_block.reshape(TILE_M, TILE_K)

    # Transpose (TILE_M, TILE_K) -> (TILE_K, TILE_M)
    fp32_block_t = tl.trans(fp32_block)

    # Requant along M (axis=1 after transpose)
    fp32_t_r = fp32_block_t.reshape(TILE_K * BLOCKS_PER_M_TILE, SCALE_BLOCK_SIZE)
    abs_t_r = tl.abs(fp32_t_r)

    descale_fp, scale_e8m0 = _calculate_scale_rceil(abs_t_r, 1, True)

    requanted_t_r = fp32_t_r * descale_fp[:, None]
    requanted_t = tl.reshape(requanted_t_r, TILE_K, TILE_M)
    requanted_t_fp8 = requanted_t.to(tl.float8e4nv)

    # Store output data in (K, M) row-major layout, shifted by out_offset
    out_row_offs = out_offset + row_offs
    out_row_mask = col_offs[:, None] < dim
    out_col_mask = row_offs[None, :] < sym_mem_buffer_rows
    out_mask = out_row_mask & out_col_mask

    out_offsets = col_offs[:, None] * out_data_stride_row + out_row_offs[None, :]
    tl.store(out_data_ptr + out_offsets, requanted_t_fp8, mask=out_mask)

    # Store output scales in (K, M//32) row-major layout, shifted by out_offset // SCALE_BLOCK_SIZE
    scale_e8m0_2d = scale_e8m0.reshape(TILE_K, BLOCKS_PER_M_TILE)

    scale_row_offs = col_offs
    local_scale_col_offs = pid_row * BLOCKS_PER_M_TILE + tl.arange(0, BLOCKS_PER_M_TILE)
    out_scale_col_offs = out_offset // SCALE_BLOCK_SIZE + local_scale_col_offs

    scale_mask = (scale_row_offs[:, None] < dim) & (
        local_scale_col_offs[None, :] < out_scale_cols
    )
    scale_offsets_out = (
        scale_row_offs[:, None] * out_scales_stride_row + out_scale_col_offs[None, :]
    )
    tl.store(out_scales_ptr + scale_offsets_out, scale_e8m0_2d, mask=scale_mask)


@triton_op(
    "torchao::mxfp8_dequant_requant_col_major",
    mutates_args={"out_data", "out_scales"},
)
def mxfp8_dequant_requant_col_major(
    e4m3_data: torch.Tensor,
    e8m0_scales: torch.Tensor,
    num_tokens: torch.Tensor,
    sym_mem_buffer_rows: int,
    out_data: torch.Tensor,
    out_scales: torch.Tensor,
    out_offset: torch.Tensor,
    scale_block_size: int = 32,
) -> None:
    """Fused dequant 1×32 -> requant 32×1 with GPU-resident count/offset.

    Takes contiguous FP8 data + E8M0 scales (quantized with 1×32
    row-major scaling along dim) and produces column-major FP8 data with
    32×1 column scaling along M, written into preallocated output buffers
    at a GPU-resident output offset.

    Args:
        e4m3_data: (M, dim) float8_e4m3fn contiguous input data
        e8m0_scales: (M, dim//32) uint8 input scales
        num_tokens: Scalar int64 GPU tensor — number of valid rows
        sym_mem_buffer_rows: Number of output M rows to process (Python int, % 32 == 0)
        out_data: (dim, out_total_cols) float8_e4m3fn — preallocated output data buffer
        out_scales: (dim, out_total_scale_cols) uint8 — preallocated output scales buffer
        out_offset: Scalar int64 GPU tensor — column offset in output buffers
        scale_block_size: Scale block size (must be 32)
    """
    assert scale_block_size == 32, "scale_block_size must be 32"
    assert (
        sym_mem_buffer_rows % scale_block_size == 0
    ), f"sym_mem_buffer_rows ({sym_mem_buffer_rows}) must be divisible by {scale_block_size}"

    dim = e4m3_data.shape[1]
    scale_dim = e8m0_scales.shape[1]
    out_scale_cols = sym_mem_buffer_rows // scale_block_size

    grid = lambda META: (
        triton.cdiv(sym_mem_buffer_rows, META["TILE_M"]),
        triton.cdiv(dim, META["TILE_K"]),
    )

    wrap_triton(_mxfp8_dequant_requant_col_major_kernel)[grid](
        e4m3_data,
        e8m0_scales.to(torch.uint8),
        out_data,
        out_scales,
        num_tokens,
        out_offset,
        input_stride_row=e4m3_data.stride(0),
        scale_input_stride_row=e8m0_scales.stride(0),
        out_data_stride_row=out_data.stride(0),
        out_scales_stride_row=out_scales.stride(0),
        dim=dim,
        scale_dim=scale_dim,
        sym_mem_buffer_rows=sym_mem_buffer_rows,
        out_scale_cols=out_scale_cols,
        SCALE_BLOCK_SIZE=scale_block_size,
    )


@triton_op(
    "torchao::triton_scale_blocked_layout_output_saved_activation_buffer",
    mutates_args=(),
)
def triton_scale_blocked_layout_output_saved_activation_buffer(
    scales_buffer: torch.Tensor,
    input_col_offset: torch.Tensor,
    num_scale_cols: int = -1,
    scale_block_size: int = 32,
) -> torch.Tensor:
    """Identical to ``triton_mx_block_rearrange`` from mx_formats/kernels.py,
    but reads from ``scales_buffer`` at a GPU-resident input column offset.

    Zero D2H sync — the offset is a GPU tensor.

    Args:
        scales_buffer: (rows, total_scale_cols) uint8 or e8m0 scale buffer.
        input_col_offset: Scalar int64 GPU tensor — raw token offset
            (divided by scale_block_size inside the kernel).
        num_scale_cols: Number of scale columns to process. If -1,
            defaults to ``scales_buffer.shape[1]``.
        scale_block_size: MXFP8 block size (default 32).

    Returns:
        Rearranged tensor in block-scaled swizzle format (flat uint8).
    """
    assert scales_buffer.ndim == 2
    assert scales_buffer.element_size() == 1

    rows = scales_buffer.shape[0]
    cols = scales_buffer.shape[1]

    if num_scale_cols < 0:
        num_scale_cols = cols

    BLOCK_ROWS, BLOCK_COLS = 128, 4

    def _ceil_div(x, y):
        return (x + y - 1) // y

    n_row_blocks = _ceil_div(rows, BLOCK_ROWS)
    n_col_blocks = _ceil_div(num_scale_cols, BLOCK_COLS)
    padded_rows = n_row_blocks * BLOCK_ROWS
    padded_cols = n_col_blocks * BLOCK_COLS

    out = scales_buffer.new_empty(padded_rows * padded_cols, dtype=torch.uint8)

    stride_per_block = BLOCK_ROWS * BLOCK_COLS
    out_stride_per_row_of_blocks = BLOCK_ROWS * BLOCK_COLS * n_col_blocks

    grid = (n_row_blocks, n_col_blocks)
    wrap_triton(_triton_scale_swizzle_with_offset)[grid](
        scales_buffer.view(torch.uint8),
        rows,
        cols,
        out,
        scales_buffer.stride(0),
        scales_buffer.stride(1),
        input_col_offset,
        OUT_STRIDE_PER_BLOCK=stride_per_block,
        OUT_STRIDE_PER_ROW_OF_BLOCKS=out_stride_per_row_of_blocks,
        SCALE_BLOCK_SIZE=scale_block_size,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )

    return out


@triton.jit
def _triton_scale_swizzle_with_offset(
    scale_ptr,
    scale_rows,
    scale_cols,
    output_ptr,
    input_row_stride,
    input_col_stride,
    input_col_offset_ptr,
    OUT_STRIDE_PER_BLOCK: tl.constexpr,
    OUT_STRIDE_PER_ROW_OF_BLOCKS: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Identical to triton_scale_swizzle from mx_formats/kernels.py,
    but reads from an input column offset.  The offset is GPU-resident
    (zero D2H sync).

    Grid: (num_row_blocks, num_col_blocks)
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    in_col_offset = tl.load(input_col_offset_ptr).to(tl.int64) // SCALE_BLOCK_SIZE

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    start_row = pid_row * BLOCK_ROWS
    start_col = in_col_offset + pid_col * BLOCK_COLS
    global_rows = start_row + rows
    global_cols = start_col + cols

    mask = (global_rows < scale_rows) & (global_cols < scale_cols)

    input_scales = tl.load(
        scale_ptr + global_rows * input_row_stride + global_cols * input_col_stride,
        mask=mask,
        other=0.0,
    )

    r_div_32 = rows // 32
    r_mod_32 = rows % 32
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
    scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

    block_offset = (
        pid_row * OUT_STRIDE_PER_ROW_OF_BLOCKS + pid_col * OUT_STRIDE_PER_BLOCK
    )

    tl.store(
        output_ptr + block_offset + dest_indices_flat,
        scales_flat,
    )


@triton_op("torchao::mxfp8_dequant_buffer", mutates_args={"out_buffer"})
def mxfp8_dequant_buffer(
    e4m3_data_buffer: torch.Tensor,
    e8m0_scales_buffer: torch.Tensor,
    num_tokens: torch.Tensor,
    sym_mem_buffer_rows: int,
    out_buffer: torch.Tensor,
    out_offset: torch.Tensor,
    out_dtype: torch.dtype,
    scale_block_size: int = 32,
) -> torch.Tensor:
    """MXFP8 dequantization with GPU-resident num_tokens/out_offset.
    Reads FP8 data + E8M0 scales from input tensors (rows 0..num_tokens-1),
    dequantizes to out_dtype, and writes into out_buffer at out_offset.
    Rows beyond num_tokens are zero-filled.
    Args:
        e4m3_data_buffer: (rows, hidden_dim) FP8 E4M3 input data
        e8m0_scales_buffer: (rows, scale_dim) E8M0 input scales
        num_tokens: Scalar int64 GPU tensor - number of valid rows to read
        sym_mem_buffer_rows: Number of output rows to process (Python int)
        out_buffer: Pre-allocated output buffer to write into at out_offset
        out_offset: Scalar int64 GPU tensor - start row in output buffer
        out_dtype: Output dtype (torch.bfloat16 or torch.float32)
        scale_block_size: Scale block size (must be 32)
    Returns:
        out_buffer
    """
    assert scale_block_size == 32, "scale_block_size must be 32 for now"
    assert out_dtype in (
        torch.bfloat16,
        torch.float32,
    ), "out_dtype must be bf16 or fp32"
    # Get dimensions
    hidden_dim = e4m3_data_buffer.shape[1]
    scale_dim = e8m0_scales_buffer.shape[1]
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
        num_tokens,
        out_offset,
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


@triton.jit
def _mxfp8_dequant_buffer_kernel(
    e4m3_data_buffer,
    e8m0_scales_buffer,
    out_buffer,
    num_tokens_ptr,
    out_offset_ptr,
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
    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)
    out_off = tl.load(out_offset_ptr).to(tl.int64)
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
    block_offs = row_offs[:, None] * buffer_stride_row + col_offs[None, :]
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
    scale_block_offs = row_offs[:, None] * scale_stride_row + scale_col_offs[None, :]
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
    # Write to output buffer at out_offset
    out_row_offs = out_off + row_offs
    out_block_offs = out_row_offs[:, None] * out_stride_row + col_offs[None, :]
    out_mask = (row_offs < sym_mem_buffer_rows) & (col_offs < hidden_dim)
    tl.store(
        out_buffer + out_block_offs,
        tl.where(valid_mask[:, None], out_buffer_block, 0.0),
        mask=out_mask,
    )


@triton.jit
def _mxfp8_quant_and_transpose_kernel(
    # Input pointer
    input_ptr,
    # Output pointers (non-transpose)
    out_e4m3_ptr,
    out_scales_ptr,
    # Output pointers (transpose)
    out_t_e4m3_ptr,
    out_t_scales_ptr,
    # GPU-resident num_tokens
    num_tokens_ptr,
    # Dimensions
    M,
    N,
    # Strides
    input_stride_row,
    out_stride_row,
    out_t_stride_row,
    # Scale layout
    n_col_blocks,
    n_col_blocks_t,
    # Constants
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_TILE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Fused MXFP8 quantize + transpose kernel.

    Reads BF16 input once and produces:
    1. Non-transposed FP8 data + blocked E8M0 scales (1x32 row-wise)
    2. Transposed FP8 data + blocked E8M0 scales (1x32 on transposed = 32x1 on original)

    Early-exits for tile blocks beyond ``num_tokens``.

    Grid: (cdiv(M, TILE_M), cdiv(N, TILE_K))
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)

    # Early-exit: skip blocks entirely beyond valid tokens.
    if pid_row * TILE_M >= num_tokens:
        return

    row_offs = pid_row * TILE_M + tl.arange(0, TILE_M)
    col_offs = pid_col * TILE_K + tl.arange(0, TILE_K)

    row_mask = row_offs < num_tokens
    col_mask = col_offs < N

    # Step 1: Load BF16 tile -> f32
    input_offsets = row_offs[:, None] * input_stride_row + col_offs[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    data_f32 = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)

    # Step 2: Non-transpose quantization (1x32 along N)
    data_r = tl.reshape(data_f32, (TILE_M * BLOCKS_PER_TILE, SCALE_BLOCK_SIZE))
    abs_data_r = tl.abs(data_r)
    descale_fp, scale_e8m0 = _calculate_scale_rceil(abs_data_r, 1, True)
    quant_r = data_r * descale_fp[:, None]
    quant = tl.reshape(quant_r, (TILE_M, TILE_K))
    quant_fp8 = quant.to(tl.float8e4nv)

    # Store non-transpose FP8 data (only valid rows)
    out_offsets = row_offs[:, None] * out_stride_row + col_offs[None, :]
    tl.store(out_e4m3_ptr + out_offsets, quant_fp8, mask=mask)

    # Store non-transpose blocked scales with swizzle
    scale_2d = tl.reshape(scale_e8m0, (TILE_M, BLOCKS_PER_TILE))

    # Blocked scale layout: each (128, 4) block
    s_rows = tl.arange(0, BLOCK_ROWS)[:, None]
    s_cols = tl.arange(0, BLOCK_COLS)[None, :]
    r_div_32 = s_rows // 32
    r_mod_32 = s_rows % 32
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + s_cols

    dest_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS,))
    scales_flat = tl.reshape(scale_2d, (BLOCK_ROWS * BLOCK_COLS,))

    # block_offset = pid_row * (512 * n_col_blocks) + pid_col * 512
    OUT_STRIDE_PER_BLOCK: tl.constexpr = BLOCK_ROWS * BLOCK_COLS
    block_offset = (
        pid_row * (OUT_STRIDE_PER_BLOCK * n_col_blocks) + pid_col * OUT_STRIDE_PER_BLOCK
    )
    tl.store(out_scales_ptr + block_offset + dest_flat, scales_flat)

    # Step 3: Transpose quantization (1x32 along M)
    data_t = tl.trans(data_f32)  # (TILE_K, TILE_M)

    data_t_r = tl.reshape(data_t, (TILE_K * BLOCKS_PER_TILE, SCALE_BLOCK_SIZE))
    abs_data_t_r = tl.abs(data_t_r)
    descale_fp_t, scale_e8m0_t = _calculate_scale_rceil(abs_data_t_r, 1, True)
    quant_t_r = data_t_r * descale_fp_t[:, None]
    quant_t = tl.reshape(quant_t_r, (TILE_K, TILE_M))
    quant_t_fp8 = quant_t.to(tl.float8e4nv)

    # Store transpose FP8 data at swapped position: (pid_col*128, pid_row*128)
    t_row_offs = pid_col * TILE_K + tl.arange(0, TILE_K)
    t_col_offs = pid_row * TILE_M + tl.arange(0, TILE_M)
    t_row_mask = t_row_offs < N
    t_col_mask = t_col_offs < num_tokens
    t_mask = t_row_mask[:, None] & t_col_mask[None, :]

    out_t_offsets = t_row_offs[:, None] * out_t_stride_row + t_col_offs[None, :]
    tl.store(out_t_e4m3_ptr + out_t_offsets, quant_t_fp8, mask=t_mask)

    # Store transpose blocked scales with swizzle (swapped block position)
    scale_t_2d = tl.reshape(scale_e8m0_t, (TILE_K, BLOCKS_PER_TILE))
    scales_t_flat = tl.reshape(scale_t_2d, (BLOCK_ROWS * BLOCK_COLS,))

    # block_offset_t: indices swapped for transposed output
    block_offset_t = (
        pid_col * (OUT_STRIDE_PER_BLOCK * n_col_blocks_t)
        + pid_row * OUT_STRIDE_PER_BLOCK
    )
    tl.store(out_t_scales_ptr + block_offset_t + dest_flat, scales_t_flat)


@triton_op(
    "torchao::mxfp8_quant_and_transpose",
    mutates_args={"out_e4m3", "out_scales", "out_t_e4m3", "out_t_scales"},
)
def mxfp8_quant_and_transpose(
    input: torch.Tensor,
    num_tokens: torch.Tensor,
    out_e4m3: torch.Tensor,
    out_scales: torch.Tensor,
    out_t_e4m3: torch.Tensor,
    out_t_scales: torch.Tensor,
) -> None:
    """Fused MXFP8 quantize + transpose.

    Reads BF16 input (M, N) once and produces:
    1. e4m3 (M, N) + blocked E8M0 scales (1x32 row-wise)
    2. e4m3 (N, M) transposed + blocked E8M0 scales (1x32 on transposed)

    Both scale outputs are in tcgen05 blocked (128, 4) swizzle format.
    Writes into pre-allocated output buffers.  Blocks beyond ``num_tokens``
    early-exit with ~0 cost.

    Args:
        input: (M, N) BF16 tensor.
        num_tokens: Scalar int64 GPU tensor — number of valid rows.
        out_e4m3: Pre-allocated (M, N) float8_e4m3fn.
        out_scales: Pre-allocated flat uint8 blocked scales for (M, N).
        out_t_e4m3: Pre-allocated (N, M) float8_e4m3fn.
        out_t_scales: Pre-allocated flat uint8 blocked scales for (N, M).
    """
    assert input.ndim == 2
    M, N = input.shape
    SCALE_BLOCK_SIZE = 32
    TILE_M = 128
    TILE_K = 128
    BLOCK_ROWS = 128
    BLOCK_COLS = 4
    BLOCKS_PER_TILE = TILE_K // SCALE_BLOCK_SIZE  # 4

    n_row_blocks = triton.cdiv(M, TILE_M)
    n_col_blocks = triton.cdiv(N, TILE_K)

    n_col_blocks_t = triton.cdiv(M, TILE_K)

    grid = (n_row_blocks, n_col_blocks)
    wrap_triton(_mxfp8_quant_and_transpose_kernel)[grid](
        input,
        out_e4m3,
        out_scales,
        out_t_e4m3,
        out_t_scales,
        num_tokens,
        M=M,
        N=N,
        input_stride_row=input.stride(0),
        out_stride_row=out_e4m3.stride(0),
        out_t_stride_row=out_t_e4m3.stride(0),
        n_col_blocks=n_col_blocks,
        n_col_blocks_t=n_col_blocks_t,
        TILE_M=TILE_M,
        TILE_K=TILE_K,
        SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        BLOCKS_PER_TILE=BLOCKS_PER_TILE,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )


@triton_op(
    "torchao::triton_mx_block_rearrange_input_sym_mem_buffer",
    mutates_args={"out"},
)
def triton_mx_block_rearrange_input_sym_mem_buffer(
    scale_tensor: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Rearrange E8M0 scales to blocked swizzle format, bounded by num_tokens.

    Only processes ``num_tokens`` rows of the input scale tensor (GPU-resident,
    zero D2H sync).  Rows beyond num_tokens are zero-filled.  Output is padded
    to multiples of (128, 4) as required by the blocked layout.

    Writes into a pre-allocated output buffer.

    Args:
        scale_tensor: (rows, scale_cols) uint8 or float8_e8m0fnu scale tensor.
            ``rows`` may be larger than ``num_tokens`` (overallocated buffer).
        num_tokens: Scalar int64 GPU tensor — number of valid rows.
        out: Pre-allocated flat uint8 buffer for blocked swizzle output.
    """
    assert scale_tensor.ndim == 2
    assert scale_tensor.element_size() == 1

    rows = scale_tensor.shape[0]
    cols = scale_tensor.shape[1]

    BLOCK_ROWS, BLOCK_COLS = 128, 4

    n_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
    n_col_blocks = triton.cdiv(cols, BLOCK_COLS)

    stride_per_block = BLOCK_ROWS * BLOCK_COLS
    out_stride_per_row_of_blocks = BLOCK_ROWS * BLOCK_COLS * n_col_blocks

    grid = (n_row_blocks, n_col_blocks)
    wrap_triton(_scale_swizzle_num_tokens_kernel)[grid](
        scale_tensor.view(torch.uint8),
        num_tokens,
        cols,
        out,
        scale_tensor.stride(0),
        scale_tensor.stride(1),
        OUT_STRIDE_PER_BLOCK=stride_per_block,
        OUT_STRIDE_PER_ROW_OF_BLOCKS=out_stride_per_row_of_blocks,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
    )


@triton.jit
def _scale_swizzle_num_tokens_kernel(
    scale_ptr,
    num_tokens_ptr,
    scale_cols,
    output_ptr,
    input_row_stride,
    input_col_stride,
    OUT_STRIDE_PER_BLOCK: tl.constexpr,
    OUT_STRIDE_PER_ROW_OF_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Block-scale swizzle kernel bounded by GPU-resident num_tokens.

    Grid: (cdiv(num_tokens, BLOCK_ROWS), cdiv(scale_cols, BLOCK_COLS))
    Rows beyond num_tokens are zero-filled in the output.
    """
    num_tokens = tl.load(num_tokens_ptr).to(tl.int64)

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]

    start_row = pid_row * BLOCK_ROWS
    start_col = pid_col * BLOCK_COLS
    global_rows = start_row + rows
    global_cols = start_col + cols

    mask = (global_rows < num_tokens) & (global_cols < scale_cols)

    input_scales = tl.load(
        scale_ptr + global_rows * input_row_stride + global_cols * input_col_stride,
        mask=mask,
        other=0,
    )

    r_div_32 = rows // 32
    r_mod_32 = rows % 32
    dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

    dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
    scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

    block_offset = (
        pid_row * OUT_STRIDE_PER_ROW_OF_BLOCKS + pid_col * OUT_STRIDE_PER_BLOCK
    )

    tl.store(
        output_ptr + block_offset + dest_indices_flat,
        scales_flat,
    )
