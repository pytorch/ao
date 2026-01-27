# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from torch.utils._triton import has_triton

from torchao.prototype.custom_fp_utils import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
)
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.utils import (
    is_cuda_version_at_least,
    is_sm_at_least_100,
    torch_version_at_least,
)

logger = logging.getLogger(__name__)


def get_bits(x: torch.Tensor) -> str:
    bits_per_byte = 8
    # Numpy has a nice function to get the string representation of binary.
    # Since we are using ints as views of floats, need to specify the width
    # to avoid numpy from using two's complement for negative numbers.
    return np.binary_repr(x.cpu().numpy(), width=x.element_size() * bits_per_byte)  # noqa: E501


EBITS_F32, MBITS_F32 = 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2

SIGN_MASK_F4 = 0x8  # 1000
MANTISSA_MASK_F4 = 0x1  # 0001

SIGN_MASK_F6_E2M3 = 0x20  # 100000
MANTISSA_MASK_F6_E2M3 = 0x7  # 000111

SIGN_MASK_F6_E3M2 = 0x20  # 100000
MANTISSA_MASK_F6_E3M2 = 0x3  # 000011

ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000


def f32_to_f4_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-3 empty and
      bits 4-7 in fp4_e2m1
    """
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def f32_to_f6_e2m3_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-1 empty and
      bits 2-7 in fp6_e2m3
    """
    return _f32_to_floatx_unpacked(x, EBITS_F6_E2M3, MBITS_F6_E2M3)


def f32_to_f6_e3m2_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-1 empty and
      bits 2-7 in fp6_e3m2
    """
    return _f32_to_floatx_unpacked(x, EBITS_F6_E3M2, MBITS_F6_E3M2)


def f4_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-3 empty and bits 4-7
      containing an fp4_e2m1 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _floatx_unpacked_to_f32(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def f6_e2m3_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-1 empty and bits 2-7
      containing an fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _floatx_unpacked_to_f32(x, EBITS_F6_E2M3, MBITS_F6_E2M3)


def f6_e3m2_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-1 empty and bits 2-7
      containing an fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _floatx_unpacked_to_f32(x, EBITS_F6_E3M2, MBITS_F6_E3M2)


# pack/unpack code copy-pasted from
# https://github.com/pytorch-labs/ao/blob/main/torchao/dtypes/uint4.py


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    assert uint8_data.is_contiguous()

    shape = uint8_data.shape

    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    #
    # Note: known slow with triton
    # * currently generates two kernels with a cat in between
    # * after https://github.com/pytorch/pytorch/pull/123278 lands I
    #   verified that we get a single triton kernel, but that is even slower
    #   than the two kernels before this PR
    # * TODO add a microbenchmark of just the cast and profile this
    first_elements = (uint8_data & 0b1111).to(torch.uint8)
    second_elements = (uint8_data >> 4).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1).view(
        up_size(shape)
    )

    # trying Bert Maher's suggestion
    # 2024-04-04: this works in unit tests but is broken on LLaMa 7B FFN with
    #   ptxas /tmp/tmp84wp7lea.ptx, line 227; error   : Unexpected instruction types specified for 'sub'  # noqa: E501
    # which seems to be the same issue as https://github.com/pytorch/pytorch/issues/118589  # noqa: E501
    # TODO(later): try removing subtractions from our cast to see if we can work around  # noqa: E501
    # shift_tensor = torch.tensor([4, 0], dtype=torch.uint8, device=uint8_data.device)  # noqa: E501
    # unpacked = (uint8_data.reshape(-1)[::, None] >> shift_tensor) & 0b1111
    # unpacked = unpacked.view(up_size(shape))

    return unpacked


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] | uint8_data[1::2] << 4).view(down_size(shape))


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    @triton.jit
    def _triton_calculate_scale(x, axis, SCALING_MODE: tl.constexpr):
        # There is no good support for accessing globals from a jit'ed triton
        # function, so we redefine them here. Since this is prototype code which
        # we plan to remove after torch.compile catches up, this is fine.
        target_max_pow2 = 8
        e8m0_exponent_bias = 127
        bf16_mbits = 7
        bf16_exp_bias = 127
        fp32_mbits = 23

        # Find the maximum absolute value for each row
        max_abs = tl.max(x, axis=axis)

        # Compute e8m0 biased scale using either RCEIL or FLOOR rounding.
        if SCALING_MODE == "rceil":
            # RCEIL scaling mode using PTX instruction supported on sm100.
            # The input should be: amax / 448.0
            # where 448.0 is the max representable value in FP8 E4M3 format.
            F8E4M3_MAX_RCP: tl.constexpr = 1.0 / 448.0
            scale_input = max_abs.to(tl.float32) * F8E4M3_MAX_RCP

            # The PTX instruction outputs a packed uint16 where:
            # - high byte = E8M0 of first input (0.0 in our case)
            # - low byte = E8M0 of second input (scale_input)
            # Casting uint16 to uint8 naturally truncates to the low byte.
            scale_e8m0_biased = tl.inline_asm_elementwise(
                asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
                constraints="=h,r",
                args=[scale_input.to(tl.float32, bitcast=False)],
                dtype=tl.uint16,
                is_pure=True,
                pack=1,
            ).to(tl.uint8)
        else:
            tl.static_assert(SCALING_MODE == "floor")

            # Original floor implementation
            # Calculate the e8m0 scale by extracting the exponent (floor)
            max_abs = max_abs.to(tl.bfloat16)
            max_abs_int16 = max_abs.to(tl.int16, bitcast=True)
            extracted_pow2 = (
                (max_abs_int16 >> bf16_mbits) & 0b11111111
            ) - bf16_exp_bias
            extracted_pow2 = extracted_pow2 - target_max_pow2
            scale_e8m0_unbiased = extracted_pow2.to(tl.bfloat16)

            # Clamp to exponents that can be represented in e8m0
            # Add 1 to capture NaNs
            scale_e8m0_unbiased = tl.clamp(
                scale_e8m0_unbiased, -1 * e8m0_exponent_bias, e8m0_exponent_bias + 1
            )

            # Create the biased e8m0 representation and cast it to 8 bits
            scale_e8m0_biased = scale_e8m0_unbiased + e8m0_exponent_bias
            scale_e8m0_biased = scale_e8m0_biased.to(tl.uint8)

        # TODO(future PR): add NaN handling here,
        # https://github.com/pytorch/pytorch/pull/100572 will likely be useful to
        # get proper NaN propagation working
        # Calculate the scale in floating point.
        scale_fp = (scale_e8m0_biased.to(tl.int32) << fp32_mbits).to(
            tl.float32, bitcast=True
        )

        fp32_exp_bias = 127.0
        fp32_min_normal = tl.exp2(-fp32_exp_bias + 1)
        scale_fp = tl.clamp(scale_fp, min=fp32_min_normal, max=float("inf"))

        return scale_fp, scale_e8m0_biased

    def _get_mxfp8_quant_autotune_configs():
        # Values to sweep over here were determined by a manual
        # sweep over a small set of shapes, it's likely that this
        # can be improved in the future.
        results = []
        for ROW_TILE_SIZE in (128, 256, 512):
            # TODO: we can't use 512 for COL_TILE_SIZE.
            # This is likely a triton bug, tracked in
            # https://github.com/pytorch/ao/issues/3362
            for COL_TILE_SIZE in (128, 256):
                for num_warps in (4, 8):
                    for num_stages in (2, 3):
                        config = triton.Config(
                            {
                                "ROW_TILE_SIZE": ROW_TILE_SIZE,
                                "COL_TILE_SIZE": COL_TILE_SIZE,
                            },
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                        results.append(config)
        return results

    @triton.autotune(
        configs=_get_mxfp8_quant_autotune_configs(),
        key=["n_cols", "INNER_BLOCK_SIZE"],
    )
    @triton.jit
    def to_mxfp8_dim1_kernel(
        x_ptr,  # pointer to input tensor
        output_col_major_ptr,  # pointer to column-major output tensor (column-normalized)
        col_scale_ptr,  # pointer to store scales
        n_rows,  # number of rows in the tensor
        n_cols,  # number of columns in the tensor
        ROW_TILE_SIZE: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
        INNER_BLOCK_SIZE: tl.constexpr,  # should be 32 for MX
        SCALING_MODE: tl.constexpr,
    ):
        """
        Example tiling for n_rows==8, n_cols=8, ROW_TILE_SIZE=4, COL_TILE_SIZE=4, INNER_BLOCK_SIZE=2,
        pid_row=0, pid_col=0:

        Input (row-major)

        cols      0  1  2  3  4  5  6  7
        --------------------------------
        rows 0 |  0  1  2  3
             1 |  8  9 10 11
             2 | 16 17 18 19
             3 | 24 25 26 27
             4 |
             5 |
             6 |
             7 |

        Output (row-major of transpose), ids are from input

        cols      0  1  2  3  4  5  6  7
        --------------------------------
        rows 0 |  0  8 16 24
             1 |  1  9 17 25
             2 |  2 10 18 26
             3 |  3 11 19 27
             4 |
             5 |
             6 |
             7 |

        Output (scales), s(0, 8) means the scale used to cast elements 0 and 8

        rows           0          1  ...      4  ...       31
        ------------------------------------------------------
                  s(0, 8)  s(16, 24) ... s(1, 9) ... s(19, 27)
        """

        BLOCKS_PER_ROW_TILE: tl.constexpr = ROW_TILE_SIZE // INNER_BLOCK_SIZE

        # Get program ID
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)

        # Calculate starting row and column for this tile
        start_row = pid_row * ROW_TILE_SIZE
        start_col = pid_col * COL_TILE_SIZE

        # Create offsets for the block
        row_offsets = tl.arange(0, ROW_TILE_SIZE)
        col_offsets = tl.arange(0, COL_TILE_SIZE)

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
        # shape: (ROW_TILE_SIZE, COL_TILE_SIZE)
        x_block = tl.load(x_ptr + row_major_offsets, mask=mask)

        # Transpose dim0 and dim1
        # shape: (COL_TILE_SIZE, ROW_TILE_SIZE)
        x_block_t = tl.trans(x_block)

        # Reshape to inner tile size
        # shape: (COL_TILE_SIZE, ROW_TILE_SIZE) -> (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, INNER_BLOCK_SIZE)
        x_block_t_r = x_block_t.reshape(
            COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, INNER_BLOCK_SIZE
        )

        # Calculate the absolute values of elements in the block
        x_block_abs_t_r = tl.abs(x_block_t_r)

        # Find the maximum absolute value for each column
        # shape: (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE,)
        col_scale_r, col_scale_e8m0_r = _triton_calculate_scale(
            x_block_abs_t_r,
            axis=1,
            SCALING_MODE=SCALING_MODE,
        )

        # Divide each column by scale
        # Broadcasting col_scale to match x_block's shape
        # x_block_t_r shape (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, INNER_BLOCK_SIZE)
        # col_scale shape (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE,) -> (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, 1)
        col_normalized_t_r = x_block_t_r / col_scale_r[:, None]

        # Reshape back to original tile size
        col_normalized_t = tl.reshape(col_normalized_t_r, COL_TILE_SIZE, ROW_TILE_SIZE)

        # Undo the transpose
        col_normalized = tl.trans(col_normalized_t)

        # Quantize to float8
        col_normalized = col_normalized.to(tl.float8e4nv)

        # Store the column-normalized result in column-major format
        # TODO(future): this mask is for row-major likely need to transpose it for col-major
        tl.store(output_col_major_ptr + col_major_offsets, col_normalized, mask=mask)

        # reshape col_scale_e8m0_r to col_scale_e8m0
        # shape: (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE,) -> (COL_TILE_SIZE, BLOCKS_PER_ROW_TILE,)
        col_scale_e8m0 = col_scale_e8m0_r.reshape(COL_TILE_SIZE * BLOCKS_PER_ROW_TILE)

        col_scale_start_offsets = (
            (pid_col * COL_TILE_SIZE * (n_rows // ROW_TILE_SIZE))
            * BLOCKS_PER_ROW_TILE  # number of blocks seen so far
            + pid_row * BLOCKS_PER_ROW_TILE  # increment BLOCKS_PER_ROW_TILE
        )

        col_scale_start_ptr = col_scale_ptr + col_scale_start_offsets

        # calculate col_scale_indices
        col_scale_indices = tl.arange(0, COL_TILE_SIZE * BLOCKS_PER_ROW_TILE)

        # How many values are in all the other columns for this row_pid, need to jump
        # over them for every BLOCKS_PER_ROW_TILE values
        jump_vals_per_col = (n_rows - ROW_TILE_SIZE) // INNER_BLOCK_SIZE

        # example transformation (specifics depend on tile sizes):
        # [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 4, 5, 8, 9, 12, 13]
        col_scale_indices = col_scale_indices + (
            (col_scale_indices // BLOCKS_PER_ROW_TILE) * jump_vals_per_col
        )

        # TODO(future): mask this store
        tl.store(col_scale_start_ptr + col_scale_indices, col_scale_e8m0)

    @triton.autotune(
        configs=_get_mxfp8_quant_autotune_configs(),
        key=["n_cols", "SCALE_BLOCK_SIZE"],
    )
    @triton.jit
    def to_mxfp8_dim0_kernel(
        x_ptr,
        output_ptr,
        scale_ptr,
        n_rows,
        n_cols,
        ROW_TILE_SIZE: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,  # should be 32 for MX
        SCALING_MODE: tl.constexpr,
    ):
        """
        Quantizes a high precision tensor to mxfp8 rowwise (1x32 scaling granularity).
        """

        SCALE_BLOCKS_PER_COL_TILE: tl.constexpr = COL_TILE_SIZE // SCALE_BLOCK_SIZE

        # Get program ID
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)

        start_row = pid_row * ROW_TILE_SIZE
        start_col = pid_col * COL_TILE_SIZE
        row_offs = start_row + tl.arange(0, ROW_TILE_SIZE)[:, None]
        col_offs = start_col + tl.arange(0, COL_TILE_SIZE)[None, :]

        # Compute memory offsets for row-major layout (rows, cols)
        row_major_offsets = (row_offs * n_cols + col_offs).to(tl.int32)

        # Load the entire block in a single operation
        # shape: (ROW_TILE_SIZE, COL_TILE_SIZE)
        mask = (row_offs < n_rows) & (col_offs < n_cols)
        x_block = tl.load(x_ptr + row_major_offsets, mask=mask)

        # Reshape to inner tile size for rowwise scaling
        # shape: (ROW_TILE_SIZE, COL_TILE_SIZE) -> (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE)
        x_block_r = x_block.reshape(
            ROW_TILE_SIZE * SCALE_BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE
        )

        # Calculate the absolute values of elements in the block
        x_block_abs_r = tl.abs(x_block_r)

        # Find the maximum absolute value for each row (across columns)
        # shape: (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE,)
        scale_fp32_r, scale_e8m0_r = _triton_calculate_scale(
            x_block_abs_r, axis=1, SCALING_MODE=SCALING_MODE
        )

        # Divide each row by scale
        # Broadcasting scale to match x_block's shape
        # x_block_r shape:
        #    (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE)
        # scale[:, None] shape:
        #    (ROW_TILE_SIZE * BLOCKS_PER_COL_TILE, 1)
        scaled_data_r = x_block_r / scale_fp32_r[:, None]

        # Reshape back to original tile size
        e4m3_data_2d = tl.reshape(scaled_data_r, ROW_TILE_SIZE, COL_TILE_SIZE).to(
            tl.float8e4nv
        )

        # Store the row-normalized result in row-major format
        tl.store(output_ptr + row_major_offsets, e4m3_data_2d, mask=mask)

        # Calculate scale offsets to write to
        scales_per_row = n_cols // SCALE_BLOCK_SIZE
        scale_row_indices = (
            pid_row * ROW_TILE_SIZE + tl.arange(0, ROW_TILE_SIZE)[:, None]
        )
        scale_col_indices = (
            pid_col * SCALE_BLOCKS_PER_COL_TILE
            + tl.arange(0, SCALE_BLOCKS_PER_COL_TILE)[None, :]
        )
        scale_offsets = scale_row_indices * scales_per_row + scale_col_indices

        # Store e8m0 scales
        scale_mask = (scale_row_indices < n_rows) & (scale_col_indices < scales_per_row)
        scale_e8m0_2d = scale_e8m0_r.reshape(ROW_TILE_SIZE, SCALE_BLOCKS_PER_COL_TILE)
        tl.store(scale_ptr + scale_offsets, scale_e8m0_2d, mask=scale_mask)

    @triton_op("torchao::triton_to_mxfp8_dim0", mutates_args={})
    def triton_to_mxfp8_dim0(
        x: torch.Tensor,
        inner_block_size: int = 32,
        scaling_mode: str = "rceil",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        * `x` - input tensor, in row major memory layout
        * `inner_block_size` - size of tiles to scale across, default is 32 for MX recipes
        * `scaling_mode` - floor or rceil

        Output:
        * `output`: the `float8_e4m3fn` values of `x` cast to mxfp8 across dim0 (rowwise)
        * `scale`: the `e8m0` values of `x_scale` used to cast `x` to mxfp8 across dim0
        """
        assert x.is_contiguous(), "`x` must be contiguous"
        assert inner_block_size <= 32, "inner_block_size must be <= 32"
        assert x.dtype == torch.bfloat16, (
            f"only bfloat16 inputs are supported, got {x.dtype}"
        )
        assert scaling_mode in ("floor", "rceil"), (
            "only floor and rceil scaling modes are supported"
        )

        # Reshape tensor to 2d if necessary and get shape
        x_orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x.shape

        assert n_cols % inner_block_size == 0, (
            "columns must be divisible by inner block size"
        )

        # Create output tensors
        output = torch.empty(
            (n_rows, n_cols), dtype=torch.float8_e4m3fn, device=x.device
        )

        # Create scale tensors for rowwise scaling
        scale = torch.empty(
            (n_rows, n_cols // inner_block_size),
            dtype=torch.uint8,
            device=x.device,
        )

        # Calculate grid dimensions based on tile size
        grid = lambda META: (
            triton.cdiv(n_rows, META["ROW_TILE_SIZE"]),
            triton.cdiv(n_cols, META["COL_TILE_SIZE"]),
        )

        # Launch the kernel
        wrap_triton(to_mxfp8_dim0_kernel)[grid](
            x_ptr=x,
            output_ptr=output,
            scale_ptr=scale,
            n_rows=n_rows,
            n_cols=n_cols,
            SCALE_BLOCK_SIZE=inner_block_size,
            SCALING_MODE=scaling_mode.lower(),
        )

        # Reshape output back to original shape
        output = output.reshape(x_orig_shape)
        scale = scale.reshape(*x_orig_shape[:-1], scale.shape[-1])

        return (
            output,
            scale.view(torch.float8_e8m0fnu),
        )

    @triton_op("torchao::triton_to_mxfp8_dim1", mutates_args={})
    def triton_to_mxfp8_dim1(
        x: torch.Tensor, inner_block_size: int = 32, scaling_mode: str = "rceil"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
        * `x` - input tensor, in row major memory layout
        * `inner_block_size` - size of tiles to scale across, default is 32 for MX recipes

        Output:
        * `output_col_major`: the `float8_e4m3fn` values of `x` cast to mxfp8 across dim1
        * `col_scale`: the `e8m0` values of `x_scale` used to cast `x` to mxfp8 across dim1
        """
        assert x.is_contiguous(), "`x` must be contiguous"
        assert inner_block_size <= 32

        # Get tensor shape
        n_rows, n_cols = x.shape

        # Masking of loads and stores is not well tested yet, so for now enforce
        # shapes which do not need masking. Note that this condition depends on max values of
        # ROW_TILE_SIZE and COL_TILE_SIZE, which are autotuned above.
        # TODO(future): implement and test masking and remove this restriction
        max_row_tile_size = 128
        max_col_tile_size = 128
        assert n_rows % max_row_tile_size == 0, "unsupported"
        assert n_cols % max_col_tile_size == 0, "unsupported"

        # Create output tensors
        output_col_major = torch.empty(
            (n_cols, n_rows), dtype=torch.float8_e4m3fn, device=x.device
        )

        # Create scale tensors
        col_scale = torch.empty(
            (n_cols, n_rows // inner_block_size, 1),
            dtype=torch.uint8,
            device=x.device,
        )

        # Calculate grid dimensions based on tile size
        grid = lambda META: (
            triton.cdiv(n_rows, META["ROW_TILE_SIZE"]),
            triton.cdiv(n_cols, META["COL_TILE_SIZE"]),
        )

        # Launch the kernel
        wrap_triton(to_mxfp8_dim1_kernel)[grid](
            x_ptr=x,
            output_col_major_ptr=output_col_major,
            col_scale_ptr=col_scale,
            n_rows=n_rows,
            n_cols=n_cols,
            INNER_BLOCK_SIZE=inner_block_size,
            SCALING_MODE=scaling_mode,
        )

        return (
            output_col_major.t(),
            col_scale.view(torch.float8_e8m0fnu).squeeze(-1),
        )

    @register_sharding(torch.ops.torchao.triton_to_mxfp8_dim1.default)
    def custom_triton_to_mxfp8_dim1_sharding(x, inner_block_size=32):
        replicate = ([Replicate(), Replicate()], [Replicate(), None])
        # Note that the data is returned transposed, which is why
        # we flip the sharding dim below
        shard_dim0 = ([Shard(1), Shard(1)], [Shard(0), None])
        shard_dim1 = ([Shard(0), Shard(0)], [Shard(1), None])
        acceptable_shardings = [replicate, shard_dim0, shard_dim1]
        return acceptable_shardings

    def triton_to_mxfp8_dim1_reference(
        x_hp: torch.Tensor,
        block_size: int = 32,
        scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A reference version of `to_mxfp8_dim1`.
        """
        from torchao.prototype.mx_formats.mx_tensor import to_mx

        # cast across dim1
        x_hp_d1 = x_hp.t().contiguous()
        scale_e8m0_dim1, x_hp_d1_normalized = to_mx(
            x_hp_d1,
            torch.float8_e4m3fn,
            block_size,
            scaling_mode=scaling_mode,
        )
        scale_e8m0_dim1 = scale_e8m0_dim1.view(torch.float8_e8m0fnu)
        return (
            x_hp_d1_normalized.t(),
            scale_e8m0_dim1,
        )

    @triton_op("torchao::triton_mxfp8_dequant_dim0", mutates_args={})
    def triton_mxfp8_dequant_dim0(
        e4m3_data: torch.Tensor,
        e8m0_scales: torch.Tensor,
        out_dtype: torch.dtype,
        scale_block_size: int = 32,
    ) -> torch.Tensor:
        assert scale_block_size == 32, "scale_block_size must be 32 for now"
        assert out_dtype in (
            torch.bfloat16,
            torch.float32,
        ), "out_dtype must be bf16 or fp32"

        # Input shape must be 2D.
        orig_shape = e4m3_data.shape
        e4m3_data = e4m3_data.reshape(-1, orig_shape[-1])
        out_buffer = torch.empty_like(e4m3_data, dtype=out_dtype)
        out_dtype_tl = tl.bfloat16 if out_dtype == torch.bfloat16 else tl.float32

        grid = lambda META: (
            triton.cdiv(e4m3_data.shape[0], META["ROW_TILE_SIZE"]),
            triton.cdiv(e4m3_data.shape[1], META["COL_TILE_SIZE"]),
        )
        wrap_triton(_dequant_mxfp8_kernel)[grid](
            e4m3_data,
            e8m0_scales.to(torch.uint8),
            out_buffer,
            e4m3_data.size(0),
            e4m3_data.size(1),
            e8m0_scales.size(0),
            e8m0_scales.size(1),
            out_dtype=out_dtype_tl,
            SCALE_BLOCK_SIZE=scale_block_size,
        )
        return out_buffer.reshape(orig_shape)

    @triton.autotune(
        configs=_get_mxfp8_quant_autotune_configs(),
        key=["input_num_cols", "SCALE_BLOCK_SIZE"],
    )
    @triton.jit
    def _dequant_mxfp8_kernel(
        e4m3_data,
        e8m0_scales,
        out_buffer,
        input_num_rows,
        input_num_cols,
        scale_num_rows,
        scale_num_cols,
        out_dtype: tl.constexpr,
        SCALE_BLOCK_SIZE: tl.constexpr,
        ROW_TILE_SIZE: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)
        SCALE_BLOCKS_PER_COL_TILE: tl.constexpr = COL_TILE_SIZE // SCALE_BLOCK_SIZE

        # Load block of e4m3 data
        row_offs = pid_row * ROW_TILE_SIZE + tl.arange(0, ROW_TILE_SIZE)
        col_offs = pid_col * COL_TILE_SIZE + tl.arange(0, COL_TILE_SIZE)
        block_offs = row_offs[:, None] * input_num_cols + col_offs[None, :]
        mask = (row_offs[:, None] < input_num_rows) & (
            col_offs[None, :] < input_num_cols
        )
        e4m3_data_block = tl.load(e4m3_data + block_offs, mask=mask)

        # Load block of e8m0 scales
        scale_col_offs = pid_col * SCALE_BLOCKS_PER_COL_TILE + tl.arange(
            0, SCALE_BLOCKS_PER_COL_TILE
        )
        scale_block_offs = row_offs[:, None] * scale_num_cols + scale_col_offs[None, :]
        scale_mask = (row_offs[:, None] < scale_num_rows) & (
            scale_col_offs[None, :] < scale_num_cols
        )
        e8m0_scale_block = tl.load(e8m0_scales + scale_block_offs, mask=scale_mask)

        # Dequantize and return output
        e4m3_data_block_r = e4m3_data_block.reshape(
            ROW_TILE_SIZE * SCALE_BLOCKS_PER_COL_TILE, SCALE_BLOCK_SIZE
        )
        e8m0_scale_block_r = e8m0_scale_block.reshape(
            ROW_TILE_SIZE * SCALE_BLOCKS_PER_COL_TILE, 1
        )
        fp32_scale = _e8m0_to_fp32(e8m0_scale_block_r)
        data_hp = e4m3_data_block_r.to(tl.float32) * fp32_scale

        # Write to output buffer
        out_buffer_block = data_hp.to(out_dtype)
        out_buffer_block = out_buffer_block.reshape(ROW_TILE_SIZE, COL_TILE_SIZE)
        tl.store(out_buffer + block_offs, out_buffer_block, mask=mask)

    @triton.jit
    def _e8m0_to_fp32(scale_e8m0):
        e8m0_nan_val = 255
        e8m0_exponent_bias = 127
        s_offset = scale_e8m0.to(tl.int16) - e8m0_exponent_bias
        s_fp = tl.exp2(s_offset.to(tl.float32))
        s_fp = tl.where(scale_e8m0 != e8m0_nan_val, s_fp, float("nan"))
        return s_fp.to(tl.float32)

    @triton.jit
    def triton_scale_swizzle(
        scale_ptr,
        scale_rows,
        scale_cols,
        output_ptr,
        input_row_stride,
        input_col_stride,
        output_block_stride,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)

        rows = tl.arange(0, BLOCK_ROWS)[:, None]
        cols = tl.arange(0, BLOCK_COLS)[None, :]

        # Calculate starting row and column for this tile
        start_row = pid_row * BLOCK_ROWS
        start_col = pid_col * BLOCK_COLS
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

        # 2) Rearrange to (32, 4, 4) then to final (32, 16) coordinates
        dest_indices = r_mod_32 * 16 + r_div_32 * 4 + cols

        # Flatten
        dest_indices_flat = tl.reshape(dest_indices, (BLOCK_ROWS * BLOCK_COLS))
        scales_flat = tl.reshape(input_scales, (BLOCK_ROWS * BLOCK_COLS))

        # Calculate block offset using provided output block stride
        LOCAL_NUMEL = BLOCK_ROWS * BLOCK_COLS
        block_offset = pid_col * LOCAL_NUMEL + (pid_row * output_block_stride)

        tl.store(
            output_ptr + block_offset + dest_indices_flat,
            scales_flat,
        )

    @torch.library.custom_op("torchao::triton_mx_block_rearrange", mutates_args=())
    def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor:
        """
        Rearranges an E8M0 tensor scale to block-scaled swizzle format.

        This format is suitable for Tmem as described in NVIDIA documentation:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

        Args:
            scale_tensor: Input tensor in row-major format with 8-bit elements

        Returns:
            Rearranged tensor in block-scaled swizzle format
        """
        assert scale_tensor.element_size() == 1, (
            "Expected element size to be 1 byte (8 bits)"
        )

        rows, cols = scale_tensor.shape

        # Calculate blocks needed
        n_row_blocks = triton.cdiv(rows, 128)
        n_col_blocks = triton.cdiv(cols, 4)
        padded_rows = n_row_blocks * 128
        padded_cols = n_col_blocks * 4

        out = scale_tensor.new_empty((padded_rows, padded_cols))

        # Input stride (for row-major format)
        input_row_stride = scale_tensor.stride()[0]
        input_col_stride = scale_tensor.stride()[1]

        # We probably want handle multiple blocks per tile but for now keep it simple
        BLOCK_ROWS, BLOCK_COLS = 128, 4

        # Output block stride for the rearranged format
        output_block_stride = BLOCK_ROWS * BLOCK_COLS * (padded_cols // BLOCK_COLS)

        grid = lambda META: (
            triton.cdiv(padded_rows, BLOCK_ROWS),
            triton.cdiv(padded_cols, BLOCK_COLS),
        )

        wrap_triton(triton_scale_swizzle)[grid](
            scale_tensor.view(torch.uint8),
            rows,
            cols,
            out.view(torch.uint8),
            input_row_stride,
            input_col_stride,
            output_block_stride,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        )

        return out

    @triton.jit
    def convert_fp32_to_fp4_packed(x_pairs):
        """Convert FP32 pairs to packed FP4 format.

        This function takes tensor where consecutive values along the last dimension
        are packed together into single bytes.

        Args:
            x_pairs: [Tensor, Tensor] both w/ shapes [..., 1] where zipped last dimension contains
                    interleaved pairs of FP32 values to be packed together.

        Returns:
            Packed tensor with shape [...] (last dimension removed) where each
            element is an int8 containing 2 FP4 values:
            - First value of pair → low nibble (bits 0-3)
            - Second value of pair → high nibble (bits 4-7)

        Example:
            Input:  [128, 32, 2] containing FP32 pairs
            Output: [128, 32] containing packed FP4 bytes

        """

        x_fp4x2 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints=("=r,r,r,r,r,r,r,r,r"),
            args=x_pairs,
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

        return x_fp4x2

    # Sauce: https://github.com/gau-nernst/quantized-training
    @triton.jit
    def quantize_nvfp4_triton_kernel(
        x_ptr,
        tensor_scale_ptr,
        q_ptr,
        s_ptr,
        stride_xm,
        stride_xn,
        M,
        N,
        USE_TENSOR_SCALE: tl.constexpr,
        MASK_SCALES: tl.constexpr,
    ):
        F4_E2M1_MAX = 6.0
        F8E4M3_MAX = 448.0
        E4M3_EPS = 1.5258789e-05

        pid_m = tl.program_id(1)
        pid_n = tl.program_id(0)

        offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
        offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
        if MASK_SCALES:
            mask = (offs_m < M) & (offs_n < N)
            other = 0.0
        else:
            mask = None
            other = None
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=mask, other=other
        )  # [128, 64]
        x_blocks = x.to(tl.float32).reshape(128, 4, 16)  # [128, 4, 16]

        # Compute block-wise scales
        block_amax = tl.max(x_blocks.abs(), axis=2)  # [128, 4]

        if USE_TENSOR_SCALE:
            # Two-level scaling: quantize block scales with per-tensor scale
            tensor_scale = tl.load(tensor_scale_ptr)

            # First compute block scales
            block_scale_f32 = (block_amax / F4_E2M1_MAX).to(tl.float32)

            # Quantize the block scales with per-tensor scale
            scaled_block_scales = block_scale_f32 / tensor_scale
            scaled_block_scales = tl.clamp(scaled_block_scales, E4M3_EPS, F8E4M3_MAX)
            scales = scaled_block_scales.to(tl.float8e4nv)

            # Apply combined scale to data: per_tensor_scale * quantized_block_scale
            total_scale = tensor_scale * scales.to(tl.float32)[:, :, None]
            x_blocks = tl.div_rn(x_blocks, total_scale)
        else:
            # Single-level scaling: use block scales directly
            scales_f32 = block_amax / F4_E2M1_MAX
            scales_f32 = tl.clamp(scales_f32, E4M3_EPS, F8E4M3_MAX)
            scales = scales_f32.to(tl.float8e4nv)

            # Apply block scale to data
            total_scale = scales.to(tl.float32)[:, :, None]
            x_blocks = tl.div_rn(x_blocks, total_scale)

        # NVIDIA layout for scales
        if MASK_SCALES:
            # Create offsets for the scale dimensions (4 blocks per row)
            scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]

            # Mask out scales to 0 if we are not aligned to 128 x 64
            scales = tl.where(
                (offs_m < M) & (scale_offs_n < N // 16),
                scales,
                0.0,
            )
        packed_scales = scales.reshape(4, 32, 4).permute(1, 0, 2).reshape(32, 16)
        offs_m = tl.arange(0, 32)[:, None]
        offs_n = tl.arange(0, 16)[None, :]
        tl.store(
            s_ptr
            + (pid_m * tl.num_programs(0) + pid_n) * (32 * 16)
            + offs_m * 16
            + offs_n,
            packed_scales,
        )

        # Convert to FP4
        x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(128, 32, 2).split())
        offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
        offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
        if MASK_SCALES:
            mask = (offs_m < M) & (offs_n < N // 2)
        else:
            mask = None
        tl.store(q_ptr + offs_m * (N // 2) + offs_n, x_fp4x2, mask=mask)

    @torch.library.custom_op("ao::triton_quantize_nvfp4", mutates_args=())
    def triton_quantize_nvfp4(
        x: torch.Tensor, per_tensor_scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor to NVFP4 format.

        Args:
            x (torch.Tensor): Input tensor to be quantized.
            tensor_scale (Optional[torch.Tensor]): Per-tensor scale for two-level quantization.
                If None, uses single-level block-wise quantization only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scales tensor in swizzled layout.

        Note:
            Since VLLM does not use dyanmo guards we need to make this a custom op
            to avoid the triton kernel being invoked w/ the wrong use of `MASK_SCALES`
        """
        # reshape to 2d
        orig_leading_dims, _orig_M, orig_N = x.shape[:-2], x.shape[-2], x.shape[-1]
        x = x.reshape(-1, orig_N)

        M, N = x.shape
        # assert M % 128 == 0 and N % 64 == 0
        assert N % 16 == 0, "N must be divisible by 16 for NVFP4 quantization"

        # Calculate blocks needed
        num_scales = N // 16
        n_row_blocks = triton.cdiv(M, 128)
        n_col_blocks = triton.cdiv(num_scales, 4)
        padded_rows = n_row_blocks * 128
        padded_cols = n_col_blocks * 4

        # mask out scales to 0 if we are not aligned to 128 x 64
        MASK_SCALES = M % 128 != 0 or N % 64 != 0

        xq = x.new_empty(M, N // 2, dtype=torch.uint8)
        scales = x.new_empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn)

        grid = (triton.cdiv(N, 64), triton.cdiv(M, 128))

        if per_tensor_scale is None:
            # Don't allocate tensor, we just steal this since it won't be used in kernel
            tensor_scale_ptr = x
            use_tensor_scale = False
        else:
            tensor_scale_ptr = per_tensor_scale
            use_tensor_scale = True

        quantize_nvfp4_triton_kernel[grid](
            x,
            tensor_scale_ptr,
            xq,
            scales,
            x.stride(0),
            x.stride(1),
            M,
            N,
            USE_TENSOR_SCALE=use_tensor_scale,
            MASK_SCALES=MASK_SCALES,
        )

        # reshape back to original shape
        scales = scales.view(*orig_leading_dims, -1, padded_cols)
        xq = xq.view(*orig_leading_dims, -1, N // 2)

        return scales, xq.view(torch.uint8)

    @triton_quantize_nvfp4.register_fake
    def _(x, per_tensor_scale=None):
        M, N = x.shape
        num_scales = N // 16
        n_row_blocks = triton.cdiv(M, 128)
        n_col_blocks = triton.cdiv(num_scales, 4)
        padded_rows = n_row_blocks * 128
        padded_cols = n_col_blocks * 4

        scales = torch.empty(
            padded_rows, padded_cols, device=x.device, dtype=torch.float8_e4m3fn
        )
        xq = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
        return scales, xq

    @triton_mx_block_rearrange.register_fake
    def _(scale_tensor):
        rows, cols = scale_tensor.shape
        n_row_blocks = triton.cdiv(rows, 128)
        n_col_blocks = triton.cdiv(cols, 4)
        padded_rows = n_row_blocks * 128
        padded_cols = n_col_blocks * 4

        return scale_tensor.new_empty((padded_rows, padded_cols))

else:

    def triton_to_mxfp8_dim0(
        x: torch.Tensor,
        inner_block_size=32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("needs torch version 2.8+ and triton")

    def triton_to_mxfp8_dim1(
        x, inner_block_size=32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("needs torch version 2.8+ and triton")

    def triton_to_mxfp8_dim1_reference(
        x_hp: torch.Tensor,
        block_size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("needs torch version 2.8+ and triton")

    def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("needs torch version 2.8+ and triton")

    def triton_quantize_nvfp4(
        x: torch.Tensor, tensor_scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("needs torch version 2.8+ and triton")

    def triton_mxfp8_dequant_dim0(
        e4m3_data: torch.Tensor,
        e8m0_scales: torch.Tensor,
        out_dtype: torch.dtype,
        inner_block_size=32,
    ) -> torch.Tensor:
        raise AssertionError("needs torch version 2.8+ and triton")


mxfp8_cuda_extension_available = is_sm_at_least_100() and is_cuda_version_at_least(
    12, 8
)

if mxfp8_cuda_extension_available:
    lib = torch.library.Library("torchao", "FRAGMENT")
    lib.define(
        "mxfp8_quantize(Tensor input, bool rowwise, bool colwise, int scale_dim_x, int scale_dim_y, str fp8_format, str scaling_mode) -> (Tensor, Tensor, Tensor, Tensor)",
        tags=[torch._C.Tag.needs_fixed_stride_order],
    )

    def mxfp8_quantize_cuda(
        x: torch.Tensor,
        rowwise: bool = False,
        colwise: bool = True,
        scaling_mode: str = "floor",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes a 2D tensor to MXFP8 format using CUDA kernels.

        This is a high-level wrapper that calls the underlying CUDA kernel via
        torch.ops.torchao.mxfp8_quantize.

        Args:
            x: Input tensor to be quantized. Must be 2D with shape (rows, cols).
            rowwise: If True, compute rowwise scales.
            colwise: If True, compute colwise scales.
            scaling_mode: Scaling mode for quantization. Defaults to "floor".

        Returns:
            Tuple of (output_rowwise, output_colwise, scales_rowwise, scales_colwise)
        """
        # Input shape must be 2D.
        assert x.ndim == 2
        rows, cols = x.shape

        # Block size must be a multiple of 32.
        block_size = 32
        assert rows % block_size == 0, "rows must be a multiple of 32"
        assert cols % block_size == 0, "cols must be a multiple of 32"

        scale_dim_x = 1
        scale_dim_y = block_size
        fp8_format = "e4m3"
        output_rowwise, output_colwise, scales_rowwise, scales_colwise = (
            torch.ops.torchao.mxfp8_quantize.default(
                x,
                rowwise,
                colwise,
                scale_dim_x,
                scale_dim_y,
                fp8_format,
                scaling_mode,
            )
        )
        return output_rowwise, output_colwise, scales_rowwise, scales_colwise

    @torch.library.register_fake("torchao::mxfp8_quantize")
    def _fake_mxfp8_quantize(
        x: torch.Tensor,
        rowwise: bool,
        colwise: bool,
        scale_dim_x: int,
        scale_dim_y: int,
        fp8_format: str,
        scaling_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fake/meta implementation for mxfp8_quantize."""
        assert x.ndim == 2
        rows, cols = x.shape
        num_row_blocks = rows // 32
        num_col_blocks = cols // 32

        # rowwise
        if rowwise:
            output_rowwise = x.new_empty(rows, cols, dtype=torch.float8_e4m3fn)
            scales_rowwise = x.new_empty(
                rows, num_col_blocks, 1, dtype=torch.float8_e8m0fnu
            )
        else:
            output_rowwise = x.new_empty(0, dtype=torch.float8_e4m3fn)
            scales_rowwise = x.new_empty(0, dtype=torch.float8_e8m0fnu)

        # colwise
        if colwise:
            # column major
            output_colwise = torch.empty_strided(
                (rows, cols), (1, rows), dtype=torch.float8_e4m3fn, device=x.device
            )

            # colwise scales are written in column-major format to avoid uncoalesced global memory accesses
            scales_colwise = torch.empty_strided(
                (cols, num_row_blocks),
                (1, cols),
                dtype=torch.float8_e8m0fnu,
                device=x.device,
            )
        else:
            output_colwise = x.new_empty(0, dtype=torch.float8_e4m3fn)
            scales_colwise = x.new_empty(0, dtype=torch.float8_e8m0fnu)

        return output_rowwise, output_colwise, scales_rowwise, scales_colwise

    @register_sharding(torch.ops.torchao.mxfp8_quantize.default)
    def custom_mxfp8_quantize_cuda_dim1_sharding(
        x: torch.Tensor,
        rowwise: bool,
        colwise: bool,
        scale_dim_x: int,
        scale_dim_y: int,
        fp8_format: str,
        scaling_mode: str,
    ):
        # This function signature can be used to understand the shardings:
        # _, colwise_data, _, colwise_scales = mxfp8_quantize_cuda(x, rowwise=False, colwise=True)

        # When inputs and scale are replicated, we return a quantized output tensor (replicated).
        inputs_replicated = [None, Replicate(), None, Replicate()]
        outputs_replicated = [None, Replicate(), None, None]
        rule_for_input_replicated = (
            inputs_replicated,
            outputs_replicated,
        )

        # When inputs and scale are sharded along dim 0,
        # we return a quantized output tensor (sharded along dim1 due to transpose).
        inputs_sharded_dim0 = [None, Shard(0), None, Shard(0)]
        outputs_sharded_dim1 = [None, Shard(1), None, None]
        rule_for_input_sharded_dim0 = (inputs_sharded_dim0, outputs_sharded_dim1)

        # When inputs and scale are sharded along dim 1,
        # we return a quantized output tensor (sharded along dim0 due to transpose).
        inputs_sharded_dim1 = [None, Shard(1), None, Shard(1)]
        outputs_sharded_dim0 = [None, Shard(0), None, None]
        rule_for_input_sharded_dim1 = (inputs_sharded_dim1, outputs_sharded_dim0)

        acceptable_shardings = [
            rule_for_input_replicated,
            rule_for_input_sharded_dim0,
            rule_for_input_sharded_dim1,
        ]
        return acceptable_shardings

else:

    def mxfp8_quantize_cuda(
        x: torch.Tensor,
        rowwise: bool = False,
        colwise: bool = True,
        scaling_mode: str = "floor",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "`mxfp8_quantize_cuda` needs (1) torch 2.8+ and (2) torchao built from source on a machine with CUDA capability 10.0+. Please see https://github.com/pytorch/ao/issues/2932 for more details."
        )
