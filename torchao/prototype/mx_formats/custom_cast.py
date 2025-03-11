# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils._triton import has_triton

from torchao.prototype.custom_fp_utils import (
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

# TODO(future): if needed, make the below work on previous PyTorch versions,
# just need to hunt down the previous location of `libdevice`. An assert
# at the callsite prevents usage of this on unsupported versions.
if TORCH_VERSION_AT_LEAST_2_4 and has_triton():
    from torch._inductor.runtime.triton_helpers import libdevice

from torchao.prototype.mx_formats.constants import (
    E8M0_EXPONENT_BIAS,
    E8M0_EXPONENT_NAN_VAL,
    F4_E2M1_EXP_BIAS,
    F6_E2M3_EXP_BIAS,
    F6_E3M2_EXP_BIAS,
    F32_EXP_BIAS,
)


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


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _fp4_packed_to_bf16(
        x_packed,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
    ):
        """
        Input: a tensor of packed fp4 values
        Output: a tensor of bfloat16 values
        """

        # low-bits: original location 0:3
        # high-bits: original location 4:7
        x_low_bits = x_packed >> 4
        x_high_bits = x_packed & 0xF
        x = tl.interleave(x_low_bits, x_high_bits)

        # cast logic below
        # output = x_unpacked.to(tl.float32)

        # save the sign
        sign_f4 = x & sign_mask_f4

        # set everything to positive, will add sign back at the end
        x_pos = x ^ sign_f4

        # Special case zero
        zero_mask = x_pos == 0

        # There is only one denormal value in fp4: s001, which is 0.5 in f32
        # Special case it.
        # TODO(later): will it be faster to repeat this for all 8 positive
        # values instead of the bit manipulations?
        denormal_mask = x_pos == 1

        # calculate the new exponent and shift it to bits 2:9 of the result
        exp_biased_f4 = x_pos >> mbits_f4_e2m1
        exp_biased_f32 = exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias
        exp_biased_f32 = exp_biased_f32.to(tl.int32) << mbits_f32

        # shift the mantissa to bits 10:32 of the result
        mantissa_f4 = x_pos & mantissa_mask_f4
        mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
        output = mantissa_f32

        # combine the pieces
        result = exp_biased_f32 | mantissa_f32
        # result[zero_mask] = ZERO_BITS_F32
        result = tl.where(zero_mask, zero_bits_f32, result)
        # result[denormal_mask] = ZERO_POINT_FIVE_BITS_F32
        result = tl.where(denormal_mask, zero_point_five_bits_f32, result)

        # add sign back
        sign_f32 = sign_f4.to(tl.int32) << (
            mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1
        )
        result = result | sign_f32

        # The bit shifting above is for float32, so for now we
        # bitcast to float32 and then regular cast to bfloat16
        # TODO(later): it should be pretty easy to cast directly to bf16, just
        # need to adjust the mbits/ebits/special values. Perf impact is likely
        # to be small as we would not be chaning memory access patterns.
        output = result.to(tl.float32, bitcast=True)
        output = output.to(tl.bfloat16)
        return output

    @triton.jit
    def triton_f4_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_elements_in,
        sign_mask_f4: tl.constexpr,
        mantissa_mask_f4: tl.constexpr,
        mbits_f4_e2m1: tl.constexpr,
        ebits_f4_e2m1: tl.constexpr,
        f4_e2m1_exp_bias: tl.constexpr,
        mbits_f32: tl.constexpr,
        ebits_f32: tl.constexpr,
        f32_exp_bias: tl.constexpr,
        zero_bits_f32: tl.constexpr,
        zero_point_five_bits_f32: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        n_elements_out = n_elements_in * 2
        BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

        block_start_in = pid * BLOCK_SIZE_IN
        offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN)

        mask_in = offsets_in < n_elements_in

        # packed uint8
        x_packed = tl.load(x_ptr + offsets_in, mask=mask_in)
        output = _fp4_packed_to_bf16(
            x_packed,
            sign_mask_f4,
            mantissa_mask_f4,
            mbits_f4_e2m1,
            ebits_f4_e2m1,
            f4_e2m1_exp_bias,
            mbits_f32,
            ebits_f32,
            f32_exp_bias,
            zero_bits_f32,
            zero_point_five_bits_f32,
        )

        # set up output offsets
        block_start_out = pid * BLOCK_SIZE_OUT
        offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT)
        mask_out = offsets_out < n_elements_out

        tl.store(output_ptr + offsets_out, output, mask=mask_out)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 128}),
            triton.Config({"BLOCK_SIZE_IN": 256}),
            triton.Config({"BLOCK_SIZE_IN": 512}),
            triton.Config({"BLOCK_SIZE_IN": 1024}),
            triton.Config({"BLOCK_SIZE_IN": 2048}),
        ],
        key=["n_elements_in"],
    )
    @triton.jit
    def triton_f4_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_elements_in,
        mx_block_size: tl.constexpr,
        sign_mask_f4: tl.constexpr,
        mantissa_mask_f4: tl.constexpr,
        mbits_f4_e2m1: tl.constexpr,
        ebits_f4_e2m1: tl.constexpr,
        f4_e2m1_exp_bias: tl.constexpr,
        mbits_f32: tl.constexpr,
        ebits_f32: tl.constexpr,
        f32_exp_bias: tl.constexpr,
        zero_bits_f32: tl.constexpr,
        zero_point_five_bits_f32: tl.constexpr,
        e8m0_exponent_bias: tl.constexpr,
        e8m0_exponent_nan_val: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        n_elements_out = n_elements_in * 2
        n_elements_s = n_elements_out // 32

        BLOCK_SIZE_S: tl.constexpr = BLOCK_SIZE_IN // 16
        BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

        block_start_in = pid * BLOCK_SIZE_IN
        offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN)
        mask_in = offsets_in < n_elements_in
        # packed uint8
        x_packed = tl.load(x_ptr + offsets_in, mask=mask_in)
        output = _fp4_packed_to_bf16(
            x_packed,
            sign_mask_f4,
            mantissa_mask_f4,
            mbits_f4_e2m1,
            ebits_f4_e2m1,
            f4_e2m1_exp_bias,
            mbits_f32,
            ebits_f32,
            f32_exp_bias,
            zero_bits_f32,
            zero_point_five_bits_f32,
        )

        # load scale
        block_start_s = pid * BLOCK_SIZE_S
        offsets_s = block_start_s + tl.arange(0, BLOCK_SIZE_S)
        mask_s = offsets_s < n_elements_s
        s = tl.load(s_ptr + offsets_s, mask=mask_s)

        # create the scale in bf16
        s_offset = s.to(tl.int16) - e8m0_exponent_bias
        s_fp = libdevice.pow(2.0, s_offset).to(tl.bfloat16)
        s_fp = tl.where(s != e8m0_exponent_nan_val, s_fp, float("nan"))

        # multiply output by scale
        # TODO(later): see if manipulating the exponent instead of fp
        # multiplication is going to give a significant speedup
        output = tl.reshape(output, (BLOCK_SIZE_OUT // mx_block_size, mx_block_size))  # noqa: E501
        s_fp = tl.reshape(s_fp, (BLOCK_SIZE_S // 1, 1))
        output = output * s_fp
        output = tl.reshape(output, (BLOCK_SIZE_OUT,))

        # set up output offsets
        block_start_out = pid * BLOCK_SIZE_OUT
        offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT)
        mask_out = offsets_out < n_elements_out

        tl.store(output_ptr + offsets_out, output, mask=mask_out)

    @triton.jit
    def _fp6_packed_to_bf16(
        packed_4bits_a,
        packed_4bits_b,
        packed_2bits,
        sign_mask_f6,
        mbits_f6,
        f6_exp_bias,
        mbits_f32,
        f32_exp_bias,
    ):
        """
        Input: a tensor of packed fp6 values
        Output: a tensor of bfloat16 values
        """

        # L/R shift and combine back into uint8 with first 2 bits empty (i.e. unpacked)
        x_0 = ((packed_4bits_a >> 2) & 0x3C) | ((packed_2bits & 0xC0) >> 6)
        x_1 = ((packed_4bits_a << 2) & 0x3C) | ((packed_2bits & 0x30) >> 4)
        x_2 = ((packed_4bits_b >> 2) & 0x3C) | ((packed_2bits & 0xC) >> 2)
        x_3 = ((packed_4bits_b << 2) & 0x3C) | (packed_2bits & 0x3)

        # repeat_interleave not supported yet, see https://github.com/triton-lang/triton/issues/1426
        # instead we can interleave(interleave(4*i, 4*i+2), interleave(4*i+1, 4*i+3))
        # TODO: is there a more performant way?
        # We could stack all 4, then transpose and ravel and do it that way?
        x_02 = tl.interleave(x_0, x_2)  # [x_0_0, x_2_0, x_0_1, x_2_1, ...]
        x_13 = tl.interleave(x_1, x_3)  # [x_1_0, x_3_0, x_1_1, x_3_1, ...]
        x = tl.interleave(x_02, x_13)  # [x_0_0, x_1_0, x_2_0, x_3_0, x_0_1, ...]

        # save the sign
        sign_f6 = x & sign_mask_f6

        # set everything to positive, will add sign back at the end
        x_pos = x ^ sign_f6

        # shift the exponent and mantissa
        result = x_pos.to(tl.int32) << (mbits_f32 - mbits_f6)

        # add sign back
        # left shift is always 26 regardless of fp6 variant
        sign_f32 = sign_f6.to(tl.int32) << 26
        result = result | sign_f32

        # The bit shifting above is for float32, so for now we
        # bitcast to float32 and then regular cast to bfloat16
        # TODO(later): it should be pretty easy to cast directly to bf16, just
        # need to adjust the mbits/ebits/special values. Perf impact is likely
        # to be small as we would not be changing memory access patterns.
        output = result.to(tl.float32, bitcast=True)

        # Scale the fp32 exponent afterwards, handles the denorms correctly
        output *= 2.0 ** (f32_exp_bias - f6_exp_bias)

        output = output.to(tl.bfloat16)
        return output

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
        ],
        key=["n_mx_blocks"],
    )
    @triton.jit
    def triton_f6_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_mx_blocks,
        mx_block_size: tl.constexpr,
        packed_mx_block_size: tl.constexpr,
        sign_mask_f6: tl.constexpr,
        mbits_f6: tl.constexpr,
        f6_exp_bias: tl.constexpr,
        mbits_f32: tl.constexpr,
        f32_exp_bias: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE_IN

        offsets_rows = block_start + tl.arange(0, BLOCK_SIZE_IN)
        offsets_cols = tl.arange(0, packed_mx_block_size // 3)
        mask_in = (offsets_rows[:, None] < n_mx_blocks) & (
            offsets_cols[None, :] < packed_mx_block_size // 3
        )
        offsets_in = (
            offsets_rows[:, None] * packed_mx_block_size + offsets_cols[None, :]
        )

        # packed 4 x fp6 into 3 x uint8
        packed_4bits_a = tl.load(x_ptr + offsets_in, mask=mask_in, other=0)
        packed_4bits_b = tl.load(
            x_ptr + offsets_in + (packed_mx_block_size // 3), mask=mask_in, other=0
        )
        packed_2bits = tl.load(
            x_ptr + offsets_in + (2 * packed_mx_block_size // 3), mask=mask_in, other=0
        )

        output = _fp6_packed_to_bf16(
            packed_4bits_a,
            packed_4bits_b,
            packed_2bits,
            sign_mask_f6,
            mbits_f6,
            f6_exp_bias,
            mbits_f32,
            f32_exp_bias,
        )

        # set up output offsets
        offsets_rows_out = block_start + tl.arange(0, BLOCK_SIZE_IN)
        offsets_cols_out = tl.arange(0, mx_block_size)
        offsets_out = (
            offsets_rows_out[:, None] * mx_block_size + offsets_cols_out[None, :]
        )
        mask_out = (offsets_rows_out[:, None] < n_mx_blocks) & (
            offsets_cols_out[None, :] < mx_block_size
        )

        tl.store(output_ptr + offsets_out, output, mask=mask_out)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
        ],
        key=["n_mx_blocks"],
    )
    @triton.jit
    def triton_f6_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_mx_blocks,
        mx_block_size: tl.constexpr,
        packed_mx_block_size: tl.constexpr,
        sign_mask_f6: tl.constexpr,
        mbits_f6: tl.constexpr,
        f6_exp_bias: tl.constexpr,
        mbits_f32: tl.constexpr,
        f32_exp_bias: tl.constexpr,
        e8m0_exponent_bias: tl.constexpr,
        e8m0_exponent_nan_val: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE_IN

        offsets_rows = block_start + tl.arange(0, BLOCK_SIZE_IN)
        offsets_cols = tl.arange(0, packed_mx_block_size // 3)
        mask_in = (offsets_rows[:, None] < n_mx_blocks) & (
            offsets_cols[None, :] < packed_mx_block_size // 3
        )
        offsets_in = (
            offsets_rows[:, None] * packed_mx_block_size + offsets_cols[None, :]
        )

        # packed 4 x fp6 into 3 x uint8
        packed_4bits_a = tl.load(x_ptr + offsets_in, mask=mask_in, other=0)
        packed_4bits_b = tl.load(
            x_ptr + offsets_in + (packed_mx_block_size // 3), mask=mask_in, other=0
        )
        packed_2bits = tl.load(
            x_ptr + offsets_in + (2 * packed_mx_block_size // 3), mask=mask_in, other=0
        )

        output = _fp6_packed_to_bf16(
            packed_4bits_a,
            packed_4bits_b,
            packed_2bits,
            sign_mask_f6,
            mbits_f6,
            f6_exp_bias,
            mbits_f32,
            f32_exp_bias,
        )

        # load scale
        offsets_s = block_start + tl.arange(0, BLOCK_SIZE_IN)
        mask_s = offsets_s < n_mx_blocks
        s = tl.load(s_ptr + offsets_s, mask=mask_s)

        # create the scale in bf16
        s_offset = s.to(tl.float32) - e8m0_exponent_bias
        s_fp = libdevice.pow(2.0, s_offset).to(tl.bfloat16)
        s_fp = tl.where(s != e8m0_exponent_nan_val, s_fp, float("nan"))

        # multiply output by scale
        # TODO(later): see if manipulating the exponent instead of fp
        # multiplication is going to give a significant speedup
        output = tl.reshape(output, (BLOCK_SIZE_IN, mx_block_size))  # noqa: E501
        s_fp = tl.reshape(s_fp, (BLOCK_SIZE_IN // 1, 1))
        output = output * s_fp
        output = tl.reshape(output, (BLOCK_SIZE_IN, mx_block_size))

        # set up output offsets
        offsets_rows_out = block_start + tl.arange(0, BLOCK_SIZE_IN)
        offsets_cols_out = tl.arange(0, mx_block_size)
        offsets_out = (
            offsets_rows_out[:, None] * mx_block_size + offsets_cols_out[None, :]
        )
        mask_out = (offsets_rows_out[:, None] < n_mx_blocks) & (
            offsets_cols_out[None, :] < mx_block_size
        )

        tl.store(output_ptr + offsets_out, output, mask=mask_out)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
        ],
        key=["n_mx_blocks"],
    )
    @triton.jit
    def triton_pack_uint6_kernel(
        input_ptr,
        output_ptr,
        n_mx_blocks,
        MX_BLOCK_SIZE: tl.constexpr,
        PACKED_MX_BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE_IN

        # input_ptr is shape [n_mx_blocks, MX_BLOCK_SIZE]
        # Load BLOCK_SIZE rows of input_ptr
        offsets_rows = block_start + tl.arange(0, BLOCK_SIZE_IN)
        offsets_cols = tl.arange(0, MX_BLOCK_SIZE // 4)
        offsets = offsets_rows[:, None] * MX_BLOCK_SIZE + (4 * offsets_cols[None, :])
        mask = (offsets_rows[:, None] < n_mx_blocks) & (
            offsets_cols[None, :] < MX_BLOCK_SIZE // 4
        )

        # x is shape [BLOCK_SIZE, MX_BLOCK_SIZE]
        x_0 = tl.load(input_ptr + offsets, mask=mask)
        x_1 = tl.load(input_ptr + offsets + 1, mask=mask)
        x_2 = tl.load(input_ptr + offsets + 2, mask=mask)
        x_3 = tl.load(input_ptr + offsets + 3, mask=mask)

        # OR between remainder 0/1, 2/3 elements to pack 2 x first-4-bit partial representations
        # next to each other. These are the middle 4 bits of the uint8, so some gymnastics required.
        # i.e. (00abcd00 >> 2) | (00wxyz00 << 2) = 0000abcd | wxyz0000 = wxyzabcd
        bits_packed_4_a = (x_1 >> 2) | ((x_0 << 2) & 0xF0)
        bits_packed_4_b = (x_3 >> 2) | ((x_2 << 2) & 0xF0)
        # Similarly pack 4 remaining 2-bit partial representations into one uint8
        # e.g. 000000ab, 0000cd00, 00ef0000, gh000000 --> abcdefgh
        bits_packed_2 = (
            (x_0 << 6) | ((x_1 << 4) & 0x30) | ((x_2 << 2) & 0xC) | (x_3 & 0x3)
        )

        # Store values in a uint8 tensor of length `3 * MX_BLOCK_SIZE / 4`
        offsets_out_4_a = (
            offsets_rows[:, None] * PACKED_MX_BLOCK_SIZE + offsets_cols[None, :]
        )
        offsets_out_4_b = (
            offsets_rows[:, None] * PACKED_MX_BLOCK_SIZE
            + offsets_cols[None, :]
            + (MX_BLOCK_SIZE // 4)
        )
        offsets_out_2 = (
            offsets_rows[:, None] * PACKED_MX_BLOCK_SIZE
            + offsets_cols[None, :]
            + (MX_BLOCK_SIZE // 2)
        )

        # Store into output tensor
        tl.store(
            output_ptr + offsets_out_4_a,
            bits_packed_4_a,
            mask=mask,
        )

        tl.store(
            output_ptr + offsets_out_4_b,
            bits_packed_4_b,
            mask=mask,
        )

        tl.store(
            output_ptr + offsets_out_2,
            bits_packed_2,
            mask=mask,
        )

else:

    def triton_f4_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_elements_in,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
        BLOCK_SIZE_IN,
    ):
        raise AssertionError("unsupported without triton")

    def triton_f4_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_elements_in,
        mx_block_size,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
        e8m0_exponent_bias,
        e8m0_exponent_nan_val,
        BLOCK_SIZE_IN,
    ):
        raise AssertionError("unsupported without triton")

    def triton_f6_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_elements_in,
        sign_mask_f6,
        mbits_f6,
        f6_exp_bias,
        mbits_f32,
        f32_exp_bias,
        BLOCK_SIZE_IN,
    ):
        raise AssertionError("unsupported without triton")

    def triton_f6_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_elements_in,
        mx_block_size,
        sign_mask_f6,
        mbits_f6,
        f6_exp_bias,
        mbits_f32,
        f32_exp_bias,
        e8m0_exponent_bias,
        e8m0_exponent_nan_val,
        BLOCK_SIZE_IN,
    ):
        raise AssertionError("unsupported without triton")

    def triton_pack_uint6_kernel(
        input_ptr,
        output_ptr,
        n_mx_blocks,
        MX_BLOCK_SIZE,
        PACKED_MX_BLOCK_SIZE,
        BLOCK_SIZE,
    ):
        raise AssertionError("unsupported without triton")


def triton_f4_to_bf16(x: torch.Tensor):
    """
    Input: a tensor of packed fp4 values
    Output: a tensor of bfloat16 values

    Note: this function is only used in testing, so we can test
      the numerical correctness of the cast without the scaling.
    """
    new_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda
    n_elements_in = x.numel()
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),
    )  # noqa: E731,E501
    triton_f4_to_bf16_kernel[grid](
        x,
        output,
        n_elements_in,
        sign_mask_f4=SIGN_MASK_F4,
        mantissa_mask_f4=MANTISSA_MASK_F4,
        mbits_f4_e2m1=MBITS_F4_E2M1,
        ebits_f4_e2m1=EBITS_F4_E2M1,
        f4_e2m1_exp_bias=F4_E2M1_EXP_BIAS,
        mbits_f32=MBITS_F32,
        ebits_f32=EBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
        zero_bits_f32=ZERO_BITS_F32,
        zero_point_five_bits_f32=ZERO_POINT_FIVE_BITS_F32,
        BLOCK_SIZE_IN=512,
    )
    return output


def triton_f4_to_scaled_bf16(
    x: torch.Tensor,
    s_e8m0: torch.Tensor,
    mx_block_size: int,
):
    """
    Input: a tensor of packed fp4 values, and a scale in e8m0 format. The block
      size is currently assumed to be 32.
    Output: a tensor of bfloat16 values, multiplied by the encoded scale
    """
    assert TORCH_VERSION_AT_LEAST_2_4, "unsupported"
    new_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda
    n_elements_in = x.numel()
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),
    )
    triton_f4_to_scaled_bf16_kernel[grid](
        x,
        s_e8m0,
        output,
        n_elements_in,
        mx_block_size,
        sign_mask_f4=SIGN_MASK_F4,
        mantissa_mask_f4=MANTISSA_MASK_F4,
        mbits_f4_e2m1=MBITS_F4_E2M1,
        ebits_f4_e2m1=EBITS_F4_E2M1,
        f4_e2m1_exp_bias=F4_E2M1_EXP_BIAS,
        mbits_f32=MBITS_F32,
        ebits_f32=EBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
        zero_bits_f32=ZERO_BITS_F32,
        zero_point_five_bits_f32=ZERO_POINT_FIVE_BITS_F32,
        e8m0_exponent_bias=E8M0_EXPONENT_BIAS,
        e8m0_exponent_nan_val=E8M0_EXPONENT_NAN_VAL,
    )
    return output


def triton_f6_e2m3_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """
    Input: a tensor of packed fp6 values
    Output: a tensor of bfloat16 values

    Note: this function is only used in testing, so we can test
      the numerical correctness of the cast without the scaling.
    """
    packed_mx_block_size = x.shape[-1]
    mx_block_size = 4 * packed_mx_block_size // 3

    x = x.view(-1, packed_mx_block_size)
    new_shape = (x.shape[0], mx_block_size)

    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)

    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda

    n_mx_blocks = x.shape[0]
    grid = lambda meta: (triton.cdiv(n_mx_blocks, meta["BLOCK_SIZE_IN"]),)
    triton_f6_to_bf16_kernel[grid](
        x,
        output,
        n_mx_blocks,
        mx_block_size,
        packed_mx_block_size,
        sign_mask_f6=SIGN_MASK_F6_E2M3,
        mbits_f6=MBITS_F6_E2M3,
        f6_exp_bias=F6_E2M3_EXP_BIAS,
        mbits_f32=MBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
    )
    return output


def triton_f6_e3m2_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """
    Input: a tensor of packed fp6 values
    Output: a tensor of bfloat16 values

    Note: this function is only used in testing, so we can test
      the numerical correctness of the cast without the scaling.
    """
    packed_mx_block_size = x.shape[-1]
    mx_block_size = 4 * packed_mx_block_size // 3

    x = x.view(-1, packed_mx_block_size)
    new_shape = (x.numel() // packed_mx_block_size, mx_block_size)

    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)

    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda

    n_mx_blocks = x.shape[0]
    grid = lambda meta: (triton.cdiv(n_mx_blocks, meta["BLOCK_SIZE_IN"]),)
    triton_f6_to_bf16_kernel[grid](
        x,
        output,
        n_mx_blocks,
        mx_block_size,
        packed_mx_block_size,
        sign_mask_f6=SIGN_MASK_F6_E3M2,
        mbits_f6=MBITS_F6_E3M2,
        f6_exp_bias=F6_E3M2_EXP_BIAS,
        mbits_f32=MBITS_F32,
        f32_exp_bias=F32_EXP_BIAS,
    )
    return output


if TORCH_VERSION_AT_LEAST_2_4:

    @torch.library.custom_op("ao::triton_f6_e2m3_to_scaled_bf16", mutates_args=())
    def triton_f6_e2m3_to_scaled_bf16(
        x: torch.Tensor,
        s_e8m0: torch.Tensor,
        mx_block_size: int,
    ) -> torch.Tensor:
        """
        Input: a tensor of packed fp6 values, and a scale in e8m0 format. The block
        size is currently assumed to be 32.
        Output: a tensor of bfloat16 values, multiplied by the encoded scale
        """

        packed_mx_block_size = 3 * mx_block_size // 4

        x = x.view(-1, packed_mx_block_size)
        new_shape = (x.numel() // packed_mx_block_size, mx_block_size)

        output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)

        assert x.is_contiguous()
        assert x.is_cuda and output.is_cuda

        n_mx_blocks = x.shape[0]
        grid = lambda meta: (triton.cdiv(n_mx_blocks, meta["BLOCK_SIZE_IN"]),)
        triton_f6_to_scaled_bf16_kernel[grid](
            x,
            s_e8m0,
            output,
            n_mx_blocks,
            mx_block_size,
            packed_mx_block_size,
            sign_mask_f6=SIGN_MASK_F6_E2M3,
            mbits_f6=MBITS_F6_E2M3,
            f6_exp_bias=F6_E2M3_EXP_BIAS,
            mbits_f32=MBITS_F32,
            f32_exp_bias=F32_EXP_BIAS,
            e8m0_exponent_bias=E8M0_EXPONENT_BIAS,
            e8m0_exponent_nan_val=E8M0_EXPONENT_NAN_VAL,
        )
        return output

    @torch.library.custom_op("ao::triton_f6_e3m2_to_scaled_bf16", mutates_args=())
    def triton_f6_e3m2_to_scaled_bf16(
        x: torch.Tensor,
        s_e8m0: torch.Tensor,
        mx_block_size: int,
    ) -> torch.Tensor:
        """
        Input: a tensor of packed fp6 values, and a scale in e8m0 format. The block
        size is currently assumed to be 32.
        Output: a tensor of bfloat16 values, multiplied by the encoded scale
        """

        packed_mx_block_size = 3 * mx_block_size // 4

        x = x.view(-1, packed_mx_block_size)
        new_shape = (x.numel() // packed_mx_block_size, mx_block_size)

        output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)

        assert x.is_contiguous()
        assert x.is_cuda and output.is_cuda

        n_mx_blocks = x.numel() // packed_mx_block_size
        grid = lambda meta: (triton.cdiv(n_mx_blocks, meta["BLOCK_SIZE_IN"]),)
        triton_f6_to_scaled_bf16_kernel[grid](
            x,
            s_e8m0,
            output,
            n_mx_blocks,
            mx_block_size,
            packed_mx_block_size,
            sign_mask_f6=SIGN_MASK_F6_E3M2,
            mbits_f6=MBITS_F6_E3M2,
            f6_exp_bias=F6_E3M2_EXP_BIAS,
            mbits_f32=MBITS_F32,
            f32_exp_bias=F32_EXP_BIAS,
            e8m0_exponent_bias=E8M0_EXPONENT_BIAS,
            e8m0_exponent_nan_val=E8M0_EXPONENT_NAN_VAL,
        )
        return output

    @triton_f6_e3m2_to_scaled_bf16.register_fake
    def _(x, s_e8m0, mx_block_size):
        _padded_mx_block_size = 3 * mx_block_size // 4
        out_shape = (x.numel() // _padded_mx_block_size, mx_block_size)
        return torch.empty(*out_shape, device=x.device, dtype=torch.bfloat16)

    @triton_f6_e2m3_to_scaled_bf16.register_fake
    def _(x, s_e8m0, mx_block_size):
        _padded_mx_block_size = 3 * mx_block_size // 4
        out_shape = (x.numel() // _padded_mx_block_size, mx_block_size)
        return torch.empty(*out_shape, device=x.device, dtype=torch.bfloat16)

else:

    def triton_f6_e2m3_to_scaled_bf16(
        x: torch.Tensor,
        s_e8m0: torch.Tensor,
        mx_block_size: int,
    ) -> torch.Tensor:
        raise AssertionError("unsupported without torch >= 2.4")

    def triton_f6_e3m2_to_scaled_bf16(
        x: torch.Tensor,
        s_e8m0: torch.Tensor,
        mx_block_size: int,
    ) -> torch.Tensor:
        raise AssertionError("unsupported without torch >= 2.4")


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
    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0b1111).to(torch.uint8)
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
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size(shape))


# PyTorch implementation of fp6 packing for reference purposes
def pack_uint6_pytorch(uint8_data: torch.Tensor) -> torch.Tensor:
    # check shape is divisible by 4 along packing axis
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0

    packed_shape = [*shape[:-1], 3 * shape[-1] // 4]

    uint8_data = uint8_data.contiguous().view(-1)

    # pack 4 bits of each of 4 numbers into 2xuint8, remaining 2 bits into 1xuint8
    bits_packed_4_a = (uint8_data[1::4] >> 2) | ((uint8_data[::4] << 2) & 0xF0)
    bits_packed_4_b = (uint8_data[2::4] >> 2) | ((uint8_data[3::4] << 2) & 0xF0)
    bits_packed_2 = (
        (uint8_data[::4] << 6)
        | ((uint8_data[1::4] << 4) & 0x30)
        | ((uint8_data[3::4] << 2) & 0xC)
        | (uint8_data[2::4] & 0x3)
    )

    return (
        torch.stack((bits_packed_4_a, bits_packed_4_b, bits_packed_2), dim=-1)
    ).view(packed_shape)


if TORCH_VERSION_AT_LEAST_2_4:

    @torch.library.custom_op("ao::pack_uint6", mutates_args=())
    def pack_uint6(uint8_data: torch.Tensor) -> torch.Tensor:
        # ensure input data is contiguous before passing to kernel
        assert uint8_data.is_contiguous()

        # tensor should already be of shape [..., mx_block_size]
        mx_block_size = uint8_data.shape[-1]
        assert mx_block_size % 4 == 0

        # effective mx block size since we're packing 2 fp4 into 1 uint8
        packed_mx_block_size = 3 * mx_block_size // 4
        packed_shape = [uint8_data.shape[0], packed_mx_block_size]
        n_mx_blocks = uint8_data.numel() // mx_block_size

        grid = lambda meta: (triton.cdiv(n_mx_blocks, meta["BLOCK_SIZE_IN"]),)

        # contiguous uint8 container in which we can store the unpacked tensor
        packed_uint8_data = torch.empty(
            packed_shape, dtype=torch.uint8, device=uint8_data.device
        )

        triton_pack_uint6_kernel[grid](
            uint8_data,
            packed_uint8_data,
            n_mx_blocks,
            MX_BLOCK_SIZE=mx_block_size,
            PACKED_MX_BLOCK_SIZE=packed_mx_block_size,
        )

        return packed_uint8_data

    @pack_uint6.register_fake
    def _(uint8_data):
        out_shape = (*uint8_data.shape[:-1], 3 * uint8_data.shape[-1] // 4)
        return torch.empty(*out_shape, device=uint8_data.device, dtype=torch.uint8)
else:

    def pack_uint6(uint8_data: torch.Tensor) -> torch.Tensor:
        # Dummy placeholder op for torch < 2.4
        raise AssertionError("fp6 packing unsupported without torch >= 2.4")
