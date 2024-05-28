# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import struct

import numpy as np

import torch
from torch.utils._triton import has_triton

from torchao.quantization.utils import TORCH_VERSION_AFTER_2_4

# TODO(future): if needed, make the below work on previous PyTorch versions,
# just need to hunt down the previous location of `libdevice`. An assert
# at the callsite prevents usage of this on unsupported versions.
if TORCH_VERSION_AFTER_2_4:
    from torch._inductor.runtime.triton_helpers import libdevice

from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    E8M0_EXPONENT_BIAS,
    E8M0_EXPONENT_NAN_VAL,
    F32_EXP_BIAS,
    F4_E2M1_EXP_BIAS,
    F4_E2M1_MAX,
    F4_E2M1_MAX_INT,
    F4_E2M1_MIN_NORMAL,
    F6_E2M3_EXP_BIAS,
    F6_E2M3_MAX,
    F6_E2M3_MAX_INT,
    F6_E2M3_MIN_NORMAL,
    F6_E3M2_EXP_BIAS,
    F6_E3M2_MAX,
    F6_E3M2_MAX_INT,
    F6_E3M2_MIN_NORMAL,
)


def get_bits(x: torch.Tensor) -> str:
    bits_per_byte = 8
    # Numpy has a nice function to get the string representation of binary.
    # Since we are using ints as views of floats, need to specify the width
    # to avoid numpy from using two's complement for negative numbers.
    return np.binary_repr(
        x.cpu().numpy(), width=x.element_size() * bits_per_byte
    )  # noqa: E501


EBITS_F32, MBITS_F32 = 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2

DENORM_F32TOF4_EXP = (
    # exp bias conversion between formats
    (F32_EXP_BIAS - F4_E2M1_EXP_BIAS)
    # mantissa length difference between formats
    + (MBITS_F32 - MBITS_F4_E2M1)
    # add one to encoded exponent for denormalized numbers
    + 1
)
DENORM_F32TOF4_MASK_INT = DENORM_F32TOF4_EXP << MBITS_F32
# reinterpret int32 as float32 in Python
# see https://stackoverflow.com/a/34446112/1058521
DENORM_F32TOF4_MASK_FLOAT = struct.unpack(
    "!f", struct.pack("!I", DENORM_F32TOF4_MASK_INT)
)[0]

DENORM_F32TOF6_E2M3_EXP = (
    # exp bias conversion between formats
    (F32_EXP_BIAS - F6_E2M3_EXP_BIAS)
    # mantissa length difference between formats
    + (MBITS_F32 - MBITS_F6_E2M3)
    # add one to encoded exponent for denormalized numbers
    + 1
)
DENORM_F32TOF6_E2M3_MASK_INT = DENORM_F32TOF6_E2M3_EXP << MBITS_F32
# reinterpret int32 as float32 in Python
# see https://stackoverflow.com/a/34446112/1058521
DENORM_F32TOF6_E2M3_MASK_FLOAT = struct.unpack(
    "!f", struct.pack("!I", DENORM_F32TOF6_E2M3_MASK_INT)
)[0]

DENORM_F32TOF6_E3M2_EXP = (
    # exp bias conversion between formats
    (F32_EXP_BIAS - F6_E3M2_EXP_BIAS)
    # mantissa length difference between formats
    + (MBITS_F32 - MBITS_F6_E3M2)
    # add one to encoded exponent for denormalized numbers
    + 1
)
DENORM_F32TOF6_E3M2_MASK_INT = DENORM_F32TOF6_E3M2_EXP << MBITS_F32
# reinterpret int32 as float32 in Python
# see https://stackoverflow.com/a/34446112/1058521
DENORM_F32TOF6_E3M2_MASK_FLOAT = struct.unpack(
    "!f", struct.pack("!I", DENORM_F32TOF6_E3M2_MASK_INT)
)[0]

#
# magic value to add during the normal path
# TODO document this better
#

# c++ code e5m2:
# f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
# 0xFFFFF is 1111 1111 1111 1111 1111, 20 ones, 20 = 23 - 3 = 23 - 2 - 1

# c++ code e4m3:
# f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
# 0x7FFFF is 0111 1111 1111 1111 1111, 19 ones, 19 = 23 - 4 = 23 - 3 - 1

MAGIC_ADDER_F4_E2M1 = 0x1FFFFF  # 21 ones
MAGIC_ADDER_F6_E2M3 = 0x7FFFF  # 19 ones
MAGIC_ADDER_F6_E3M2 = 0xFFFFF  # 20 ones

# c++ code named vars
# f_bits += ((uint32_t)(f8_exp_bias - f32_exp_bias) << f32_mbits) + MAGIC_ADDER;  # noqa: E501

SIGN_MASK_F4 = 0x8  # 1000
SIGN_MASK_F6_E2M3 = 0x20  # 100000
SIGN_MASK_F6_E3M2 = 0x20  # 100000

MANTISSA_MASK_F4 = 0x1  # 0001
MANTISSA_MASK_F6_E2M3 = 0x7  # 000111
MANTISSA_MASK_F6_E3M2 = 0x3  # 000011

ZERO_BITS_F32 = 0x0
ZERO_POINT_FIVE_BITS_F32 = 0x3F000000


def _f32_to_f4_or_f6_unpacked(
    x,
    max_normal,
    min_normal,
    denorm_mask_float,
    denorm_mask_int,
    ebits,
    mbits,
    exp_bias,
    magic_adder,
    max_int,
    sign_mask,
):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8,
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in the fp6_e2m3 or fp6_e3m2 encoding

    Note: there is no special values (NaN, inf) support in this code as the
    OCP spec does not define special values for fp6 and fp4 dtypes.

    Code below is an adaptation of https://fburl.com/code/ciwofcg4 for f4/f6

    Background 1: last answer in https://stackoverflow.com/questions/8981913/how-to-perform-round-to-even-with-floating-point-numbers  # noqa: E501
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(
        torch.logical_not(saturate_mask), x < min_normal
    )  # noqa: E501
    normal_mask = torch.logical_not(
        torch.logical_or(saturate_mask, denormal_mask)
    )  # noqa: E501

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def f32_to_f4_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-3 empty and
      bits 4-7 in fp4_e2m1
    """
    return _f32_to_f4_or_f6_unpacked(
        x,
        F4_E2M1_MAX,
        F4_E2M1_MIN_NORMAL,
        DENORM_F32TOF4_MASK_FLOAT,
        DENORM_F32TOF4_MASK_INT,
        EBITS_F4_E2M1,
        MBITS_F4_E2M1,
        F4_E2M1_EXP_BIAS,
        MAGIC_ADDER_F4_E2M1,
        F4_E2M1_MAX_INT,
        SIGN_MASK_F4,
    )


def f32_to_f6_e2m3_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-1 empty and
      bits 2-7 in fp6_e2m3
    """
    return _f32_to_f4_or_f6_unpacked(
        x,
        F6_E2M3_MAX,
        F6_E2M3_MIN_NORMAL,
        DENORM_F32TOF6_E2M3_MASK_FLOAT,
        DENORM_F32TOF6_E2M3_MASK_INT,
        EBITS_F6_E2M3,
        MBITS_F6_E2M3,
        F6_E2M3_EXP_BIAS,
        MAGIC_ADDER_F6_E2M3,
        F6_E2M3_MAX_INT,
        SIGN_MASK_F6_E2M3,
    )


def f32_to_f6_e3m2_unpacked(x):
    """
    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, with bits 0-1 empty and
      bits 2-7 in fp6_e3m2
    """
    return _f32_to_f4_or_f6_unpacked(
        x,
        F6_E3M2_MAX,
        F6_E3M2_MIN_NORMAL,
        DENORM_F32TOF6_E3M2_MASK_FLOAT,
        DENORM_F32TOF6_E3M2_MASK_INT,
        EBITS_F6_E3M2,
        MBITS_F6_E3M2,
        F6_E3M2_EXP_BIAS,
        MAGIC_ADDER_F6_E3M2,
        F6_E3M2_MAX_INT,
        SIGN_MASK_F6_E3M2,
    )


def _f4_or_f6_unpacked_to_f32(x: torch.Tensor, lp_dtype_name: str):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-3 empty and bits 4-7
      containing an fp4_e2m1 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value

    TODO(future): check if LUT for everything is faster than bit shifting,
      especially for fp4.
    """
    assert x.dtype == torch.uint8

    if lp_dtype_name == DTYPE_FP4:
        sign_mask = SIGN_MASK_F4
        ebits = EBITS_F4_E2M1
        mbits = MBITS_F4_E2M1
        exp_bias = F4_E2M1_EXP_BIAS
        mantissa_mask = MANTISSA_MASK_F4
    elif lp_dtype_name == DTYPE_FP6_E2M3:
        sign_mask = SIGN_MASK_F6_E2M3
        ebits = EBITS_F6_E2M3
        mbits = MBITS_F6_E2M3
        exp_bias = F6_E2M3_EXP_BIAS
        mantissa_mask = MANTISSA_MASK_F6_E2M3
    elif lp_dtype_name == DTYPE_FP6_E3M2:
        sign_mask = SIGN_MASK_F6_E3M2
        ebits = EBITS_F6_E3M2
        mbits = MBITS_F6_E3M2
        exp_bias = F6_E3M2_EXP_BIAS
        mantissa_mask = MANTISSA_MASK_F6_E3M2
    else:
        raise AssertionError(f"unsupported lp_dtype_name {lp_dtype_name}")

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = ZERO_BITS_F32
    # Note: for now the denormal path cast is written for readability and
    # numerical correctness. There is likely a way to optimize the performance,
    # I just haven't had time to look into it.
    if lp_dtype_name == DTYPE_FP4:
        result[denormal_mask] = ZERO_POINT_FIVE_BITS_F32

    elif lp_dtype_name == DTYPE_FP6_E2M3:
        # Only 7 possible values, just do a LUT
        # Note: calculate the booleans first because we are modifying
        # this variable inplace.
        is_val1 = mantissa_lp_int32 == 1
        is_val2 = mantissa_lp_int32 == 2
        is_val3 = mantissa_lp_int32 == 3
        is_val4 = mantissa_lp_int32 == 4
        is_val5 = mantissa_lp_int32 == 5
        is_val6 = mantissa_lp_int32 == 6
        is_val7 = mantissa_lp_int32 == 7
        mantissa_lp_int32[is_val1] = 0x3E000000  # 0.125
        mantissa_lp_int32[is_val2] = 0x3E800000  # 0.25
        mantissa_lp_int32[is_val3] = 0x3EC00000  # 0.375
        mantissa_lp_int32[is_val4] = 0x3F000000  # 0.5
        mantissa_lp_int32[is_val5] = 0x3F200000  # 0.625
        mantissa_lp_int32[is_val6] = 0x3F400000  # 0.75
        mantissa_lp_int32[is_val7] = 0x3F600000  # 0.875
        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    elif lp_dtype_name == DTYPE_FP6_E3M2:
        # Only 3 possible values, just do a LUT
        # Note: calculate the booleans first because we are modifying
        # this variable inplace.
        is_val1 = mantissa_lp_int32 == 1
        is_val2 = mantissa_lp_int32 == 2
        is_val3 = mantissa_lp_int32 == 3
        mantissa_lp_int32[is_val1] = 0x3D800000  # 0.0625
        mantissa_lp_int32[is_val2] = 0x3E000000  # 0.125
        mantissa_lp_int32[is_val3] = 0x3E400000  # 0.1875
        result = torch.where(denormal_mask, mantissa_lp_int32, result)
    else:
        raise AssertionError(f"unsupported lp_dtype_name {lp_dtype_name}")

    # add sign back
    sign_f32 = sign_lp.to(torch.int32) << (
        MBITS_F32 - mbits + EBITS_F32 - ebits
    )  # noqa: E501
    result = result | sign_f32

    return result.view(torch.float)


def f4_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-3 empty and bits 4-7
      containing an fp4_e2m1 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _f4_or_f6_unpacked_to_f32(x, DTYPE_FP4)


def f6_e2m3_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-1 empty and bits 2-7
      containing an fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _f4_or_f6_unpacked_to_f32(x, DTYPE_FP6_E2M3)


def f6_e3m2_unpacked_to_f32(x: torch.Tensor):
    """
    Input: torch.Tensor of dtype uint8, with bits 0-1 empty and bits 2-7
      containing an fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    return _f4_or_f6_unpacked_to_f32(x, DTYPE_FP6_E3M2)


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _fp4_packed_to_bf16(x_packed):
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
        sign_f4 = x & SIGN_MASK_F4

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
        exp_biased_f4 = x_pos >> MBITS_F4_E2M1
        exp_biased_f32 = exp_biased_f4 - F4_E2M1_EXP_BIAS + F32_EXP_BIAS
        exp_biased_f32 = exp_biased_f32.to(tl.int32) << MBITS_F32

        # shift the mantissa to bits 10:32 of the result
        mantissa_f4 = x_pos & MANTISSA_MASK_F4
        mantissa_f32 = mantissa_f4.to(tl.int32) << (MBITS_F32 - MBITS_F4_E2M1)
        output = mantissa_f32

        # combine the pieces
        result = exp_biased_f32 | mantissa_f32
        # result[zero_mask] = ZERO_BITS_F32
        result = tl.where(zero_mask, ZERO_BITS_F32, result)
        # result[denormal_mask] = ZERO_POINT_FIVE_BITS_F32
        result = tl.where(denormal_mask, ZERO_POINT_FIVE_BITS_F32, result)

        # add sign back
        sign_f32 = sign_f4.to(tl.int32) << (
            MBITS_F32 - MBITS_F4_E2M1 + EBITS_F32 - EBITS_F4_E2M1
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
        output = _fp4_packed_to_bf16(x_packed)

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
        output = _fp4_packed_to_bf16(x_packed)

        # load scale
        block_start_s = pid * BLOCK_SIZE_S
        offsets_s = block_start_s + tl.arange(0, BLOCK_SIZE_S)
        mask_s = offsets_s < n_elements_s
        s = tl.load(s_ptr + offsets_s, mask=mask_s)

        # create the scale in bf16
        s_offset = s.to(tl.int16) - E8M0_EXPONENT_BIAS
        s_fp = libdevice.pow(2.0, s_offset).to(tl.bfloat16)
        s_fp = tl.where(s != E8M0_EXPONENT_NAN_VAL, s_fp, float("nan"))

        # multiply output by scale
        # TODO(later): see if manipulating the exponent instead of fp
        # multiplication is going to give a significant speedup
        output = tl.reshape(
            output, (BLOCK_SIZE_OUT // mx_block_size, mx_block_size)
        )  # noqa: E501
        s_fp = tl.reshape(s_fp, (BLOCK_SIZE_S // 1, 1))
        output = output * s_fp
        output = tl.reshape(output, (BLOCK_SIZE_OUT,))

        # set up output offsets
        block_start_out = pid * BLOCK_SIZE_OUT
        offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT)
        mask_out = offsets_out < n_elements_out

        tl.store(output_ptr + offsets_out, output, mask=mask_out)

else:

    def triton_f4_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_elements_in,
        BLOCK_SIZE_IN,
    ):
        raise AssertionError("unsupported without triton")

    def triton_f4_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_elements_in,
        mx_block_size,
        BLOCK_SIZE_IN,
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
    triton_f4_to_bf16_kernel[grid](x, output, n_elements_in, BLOCK_SIZE_IN=512)
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
    assert TORCH_VERSION_AFTER_2_4, "unsupported"
    new_shape = (*x.shape[:-1], x.shape[-1] * 2)
    output = torch.empty(*new_shape, device=x.device, dtype=torch.bfloat16)
    assert x.is_contiguous()
    assert x.is_cuda and output.is_cuda
    n_elements_in = x.numel()
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(n_elements_in, meta["BLOCK_SIZE_IN"]),
    )
    triton_f4_to_scaled_bf16_kernel[grid](
        x, s_e8m0, output, n_elements_in, mx_block_size
    )
    return output


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


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size(shape))
