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
from torchao.prototype.mx_formats.constants import (
    E8M0_EXPONENT_BIAS,
    E8M0_EXPONENT_NAN_VAL,
    F4_E2M1_EXP_BIAS,
    F32_EXP_BIAS,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4


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
        # S is already biased by 127, so we just have to shift it to align w/ bf16
        s_fp = (s.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
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
