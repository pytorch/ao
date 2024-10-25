import torch
from torch import Tensor


# https://github.com/TimDettmers/bitsandbytes/blob/dada530149212d64d4b69534716202659ef37ec8/bitsandbytes/functional.py#L339-L391
# NOTE: zero padding is removed so this function can work with 4-bit qmap
def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    data.sort()
    return data


def scale_tensor(input: Tensor, block_size: int):
    """Scale tensor so that max(abs(input)) = 1"""
    shape = input.shape

    # section 2.1 from https://arxiv.org/abs/2110.02861
    input = input.view(-1, block_size)
    scale = input.abs().amax(-1).clip(1e-12)
    input = input / scale.view(-1, 1)
    return input.view(shape), scale


def quantize_8bit_with_qmap(input: Tensor, qmap: Tensor):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[128], 128, 0)
    codes += torch.where(input >= qmap[codes + 64], 64, 0)
    codes += torch.where(input >= qmap[codes + 32], 32, 0)
    codes += torch.where(input >= qmap[codes + 16], 16, 0)
    codes += torch.where(input >= qmap[codes + 8], 8, 0)
    codes += torch.where(input >= qmap[codes + 4], 4, 0)
    codes += torch.where(input >= qmap[codes + 2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes_up = (codes + 1).clip(max=255)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down
    codes = torch.where(residual >= (val_up - val_down) * 0.5, codes_up, codes)

    return codes.to(torch.uint8)


def quantize_4bit_with_qmap(input: Tensor, qmap: Tensor):
    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    codes = torch.where(input >= qmap[8], 8, 0)
    codes += torch.where(input >= qmap[codes + 4], 4, 0)
    codes += torch.where(input >= qmap[codes + 2], 2, 0)
    codes += torch.where(input >= qmap[codes + 1], 1, 0)

    # rounding
    codes_up = (codes + 1).clip(max=15)
    val_down = qmap[codes]
    val_up = qmap[codes_up]
    residual = input - val_down
    codes = torch.where(residual >= (val_up - val_down) * 0.5, codes_up, codes)

    return codes.to(torch.uint8)


def dequant_with_qmap(codes: Tensor, qmap: Tensor, scale: Tensor):
    # torch.compile() cannot use uint8 as index
    out = qmap[codes.int()].view(scale.shape[0], -1) * scale.view(-1, 1)
    return out.view(codes.shape)


def _fp32_to_bf16_sr(x_f32: Tensor) -> Tensor:
    # For an FP32 number      [a31, ..., a16, a15, ..., a0] to be converted to BF16
    # - Round towards zero:   [a31, ..., a16,   0, ...,  0]
    # - Round away from zero: [a31, ..., a16+1, 0, ...,  0]
    # (since the value can be negative, we use round towards/away from zero instead of round up/down)
    #
    # For stochastic rounding, we round away from zero with the probability of
    # [a15, ..., a0] / 2^16, where the bit pattern [a15, ..., a0] is interpreted as uint16
    #
    # we have to use int32 since most arithmetic ops are not implemented for uint32/int16/uint16
    rand_16bit = torch.randint(0, 1 << 16, x_f32.shape, device=x_f32.device, dtype=torch.int32)
    x_f32_bits = x_f32.view(torch.int32)
    x_fraction = x_f32_bits & 0xFFFF                # lower 16 bits
    x_bf16_towards_zero = x_f32_bits & 0xFFFF0000   # upper 16 bits

    x_f32_bits = torch.where(
        rand_16bit < x_fraction,        # this is True with the probability of p_fraction
        x_bf16_towards_zero + 0x10000,  # this might overflow, which will result in UB due to signed integer
        x_bf16_towards_zero,
    )
    # alternative, slightly faster
    # x_f32_bits = (x_f32_bits + rand_16bit) & 0xFFFF0000
    return x_f32_bits.view(torch.float32).bfloat16()
