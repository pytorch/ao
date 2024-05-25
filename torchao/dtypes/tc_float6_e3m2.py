# https://arxiv.org/abs/2401.14112

import torch
from torch import Tensor

# NOTE: This implementation requires FP32 denormal numbers to be handled correctly.
# On CPU, denormal numbers might be flushed to zero for performance gain (FTZ and DAZ flags).
def _to_float6_e3m2_pt(tensor: Tensor, packed: bool = False) -> Tensor:
    tensor = tensor.float()

    # correct exponent bias. this also handles subnormal numbers correctly
    tensor = tensor * 2.0 ** (-127 + 3)
    bits = tensor.view(torch.int32)

    sign = ((bits >> 31) & 0x1) << 5
    exp_and_man = (bits >> 21) & 0x1F
    result = sign | exp_and_man

    # round to nearest even
    remainder = bits & 0x1F_FFFF  # truncated mantissa bits
    do_round_up = (remainder > 0x10_0000) | ((remainder == 0x10_0000) & ((result & 1) == 1))
    result = torch.where(do_round_up, result + 1, result)
    result = result.to(torch.uint8)

    if not packed:
        return result

    # bit packing
    val0, val1, val2, val3 = result.unflatten(-1, (-1, 4)).unbind(-1)
    bits0 = (val0 << 2) | (val1 >> 4)  # 0000 0011
    bits1 = (val1 << 4) | (val2 >> 2)  # 1111 2222
    bits2 = (val2 << 6) | (val3);      # 2233 3333
    return torch.stack([bits0, bits1, bits2], dim=-1).flatten(-2)


def pack_2bit(x: Tensor) -> Tensor:
    return (x[..., ::4] << 6) | (x[..., 1::4] << 4) | (x[..., 2::4] << 2) | x[..., 3::4]


def unpack_2bit(x: Tensor) -> Tensor:
    return torch.stack([x >> 6, (x >> 4) & 0b11, (x >> 2) & 0b11, x & 0b11], dim=-1).flatten(-2)


def pack_4bit(x: Tensor) -> Tensor:
    return (x[..., ::2] << 4) | x[..., 1::2]


def unpack_4bit(x: Tensor) -> Tensor:
    return torch.stack([x >> 4, x & 0b1111], dim=-1).flatten(-2)


def to_tc_float6_e3m2(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fp6 = _to_float6_e3m2_pt(tensor)

    tensor_fp6 = tensor_fp6.view(M // 64, 4, 2, 8, N // 16, 2, 8)
    tensor_fp6 = tensor_fp6.permute(0, 4, 1, 5, 2, 3, 6)
    tensor_fp6 = tensor_fp6.reshape(-1, 32, 2)
    tensor_fp6 = tensor_fp6.permute(1, 0, 2)
    tensor_fp6 = tensor_fp6.flatten()

    tensor_2bit = pack_2bit((tensor_fp6 >> 4) & 0b11)
    tensor_4bit = pack_4bit(tensor_fp6 & 0b1111)

    tensor_2bit = tensor_2bit.view(32, -1, 4).permute(1, 0, 2).flip(2)
    tensor_4bit = tensor_4bit.view(32, -1, 4).permute(1, 0, 2).flip(2)

    tensor_2bit = unpack_2bit(tensor_2bit).view(-1, 16)
    tensor_2bit = tensor_2bit[:, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_2bit = tensor_2bit[:, [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14]]
    tensor_2bit = tensor_2bit[:, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_2bit = pack_2bit(tensor_2bit)

    tensor_4bit = unpack_4bit(tensor_4bit).view(-1, 8)
    tensor_4bit = tensor_4bit[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_4bit = tensor_4bit[:, [1, 5, 3, 7, 0, 4, 2, 6]]
    tensor_4bit = tensor_4bit[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_4bit = pack_4bit(tensor_4bit)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0)
