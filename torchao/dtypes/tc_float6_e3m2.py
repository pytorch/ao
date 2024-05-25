# https://arxiv.org/abs/2401.14112

import torch
from torch import Tensor
from .float6_e3m2 import to_float6_e3m2


def pack_2bit(x: Tensor) -> Tensor:
    return (x[..., ::4] << 6) | (x[..., 1::4] << 4) | (x[..., 2::4] << 2) | x[..., 3::4]


def unpack_2bit(x: Tensor) -> Tensor:
    return torch.stack([x >> 6, (x >> 4) & 0b11, (x >> 2) & 0b11, x & 0b11], dim=-1).flatten(-2)


def pack_4bit(x: Tensor) -> Tensor:
    return (x[..., ::2] << 4) | x[..., 1::2]


def unpack_4bit(x: Tensor) -> Tensor:
    return torch.stack([x >> 4, x & 0b1111], dim=-1).flatten(-2)


# this is a literal adaptation of ahead-of-time bit-level pre-packing
# https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_prepacking.h
def to_tc_float6_e3m2(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fp6 = to_float6_e3m2(tensor, no_bit_packing=True)

    # Pass 1 from original code
    tensor_fp6 = tensor_fp6.view(M // 64, 4, 2, 8, N // 16, 2, 8)
    tensor_fp6 = tensor_fp6.permute(0, 4, 1, 5, 2, 3, 6)
    tensor_fp6 = tensor_fp6.reshape(-1, 32, 2)
    tensor_fp6 = tensor_fp6.permute(1, 0, 2)
    tensor_fp6 = tensor_fp6.flatten()

    tensor_2bit = pack_2bit((tensor_fp6 >> 4) & 0b11)
    tensor_4bit = pack_4bit(tensor_fp6 & 0b1111)

    # Pass 2 from original code
    tensor_2bit = tensor_2bit.view(32, -1, 4).permute(1, 0, 2).flip(2)
    tensor_4bit = tensor_4bit.view(32, -1, 4).permute(1, 0, 2).flip(2)

    # Pass 3 from original code
    tensor_2bit = unpack_2bit(tensor_2bit).view(-1, 16)
    tensor_4bit = unpack_4bit(tensor_4bit).view(-1, 8)

    # permutation like the original code (BitInterleaving_2bit)
    # the 1st and 3rd permutations are needed because the author unpacks/packs the values from/to uint32
    # while we still unpack/pack the values from/to uint8
    # tensor_2bit = tensor_2bit[:, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
    # tensor_2bit = tensor_2bit[:, [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14]]
    # tensor_2bit = tensor_2bit[:, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]

    # merged 3 permutations into 1
    tensor_2bit = tensor_2bit[:, [14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1]]
    tensor_2bit = pack_2bit(tensor_2bit).view(-1)

    # permutation like the original code (BitInterleaving_4bit)
    # the 1st and 3rd permutations are needed because the author unpacks/packs the values from/to uint32
    # while we still unpack/pack the values from/to uint8
    # tensor_4bit = tensor_4bit[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    # tensor_4bit = tensor_4bit[:, [1, 5, 3, 7, 0, 4, 2, 6]]
    # tensor_4bit = tensor_4bit[:, [4, 5, 6, 7, 0, 1, 2, 3]]

    # merged 3 permutations into 1
    tensor_4bit = tensor_4bit[:, [4, 0, 6, 2, 5, 1, 7, 3]]
    tensor_4bit = pack_4bit(tensor_4bit).view(-1)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0)
