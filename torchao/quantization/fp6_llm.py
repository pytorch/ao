from typing import Optional

import torch
from torch import nn, Tensor
from torchao.dtypes.float6_e3m2 import FLOAT6_E3M2_MAX, to_float6_e3m2, from_float6_e3m2
from torchao.ops import fp16act_fp6weight_linear


def _pack_2bit(x: Tensor) -> Tensor:
    return (x[..., ::4] << 6) | (x[..., 1::4] << 4) | (x[..., 2::4] << 2) | x[..., 3::4]


def _unpack_2bit(x: Tensor) -> Tensor:
    return torch.stack([x >> 6, (x >> 4) & 0b11, (x >> 2) & 0b11, x & 0b11], dim=-1).flatten(-2)


def _pack_4bit(x: Tensor) -> Tensor:
    return (x[..., ::2] << 4) | x[..., 1::2]


def _unpack_4bit(x: Tensor) -> Tensor:
    return torch.stack([x >> 4, x & 0b1111], dim=-1).flatten(-2)


# this is a literal adaptation of FP6-LLM ahead-of-time bit-level pre-packing
# https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_prepacking.h
def _to_tc_float6_e3m2_original(tensor: Tensor) -> Tensor:
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

    tensor_2bit = _pack_2bit((tensor_fp6 >> 4) & 0b11)
    tensor_4bit = _pack_4bit(tensor_fp6 & 0b1111)

    # Pass 2 from original code
    tensor_2bit = tensor_2bit.view(32, -1, 4).permute(1, 0, 2).flip(2)
    tensor_4bit = tensor_4bit.view(32, -1, 4).permute(1, 0, 2).flip(2)

    # Pass 3 from original code
    # BitInterleaving_2bit
    # the 1st and 3rd permutations are needed because the author unpacks/packs the values from/to uint32
    # while we still unpack/pack the values from/to uint8
    tensor_2bit = _unpack_2bit(tensor_2bit).view(-1, 16)
    tensor_2bit = tensor_2bit[:, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_2bit = tensor_2bit[:, [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14]]
    tensor_2bit = tensor_2bit[:, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_2bit = _pack_2bit(tensor_2bit).view(-1)

    # BitInterleaving_4bit
    # the 1st and 3rd permutations are needed because the author unpacks/packs the values from/to uint32
    # while we still unpack/pack the values from/to uint8
    tensor_4bit = _unpack_4bit(tensor_4bit).view(-1, 8)
    tensor_4bit = tensor_4bit[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_4bit = tensor_4bit[:, [1, 5, 3, 7, 0, 4, 2, 6]]
    tensor_4bit = tensor_4bit[:, [4, 5, 6, 7, 0, 1, 2, 3]]
    tensor_4bit = _pack_4bit(tensor_4bit).view(-1)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0)


# more optimized version of _to_tc_float6_e3m2_original() by merging ops
# https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_prepacking.h
def to_tc_float6_e3m2(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fp6 = to_float6_e3m2(tensor, no_bit_packing=True)
    tensor_fp6 = tensor_fp6.view(M // 64, 2, 2, 2, 8, N // 16, 2, 8)
    tensor_fp6 = tensor_fp6.flip(3)

    tensor_2bit = (tensor_fp6 >> 4) & 0b11
    tensor_2bit = tensor_2bit.permute(0, 5, 1, 4, 7, 3, 2, 6)
    tensor_2bit = _pack_2bit(tensor_2bit.flatten())

    tensor_4bit = tensor_fp6 & 0b1111
    tensor_4bit = tensor_4bit.permute(0, 5, 1, 2, 4, 7, 3, 6)
    tensor_4bit = _pack_4bit(tensor_4bit.flatten())

    return torch.cat([tensor_2bit, tensor_4bit], dim=0)


def from_tc_float6_e3m2(tensor: Tensor, M: int, N: int, dtype: torch.dtype = torch.float32) -> Tensor:
    assert tensor.ndim == 1
    assert (M % 64 == 0) and (N % 64 == 0)
    size_2bit = M * N // 4
    size_4bit = M * N // 2
    assert tensor.numel() == size_2bit + size_4bit

    tensor_2bit, tensor_4bit = tensor.split([size_2bit, size_4bit])

    tensor_2bit = _unpack_2bit(tensor_2bit)
    tensor_2bit = tensor_2bit.view(M // 64, N // 16, 2, 8, 8, 2, 2, 2)
    tensor_2bit = tensor_2bit.permute(0, 2, 6, 5, 3, 1, 7, 4)

    tensor_4bit = _unpack_4bit(tensor_4bit)
    tensor_4bit = tensor_4bit.view(M // 64, N // 16, 2, 2, 8, 8, 2, 2)
    tensor_4bit = tensor_4bit.permute(0, 2, 3, 6, 4, 1, 7, 5)

    tensor_fp6 = (tensor_2bit << 4) | tensor_4bit
    tensor_fp6 = tensor_fp6.flip(3).reshape(M, N)
    return from_float6_e3m2(tensor_fp6, no_bit_packing=True, dtype=dtype)


class Fp6LlmLinear(nn.Module):
    """FP6-LLM Linear layer as described in https://arxiv.org/pdf/2401.14112.
    """

    def __init__(self, weight: Tensor, scales: Tensor, bias: Optional[Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] * 16 // 3

    def forward(self, x: Tensor):
        out = fp16act_fp6weight_linear(x.half(), self.weight, self.scales, splitK=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_float(cls, linear: nn.Linear):
        assert (linear.in_features % 64 == 0) and (linear.out_features % 256 == 0)

        fp32_weight = linear.weight.detach().float()
        scales = fp32_weight.abs().amax(1) / FLOAT6_E3M2_MAX
        scales[scales == 0.0] = 1.0  # avoid 0 scale

        tc_fp6_weight = to_tc_float6_e3m2(fp32_weight / scales.view(-1, 1))
        tc_fp6_weight = tc_fp6_weight.view(linear.out_features, -1).view(torch.int32)

        bias = linear.bias.detach().half() if linear.bias is not None else None
        return cls(tc_fp6_weight, scales.half(), bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def convert_fp6_llm(model: nn.Module, skip_fqn_list: Optional[list[str]] = None, cur_fqn: str = "") -> None:
    for name, child in model.named_children():
        new_fqn = name if cur_fqn == "" else f"{cur_fqn}.{name}"

        if ((skip_fqn_list is None) or (new_fqn not in skip_fqn_list)) and (isinstance(child, nn.Linear)):
            if (child.in_features % 64 == 0) and (child.out_features % 256 == 0):
                new_child = Fp6LlmLinear.from_float(child)  
                setattr(model, name, new_child)
        else:
            convert_fp6_llm(child, skip_fqn_list, new_fqn)
