import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchao.prototype.mx_formats.custom_cast import f32_to_f6_e3m2_unpacked, f6_e3m2_unpacked_to_f32
from torchao.prototype.mx_formats.constants import F6_E3M2_MAX
from torchao.ops import fp6_llm_linear


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
def _to_tc_float6_e3m2_ref(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fp6 = f32_to_f6_e3m2_unpacked(tensor.float())

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

    return torch.cat([tensor_2bit, tensor_4bit], dim=0).view(M, -1)


# more optimized version of _to_tc_float6_e3m2_original() by merging ops
# https://github.com/usyd-fsalab/fp6_llm/blob/ce76774bcfc26b325c1b558abcf1935026d9abbc/fp6_llm/csrc/utils/weight_prepacking.h
def to_tc_float6_e3m2(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor_fp6 = f32_to_f6_e3m2_unpacked(tensor.float())
    tensor_fp6 = tensor_fp6.view(M // 64, 2, 2, 2, 8, N // 16, 2, 8)
    tensor_fp6 = tensor_fp6.flip(3)

    tensor_2bit = (tensor_fp6 >> 4) & 0b11
    tensor_2bit = tensor_2bit.permute(0, 5, 1, 4, 7, 3, 2, 6)
    tensor_2bit = _pack_2bit(tensor_2bit.flatten())

    tensor_4bit = tensor_fp6 & 0b1111
    tensor_4bit = tensor_4bit.permute(0, 5, 1, 2, 4, 7, 3, 6)
    tensor_4bit = _pack_4bit(tensor_4bit.flatten())

    return torch.cat([tensor_2bit, tensor_4bit], dim=0).view(M, -1)


def to_scaled_tc_float6_e3m2(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    scale = F6_E3M2_MAX / tensor.abs().amax(1).clamp(min=1e-12)
    tc_fp6_tensor = to_tc_float6_e3m2(tensor * scale.view(-1, 1))
    return tc_fp6_tensor, scale.reciprocal().half()


def from_tc_float6_e3m2(tensor: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    N = tensor.shape[1] // 3 * 4
    assert (M % 64 == 0) and (N % 64 == 0)
    size_2bit = M * N // 4
    size_4bit = M * N // 2
    tensor = tensor.view(-1).view(torch.uint8)
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
    return f6_e3m2_unpacked_to_f32(tensor_fp6).to(dtype)


# https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
_SPLIT_K_MAP = [
    {  # tokens: [1, 64]
        3072: 18,
        4096: 13,
        5120: 10,
        6144: 9,
        8192: 6,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 7
    },
    {  # tokens: [65:128]
        3072: 9,
        4096: 6,
        5120: 5,
        6144: 9,
        8192: 3,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 6
    },
    {  # tokens: [129:192]
        3072: 6,
        4096: 4,
        5120: 7,
        6144: 3,
        8192: 2,
        10240: 5,
        14336: 5,
        28672: 5,
        57344: 4
    },
    {  # tokens: [193:256]
        3072: 9,
        4096: 3,
        5120: 5,
        6144: 2,
        8192: 5,
        10240: 4,
        14336: 8,
        28672: 6,
        57344: 4
    },
    {  # tokens: [257:320]
        3072: 7,
        4096: 5,
        5120: 2,
        6144: 5,
        8192: 4,
        10240: 1,
        14336: 3,
        28672: 3,
        57344: 4
    },
    {  # tokens: [321:384]
        3072: 3,
        4096: 2,
        5120: 5,
        6144: 3,
        8192: 1,
        10240: 8,
        14336: 3,
        28672: 4,
        57344: 3
    },
    {  # tokens: [385:448]
        3072: 5,
        4096: 7,
        5120: 3,
        6144: 5,
        8192: 7,
        10240: 3,
        14336: 1,
        28672: 1,
        57344: 3
    },
    {  # tokens: [449:512]
        3072: 2,
        4096: 5,
        5120: 4,
        6144: 1,
        8192: 5,
        10240: 2,
        14336: 6,
        28672: 4,
        57344: 1
    },
    {  # tokens: [513:576]
        3072: 2,
        4096: 3,
        5120: 1,
        6144: 1,
        8192: 3,
        10240: 3,
        14336: 3,
        28672: 1,
        57344: 1
    },
    {  # tokens: [577:640]
        3072: 5,
        4096: 4,
        5120: 1,
        6144: 4,
        8192: 2,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [641:704]
        3072: 3,
        4096: 1,
        5120: 2,
        6144: 2,
        8192: 1,
        10240: 2,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [705:768]
        3072: 3,
        4096: 1,
        5120: 3,
        6144: 2,
        8192: 1,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    }
]


class Fp6LlmLinear(nn.Module):
    """FP6-LLM Linear layer as described in https://arxiv.org/pdf/2401.14112.
    """

    def __init__(self, weight: Tensor, scales: Tensor, bias: Optional[Tensor] = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight.view(torch.int32))
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1] // 3 * 4

    def forward(self, x: Tensor) -> Tensor:
        splitK = self.get_split_k(math.prod(x.shape[:-1]), self.out_features)
        out = fp6_llm_linear(x.view(-1, self.in_features).half(), self.weight, self.scales, splitK=splitK)
        if self.bias is not None:
            out = out + self.bias
        return out.view(*x.shape[:-1], self.out_features).to(x.dtype)

    @staticmethod
    def get_split_k(bsize: int, out_dim: int) -> int:
        # https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
        return _SPLIT_K_MAP[(bsize - 1) // 64].get(out_dim, 1) if bsize <= 768 else 1

    @classmethod
    def from_float(cls, linear: nn.Linear):
        assert (linear.in_features % 64 == 0) and (linear.out_features % 256 == 0)

        fp6_weight, scale = to_scaled_tc_float6_e3m2(linear.weight.detach())
        bias = linear.bias.detach().half() if linear.bias is not None else None
        return cls(fp6_weight, scale, bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def convert_fp6_llm(model: nn.Module, skip_fqn_list: Optional[List[str]] = None, cur_fqn: str = "") -> None:
    for name, child in model.named_children():
        new_fqn = name if cur_fqn == "" else f"{cur_fqn}.{name}"

        if ((skip_fqn_list is None) or (new_fqn not in skip_fqn_list)) and (isinstance(child, nn.Linear)):
            if (child.in_features % 64 == 0) and (child.out_features % 256 == 0):
                new_child = Fp6LlmLinear.from_float(child)  
                setattr(model, name, new_child)
        else:
            convert_fp6_llm(child, skip_fqn_list, new_fqn)
