import torch
from torch import Tensor

from .subclass_8bit import create_dynamic_map


# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/config.py#L35-L65
# NOTE: power-1 is linear
QMAP_SIGNED = create_dynamic_map(max_exponent_bits=3, total_bits=4)
QMAP_UNSIGNED = torch.linspace(0, 1, 17)[1:].tolist()  # no zero

ZERO_CODE_SIGNED = QMAP_SIGNED.index(0)
ZERO_CODE_UNSIGNED = QMAP_UNSIGNED[0]  # nearest to zero


def quantize_4bit_with_qmap(input: Tensor, qmap: Tensor, block_size: int, implementation: int = 1):
    # section 2.1 from https://arxiv.org/abs/2110.02861
    input = input.view(-1, block_size)
    scale = input.abs().amax(-1).clip(1e-12)
    input = input / scale.view(-1, 1)

    # reference implementation. equation 4 from https://arxiv.org/abs/2110.02861
    if implementation == 0:
        codes = (qmap.view(1, -1) - input.view(-1, 1)).abs().argmin(-1)
        codes = codes.to(torch.uint8)

    # GPU-friendly binary search
    # https://blog.demofox.org/2017/06/20/simd-gpu-friendly-branchless-binary-search/
    elif implementation == 1:
        input = input.view(-1)
        codes = torch.where(input >= qmap[8], 8, 0)
        codes += torch.where(input >= qmap[codes + 4], 4, 0)
        codes += torch.where(input >= qmap[codes + 2], 2, 0)
        codes += torch.where(input >= qmap[codes + 1], 1, 0)

        # rounding
        codes_up = (codes + 1).clip(max=16)
        val_down = qmap[codes]
        val_up = qmap[codes_up]
        residual = input - val_down
        codes = torch.where(residual >= (val_up - val_down) * 0.5, codes_up, codes)

        codes = codes.to(torch.uint8)

    else:
        raise ValueError(f"Unsupported implementation={implementation}")

    return codes, scale


class Optim2State4bit(Tensor):
    tensor_attrs = ["codes", "scale1", "scale2", "qmap1", "qmap2"]

    @staticmethod
    def __new__(cls, codes1: Tensor, scale1: Tensor, qmap1: Tensor, codes2: Tensor, scale2: Tensor, qmap2: Tensor):
        return Tensor._make_wrapper_subclass(
            cls,
            codes1.shape,
            device=codes1.device,
            requires_grad=False,
        )

    def __init__(self, codes1: Tensor, scale1: Tensor, qmap1: Tensor, codes2: Tensor, scale2: Tensor, qmap2: Tensor):
        assert codes1.dtype is torch.uint8 and codes2.dtype is torch.uint8
        assert codes1.shape == codes2.shape
        assert scale1.numel() == scale2.numel()  # must use the same block_size
        self.codes = (codes1 << 4) & codes2  # packing
        self.scale1 = scale1
        self.scale2 = scale2
        self.qmap1 = qmap1
        self.qmap2 = qmap2

    @property
    def block_size(self):
        return self.codes.numel() // self.scale1.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    @classmethod
    def from_float(cls, state1: Tensor, state2: Tensor, block_size: int = 128):
        qmap1 = torch.tensor(QMAP_SIGNED, device=state1.device)
        qmap2 = torch.tensor(QMAP_UNSIGNED, device=state2.device)
        codes1, scale1 = quantize_4bit_with_qmap(state1, qmap1, block_size)
        codes2, scale2 = quantize_4bit_with_qmap(state2, qmap2, block_size)
        return cls(codes1.view(state1.shape), scale1, qmap1, codes2.view(state2.shape), scale2, qmap2)

    def dequantize(self, output_dtype=None):
        # unpack
        codes1 = self.codes >> 4
        codes2 = self.codes & 0b1111

        # torch.compile() cannot use uint8 as index
        state1 = self.qmap1[codes1.int()]
        state2 = self.qmap2[codes2.int()]

        state1 = state1.view(-1, self.block_size) * self.scale1.view(-1, 1)
        state2 = state2.view(-1, self.block_size) * self.scale2.view(-1, 1)

        dtype = output_dtype or torch.get_default_dtype()
        return state1.view(self.codes.shape).to(dtype), state2.view(self.codes.shape).to(dtype)

    def copy_(self, state1: Tensor, state2: Tensor):
        codes1, scale1 = quantize_4bit_with_qmap(state1, self.qmap1, self.block_size)
        codes2, scale2 = quantize_4bit_with_qmap(state2, self.qmap2, self.block_size)
        self.codes.copy_((codes1 << 4) & codes2)
        self.scale1.copy_(scale1)
        self.scale2.copy_(scale2)

    @classmethod
    def zeros(cls, shape, block_size: int = 128, device=None):
        shape = (shape,) if isinstance(shape, int) else shape
        codes1 = torch.full(shape, ZERO_CODE_SIGNED, dtype=torch.uint8, device=device)
        codes2 = torch.full(shape, ZERO_CODE_UNSIGNED, dtype=torch.uint8, device=device)
        qmap1 = torch.tensor(QMAP_SIGNED, device=device)
        qmap2 = torch.tensor(QMAP_UNSIGNED, device=device)
        scale1 = torch.ones(codes1.numel() // block_size, device=device)
        scale2 = torch.ones(codes2.numel() // block_size, device=device)
        return cls(codes1, scale1, qmap1, codes2, scale2, qmap2)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )
