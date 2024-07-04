import math
from typing import Tuple

import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE

from .subclass_8bit import create_dynamic_map


aten = torch.ops.aten


# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/configs/default.yml
# NOTE: power-1 is linear
QMAP_SIGNED = create_dynamic_map(True, 3, 4)
QMAP_UNSIGNED = torch.linspace(0, 1, 17)[1:].tolist()  # no zero

ZERO_CODE_SIGNED = QMAP_SIGNED.index(0)
ZERO_CODE_UNSIGNED = 0  # nearest to zero


# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/functional.py#L521-L531
def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result


# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/functional.py#L512-L518
def get_sm3_statistics(x):
    qx = x.abs()
    max_dims = []
    for i in range(x.dim()):
        nu_max = _max_reduce_except_dim(qx, i)
        max_dims.append(nu_max)
    return max_dims


# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/functional.py#L503C1-L509C24
def _compute_sm3_scale_tensor(max_dims):
    rank = len(max_dims)
    scale_tensor = max_dims[0].clone()
    for i in range(1, rank):
        # We rely on broadcasting to get the proper end shape.
        scale_tensor = torch.min(scale_tensor, max_dims[i])
    return scale_tensor


def quantize_4bit_with_qmap(input: Tensor, qmap: Tensor, block_size: int, use_rank1_scale: bool, implementation: int = 1):
    # https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/functional.py#L190-L203
    if use_rank1_scale:
        max_dims = get_sm3_statistics(input)
        st = _compute_sm3_scale_tensor(max_dims)
        input = input / st
        scale = torch.cat([x.view(-1) for x in max_dims], dim=0)

    else:
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
        codes_up = (codes + 1).clip(max=15)
        val_down = qmap[codes]
        val_up = qmap[codes_up]
        residual = input - val_down
        codes = torch.where(residual >= (val_up - val_down) * 0.5, codes_up, codes)

        codes = codes.to(torch.uint8)

    else:
        raise ValueError(f"Unsupported implementation={implementation}")

    # packing
    codes1, codes2 = codes.chunk(2, 0)
    codes = (codes1 << 4) | codes2

    return codes, scale


def _use_rank1_scale(signed: bool, shape: Tuple[int, ...]):
    return (not signed) and (len(shape) > 1)


class OptimState4bit(Tensor):
    implements = classmethod(_implements)
    tensor_attrs = ["codes", "scale", "qmap"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool, block_size: int, shape):
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=codes.device,
            requires_grad=False,
        )

    def __init__(self, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool, block_size: int, shape):
        assert codes.dtype is torch.uint8
        assert codes.ndim == 1  # flattened buffer
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed
        self.block_size = block_size
        self._shape = shape

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.signed, self.block_size, self._shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    @classmethod
    def from_float(cls, input_float: Tensor, signed: bool = True, block_size: int = 128):
        qmap = torch.tensor(QMAP_SIGNED if signed else QMAP_UNSIGNED, device=input_float.device)
        codes, scale = quantize_4bit_with_qmap(input_float, qmap, block_size, _use_rank1_scale(signed, input_float.shape))
        return cls(codes, scale, qmap, signed, block_size, input_float.shape)

    def dequantize(self, output_dtype=None):
        # unpack
        codes1 = self.codes >> 4
        codes2 = self.codes & 0b1111
        codes = torch.stack([codes1, codes2], -1)

        # torch.compile() cannot use uint8 as index
        float_data = self.qmap[codes.int()]

        if _use_rank1_scale(self.signed, self._shape):
            max_dims = self.scale.split(self._shape)
            scale = max_dims[0]
            for i in range(1, len(max_dims)):
                scale = torch.maximum(scale.unsqueeze(-1), max_dims[i])
            float_data = float_data.view(self._shape) * scale

        else:
            float_data = float_data.view(-1, self.block_size) * self.scale.view(-1, 1)
            float_data = float_data.view(self._shape)

        dtype = output_dtype or torch.get_default_dtype()
        return float_data.to(dtype)

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 128, device=None):
        shape = (shape,) if isinstance(shape, int) else shape
        n_elems = math.prod(shape)
        zero_code = ZERO_CODE_SIGNED if signed else ZERO_CODE_UNSIGNED
        zero_code = (zero_code << 4) | zero_code  # packing

        codes = torch.full((n_elems // 2,), zero_code, dtype=torch.uint8, device=device)
        qmap = torch.tensor(QMAP_SIGNED if signed else QMAP_UNSIGNED, device=device)

        scale_size = sum(shape) if _use_rank1_scale(signed, shape) else (n_elems // block_size)
        scale = torch.full((scale_size,), 1e-8, device=device)
        return cls(codes, scale, qmap, signed, block_size, shape)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(signed={self.signed}, block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


@OptimState4bit.implements(aten.copy_.default)
def _(func, *args, **kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimState4bit) and isinstance(src, OptimState4bit):
        assert dst.signed == src.signed and dst._shape == src._shape
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        # qmap should be the same, don't need to copy

    elif isinstance(dst, OptimState4bit):
        codes, scale = quantize_4bit_with_qmap(src, dst.qmap, dst.block_size, _use_rank1_scale(dst.signed, dst._shape))
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimState4bit.implements(aten.lerp.Scalar)
def _(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, OptimState4bit) else x for x in args]
    return func(*args, **kwargs)


# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/config.py#L37
# only apply quantization for tensor with more than 4096 values
# TODO: also skip 1D tensor? e.g. biases and norm scales
def maybe_new_4bit_zero_buffer(p: Tensor, signed: bool = True, block_size: int = 128):
    if p.numel() >= 4096 and p.numel() % block_size == 0:
        out = OptimState4bit.zeros(p.shape, signed, block_size, device=p.device)
    else:
        out = torch.zeros_like(p)
    return out
