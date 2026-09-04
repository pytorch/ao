# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch import Tensor
from torch.serialization import add_safe_globals
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor, torch_version_at_least

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

DTYPE = torch.float8_e4m3fn
_EXPANSION_STEP = 16
_QUANT_MIN_VAL = 1e-30


def quantize_fp8_coat(input: Tensor, block_size: int):
    """Quantize optimizer state with COAT Dynamic Range Expansion.

    COAT applies a sign-preserving power transform to each block before FP8
    quantization. ``sqrt_minmax`` recenters the block around one to avoid
    overflow/underflow while applying the power. Zeros are excluded when
    finding the minimum magnitude, as in the reference CUDA kernel.
    """
    shape = input.shape
    input = input.float().view(-1, block_size)
    abs_input = input.abs()

    block_max = abs_input.amax(-1)
    block_min = torch.where(abs_input > 0, abs_input, torch.inf).amin(-1)
    block_min = torch.where(torch.isfinite(block_min), block_min, block_max)

    block_max = block_max + _QUANT_MIN_VAL
    block_min = block_min + _QUANT_MIN_VAL
    sqrt_minmax = block_max.sqrt() * block_min.sqrt()
    ratio = block_max / block_min

    ratio_upper_bound = torch.finfo(DTYPE).max**2 / 2
    expansion = torch.floor(
        math.log2(ratio_upper_bound) / torch.log2(ratio) * _EXPANSION_STEP
    ) / _EXPANSION_STEP
    # A constant (including all-zero) block already round-trips exactly, and
    # log2(1) would otherwise produce an infinite exponent.
    expansion = torch.where(ratio > 1, expansion, torch.ones_like(expansion))
    expansion = expansion.nan_to_num(
        nan=1.0,
        posinf=1.0,
        neginf=1 / _EXPANSION_STEP,
    ).clamp(min=1 / _EXPANSION_STEP)

    expanded = input.sign() * (abs_input / sqrt_minmax.view(-1, 1)).pow(
        expansion.view(-1, 1)
    )
    scale = expanded.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
    codes = (expanded / scale.view(-1, 1)).to(DTYPE).view(shape)
    return codes, scale, expansion, sqrt_minmax


class OptimStateFp8Coat(TorchAOBaseTensor):
    """FP8 optimizer state using COAT Dynamic Range Expansion."""

    tensor_attrs = ["codes", "scale", "expansion", "sqrt_minmax"]

    @staticmethod
    def __new__(
        cls,
        codes: Tensor,
        scale: Tensor,
        expansion: Tensor,
        sqrt_minmax: Tensor,
        dtype: torch.dtype | None = None,
    ):
        return Tensor._make_wrapper_subclass(
            cls, codes.shape, device=codes.device, dtype=dtype
        )

    def __init__(
        self,
        codes: Tensor,
        scale: Tensor,
        expansion: Tensor,
        sqrt_minmax: Tensor,
        dtype: torch.dtype | None = None,
    ):
        assert codes.dtype is DTYPE
        assert scale.ndim == expansion.ndim == sqrt_minmax.ndim == 1
        assert scale.shape == expansion.shape == sqrt_minmax.shape
        self.codes = codes
        self.scale = scale
        self.expansion = expansion
        self.sqrt_minmax = sqrt_minmax
        self.block_size = codes.numel() // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes
        )

    def dequantize(self, output_dtype=None):
        float_data = self.codes.float().view(-1, self.block_size)
        float_data = float_data * self.scale.view(-1, 1)
        float_data = float_data.sign() * float_data.abs().pow(
            self.expansion.reciprocal().view(-1, 1)
        )
        float_data = float_data * self.sqrt_minmax.view(-1, 1)

        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data.view(self.codes.shape)

    @classmethod
    def zeros(
        cls,
        shape,
        block_size: int = 256,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        codes = torch.zeros(shape, dtype=DTYPE, device=device)
        metadata_shape = (codes.numel() // block_size,)
        scale = torch.zeros(metadata_shape, device=device)
        expansion = torch.ones(metadata_shape, device=device)
        sqrt_minmax = torch.ones(metadata_shape, device=device)
        return cls(codes, scale, expansion, sqrt_minmax, dtype=dtype)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )


@OptimStateFp8Coat.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimStateFp8Coat) and isinstance(src, OptimStateFp8Coat):
        assert dst.block_size == src.block_size
        for attr in OptimStateFp8Coat.tensor_attrs:
            getattr(dst, attr).copy_(getattr(src, attr))
    elif isinstance(dst, OptimStateFp8Coat):
        values = quantize_fp8_coat(src, dst.block_size)
        for attr, value in zip(OptimStateFp8Coat.tensor_attrs, values):
            getattr(dst, attr).copy_(value)
    else:
        dst.copy_(src.dequantize())

    return dst


@OptimStateFp8Coat.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", None)
    out = OptimStateFp8Coat(
        *[
            getattr(args[0], attr).to(device=device)
            for attr in OptimStateFp8Coat.tensor_attrs
        ],
        dtype=dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@OptimStateFp8Coat.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateFp8Coat) else x for x in args]
    return func(*args, **kwargs)


@OptimStateFp8Coat.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args
    return OptimStateFp8Coat(
        x.codes.view(shape),
        x.scale,
        x.expansion,
        x.sqrt_minmax,
        dtype=x.dtype,
    )


_optim_state_fp8_coat_c10d_ops = [
    c10d_functional.all_gather_into_tensor.default,
    _c10d_functional.all_gather_into_tensor.default,
    c10d_functional.wait_tensor.default,
    _c10d_functional.wait_tensor.default,
    aten.detach.default,
]
if torch_version_at_least("2.11.0.dev"):
    _optim_state_fp8_coat_c10d_ops.append(
        _c10d_functional._wrap_tensor_autograd.default
    )


@OptimStateFp8Coat.implements(_optim_state_fp8_coat_c10d_ops)
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, OptimStateFp8Coat):
        raise ValueError(f"expecting a OptimStateFp8Coat but found {type(x)}")
    return OptimStateFp8Coat(
        *[
            func(getattr(x, attr), *args[1:], **kwargs)
            for attr in OptimStateFp8Coat.tensor_attrs
        ],
        dtype=x.dtype,
    )


@OptimStateFp8Coat.implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    return all(getattr(args[0], attr).is_pinned() for attr in args[0].tensor_attrs)


@OptimStateFp8Coat.implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    x, dim, start, end = args[:4]
    step = args[4] if len(args) > 4 else 1

    if dim != 0:
        raise ValueError("Only support aten.slice along the first dim")
    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    block_size = x.block_size
    stride = math.prod(x.shape[1:])
    if (start * stride) % block_size != 0 or (end * stride) % block_size != 0:
        raise ValueError(
            f"Invalid start or end for shape={x.shape} and block_size={block_size}. "
            "Make sure start and end align with block boundary. "
            f"Received start={start}, end={end}."
        )

    metadata_slice = slice(
        start * stride // block_size, end * stride // block_size
    )
    return OptimStateFp8Coat(
        x.codes[start:end],
        x.scale[metadata_slice],
        x.expansion[metadata_slice],
        x.sqrt_minmax[metadata_slice],
        dtype=x.dtype,
    )


add_safe_globals([OptimStateFp8Coat])
