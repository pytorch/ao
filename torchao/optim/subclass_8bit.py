# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_4,
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

from .quant_utils import (
    create_dynamic_map,
    dequant_with_qmap,
    quantize_8bit_with_qmap,
    scale_tensor,
)

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

# Lazy initialization to avoid meta device issues during import
from functools import lru_cache


@lru_cache(maxsize=1)
def get_qmap_signed():
    return create_dynamic_map(signed=True)


@lru_cache(maxsize=1)
def get_qmap_unsigned():
    return create_dynamic_map(signed=False)


class OptimState8bit(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale", "qmap"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool):
        return Tensor._make_wrapper_subclass(cls, codes.shape, device=codes.device)

    def __init__(self, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool):
        """Create quantized 8-bit optimizer state as proposed in https://arxiv.org/abs/2110.02861

        Args
            codes: quantized 8-bit data stored as uint8. Has the same shape as the original float tensor.
            scale: scale data for block-wise quantization.
            qmap: lookup table that maps between quantized value (code) and float value.
            signed: whether the tensor is signed or unsigned.

        NOTE: To get block-wise scale, the original float tensor is first reshape to (-1, block_size).
        Thus, the last dimension of the original float tensor is not necessarily divisible by block size.
        Given `codes` and `scale`, `block_size` is calculated as `codes.numel() // scale.numel()`.
        """
        assert codes.dtype is torch.uint8
        assert scale.ndim == 1
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed
        self.block_size = codes.numel() // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.signed]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes
        )

    def dequantize(self, output_dtype=None):
        float_data = dequant_with_qmap(self.codes, self.qmap, self.scale)
        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 256, device=None):
        codes = torch.zeros(shape, dtype=torch.uint8, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        qmap = torch.tensor(
            get_qmap_signed() if signed else get_qmap_unsigned(), device=device
        )
        return cls(codes, scale, qmap, signed)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(signed={self.signed}, block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )


# in pre-2.4, calling .to(device, dtype) will not dispatch aten._to_copy.default when
# dtype is the same but device is different. thus, we must override .to() method instead.
if not TORCH_VERSION_AT_LEAST_2_4:

    def _to(self, *args, **kwargs):
        # ignore other args/kwargs
        device = kwargs.pop("device", None)
        return OptimState8bit(
            self.codes.to(device),
            self.scale.to(device),
            self.qmap.to(device),
            self.signed,
        )

    OptimState8bit.to = _to
    del _to  # make sure to not re-use


@OptimState8bit.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimState8bit) and isinstance(src, OptimState8bit):
        assert dst.signed == src.signed and dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        # qmap should be the same, don't need to copy

    elif isinstance(dst, OptimState8bit):
        scaled_src, scale = scale_tensor(src, dst.block_size)
        codes = quantize_8bit_with_qmap(scaled_src, dst.qmap)
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimState8bit.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # ignore dtype
    device = kwargs.get("device", None)
    out = OptimState8bit(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
        args[0].qmap.to(device=device),
        args[0].signed,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@OptimState8bit.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimState8bit) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local()
@OptimState8bit.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args
    return OptimState8bit(x.codes.view(shape), x.scale, x.qmap, x.signed)


@OptimState8bit.implements(
    [
        # required by DTensor.full_tensor()
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
        c10d_functional.wait_tensor.default,
        _c10d_functional.wait_tensor.default,
        # required by torch.distributed.checkpoint.save
        aten.detach.default,
    ]
)
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, OptimState8bit):
        raise ValueError(f"expecting a OptimState8bit but found {type(x)}")

    # assume tensors from all ranks have the same signedness
    return OptimState8bit(
        func(x.codes, *args[1:], **kwargs),
        func(x.scale, *args[1:], **kwargs),
        x.qmap.clone(),
        x.signed,
    )


# required by torch.distributed.checkpoint.save
# note that we don't actually implement pin memory for this tensor subclass
# (pin_memory argument is ignored in aten._to_copy)
@OptimState8bit.implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    return (
        args[0].codes.is_pinned()
        and args[0].scale.is_pinned()
        and args[0].qmap.is_pinned()
    )


# required by torch.distributed.checkpoint.load when world size changes i.e. re-sharding
@OptimState8bit.implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    x, dim, start, end = args[:4]
    step = args[4] if len(args) > 4 else 1

    # input validation
    if dim != 0:
        raise ValueError("Only support aten.slice along the first dim")
    if step != 1:
        raise ValueError("Only support aten.slice with step=1")

    block_size = x.block_size
    stride = math.prod(x.shape[1:])

    # for 1 increment in x along the first dim,
    # (flattened) scale will increment by stride / block_size
    if (start * stride) % block_size != 0 or (end * stride) % block_size != 0:
        raise ValueError(
            f"Invalid start or end for shape={x.shape} and block_size={block_size}. "
            f"Make sure start and end align with block boundary. "
            f"Received start={start}, end={end}."
        )

    return OptimState8bit(
        x.codes[start:end],
        x.scale[start * stride // block_size : end * stride // block_size],
        x.qmap.clone(),
        x.signed,
    )


if TORCH_VERSION_AT_LEAST_2_5:
    from torch.serialization import add_safe_globals

    add_safe_globals([OptimState8bit])
