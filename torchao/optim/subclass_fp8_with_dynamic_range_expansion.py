# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, TorchAOBaseTensor

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

DTYPE = torch.float8_e4m3fn


def quantize_fp8( input: Tensor, block_size: int):

    shape = input.shape
    input = input.view(-1, block_size)
    k = None
    SqrtMinMax = None


    scale = input.abs().amax(-1).clip(1e-12)

    # NOTE: the calculation is from the paper https://arxiv.org/abs/2410.19313
    # The idea is to align optimizer state distributions more closely
    # with the FP8 representation range, reducing the quantization error.

    k = torch.ones(input.shape[0], device=input.device)
    expand_min = torch.tensor(16.0, device=input.device).view(-1, 1)
    Rdtype = torch.tensor(
        torch.finfo(DTYPE).max * torch.finfo(DTYPE).max / 2, device=input.device
    ).view(-1, 1)

    MaxValue = (input.abs().amax(-1) + 1e-20).view(-1, 1)
    MinValue = (input.abs().amin(-1) + 1e-20).view(-1, 1)
    SqrtMinMax = torch.sqrt(MaxValue * MinValue)  # geomatric mean of max and min

    Rx = MaxValue / MinValue  # range of input max and min

    k = (
        torch.floor((torch.log2(Rdtype) / torch.log2(Rx)) * expand_min) / expand_min
    ).view(-1)  # calculating optimal value k dynamically

    scale = (MaxValue / SqrtMinMax) ** k.view(-1, 1) / torch.finfo(DTYPE).max
    input = (input.sign() * (input.abs() / SqrtMinMax) ** k.view(-1, 1)) / scale.view(-1,1)
    
    k = k.view(-1)
    SqrtMinMax = SqrtMinMax.view(-1)
    codes = input.to(DTYPE).view(-1)
    return codes.view(shape), scale.view(-1), k, SqrtMinMax


# NOTE: FP8 sign bit is redundant for unsigned optim state.
# we may investigate how to use it to increase range/precision for unsigned optim state.
# https://arxiv.org/abs/2409.12517 uses FP8 E5M2 for 2nd Adam buffer
class OptimStateFp8WithDynamicRangeExpansion(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale", "k", "sqrt_min_max"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor, k: Tensor, sqrt_min_max: Tensor):
        return Tensor._make_wrapper_subclass(cls, codes.shape, device=codes.device)

    def __init__(self, codes: Tensor, scale: Tensor, k: Tensor, sqrt_min_max: Tensor):
        """Create quantized FP8 optimizer state.

        Args
            codes: quantized FP8 E4M3FN data. Has the same shape as the original float tensor.
            scale: scale data for block-wise quantization.

        NOTE: To get block-wise scale, the original float tensor is first reshape to (-1, block_size).
        Thus, the last dimension of the original float tensor is not necessarily divisible by block size.
        Given `codes` and `scale`, `block_size` is calculated as `codes.numel() // scale.numel()`.
        """
        assert codes.dtype is DTYPE
        assert scale.ndim == 1
        self.codes = codes
        self.scale = scale
        self.block_size = codes.numel() // scale.numel()
        self.k = k 
        self.sqrt_min_max = sqrt_min_max

    def __tensor_flatten__(self):
        return self.tensor_attrs, []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[
                tensor_data_dict[name]
                for name in cls.tensor_attrs
            ],
            *tensor_attributes,
        )

    
    def dequantize(self, output_dtype=None):
        float_data = self.codes.float()
        float_data = float_data.view(-1, self.block_size) * self.scale.view(-1, 1)
        float_data = float_data.sign() * (float_data.abs() ** (1 / self.k.view(-1, 1))) 
        float_data = float_data * self.sqrt_minmax_exp.view(-1, 1)

        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data.view(self.codes.shape)

    @classmethod
    def zeros(cls, shape, block_size: int = 256, device=None):
        codes = torch.zeros(shape, dtype=DTYPE, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        k = torch.ones(codes.numel() // block_size, device=device)
        sqrt_min_max = torch.zeros(codes.numel() // block_size , device=device)
        return cls(codes, scale, k , sqrt_min_max)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )


@OptimStateFp8WithDynamicRangeExpansion.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]
    k = args[2]
    sqrt_minmax_exp = args[3]

    if isinstance(dst, OptimStateFp8WithDynamicRangeExpansion) and isinstance(src, OptimStateFp8WithDynamicRangeExpansion):
        assert dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        dst.k.copy_(src.k)
        dst.sqrt_min_max.copy_(src.sqrt_min_max)

    elif isinstance(dst, OptimStateFp8WithDynamicRangeExpansion):
        codes, scale, k, sqrt_minmax_exp= quantize_fp8(
            src, dst.block_size
        )

        dst.codes.copy_(codes)
        dst.scale.copy_(scale)
        dst.k.copy_(k)
        dst.sqrt_min_max.copy_(sqrt_minmax_exp)
        
    else:
        dst.copy_(src.dequantize())

    return dst


@OptimStateFp8WithDynamicRangeExpansion.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # ignore dtype
    device = kwargs.get("device", None)
    out = OptimStateFp8WithDynamicRangeExpansion(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
        args[0].k.to(device=device),
        args[0].sqrt_min_max.to(device=device)
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


#TODO: Check this computation
@OptimStateFp8WithDynamicRangeExpansion.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateFp8WithDynamicRangeExpansion) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local()
@OptimStateFp8WithDynamicRangeExpansion.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args
    return OptimStateFp8WithDynamicRangeExpansion(x.codes.view(shape), x.scale, x.k, x.sqrt_min_max)


@OptimStateFp8WithDynamicRangeExpansion.implements(
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
    if not isinstance(x, OptimStateFp8WithDynamicRangeExpansion):
        raise ValueError(f"expecting a OptimStateFp8 but found {type(x)}")

    # assume tensors from all ranks have the same signedness
    return OptimStateFp8WithDynamicRangeExpansion(
        func(x.codes, *args[1:], **kwargs),
        func(x.scale, *args[1:], **kwargs),
        func(x.k, *args[1:], **kwargs),
        func(x.sqrt_min_max, *args[1:], **kwargs)
    )


# required by torch.distributed.checkpoint.save
# note that we don't actually implement pin memory for this tensor subclass
# (pin_memory argument is ignored in aten._to_copy)
@OptimStateFp8WithDynamicRangeExpansion.implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    return args[0].codes.is_pinned() and args[0].scale.is_pinned() and args[0].k.is_pinned() and args[0].sqrt_min_max.is_pinned()


#TODO: need to check for this calculation, ideally shapes must be equal to scale dimension
# required by torch.distributed.checkpoint.load when world size changes i.e. re-sharding
@OptimStateFp8WithDynamicRangeExpansion.implements(aten.slice.Tensor)
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

    return OptimStateFp8WithDynamicRangeExpansion(
        x.codes[start:end],
        x.scale[start * stride // block_size : end * stride // block_size],
    )


if TORCH_VERSION_AT_LEAST_2_5:
    from torch.serialization import add_safe_globals

    add_safe_globals([OptimStateFp8WithDynamicRangeExpansion])
