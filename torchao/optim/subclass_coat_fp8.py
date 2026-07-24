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

# Dynamic range of E4M3FN: largest representable (448) divided by smallest positive
# subnormal (2 ** -9 = 1 / 512). See COAT https://arxiv.org/abs/2410.19313, Section 4.1.
_RANGE_FP8_E4M3 = torch.finfo(DTYPE).max * 512  # 448 * 512 = 229376
_LOG_RANGE_FP8_E4M3 = math.log(_RANGE_FP8_E4M3)


def quantize_coat_fp8(input: Tensor, block_size: int):
    """Block-wise FP8 quantization with COAT dynamic range expansion.

    The dynamic range of optimizer states is typically much smaller than FP8's
    representable range, so naive FP8 quantization wastes most of the available
    exponent bits. COAT (https://arxiv.org/abs/2410.19313) first applies an
    expansion function ``f(x) = sign(x) * |x| ** k`` that stretches each block's
    dynamic range to match FP8's, then quantizes. ``k`` is chosen per block so
    that ``R_x ** k == R_fp8`` where ``R`` denotes a dynamic range (max abs / min
    abs). We only ever expand (``k >= 1``); a block whose dynamic range already
    exceeds FP8's is left untouched.

    To keep the expanded values centered (and avoid FP32 over/underflow when
    ``k`` is large), each block is first divided by the geometric mean of its
    min and max magnitudes before expansion. Both ``k`` and that geometric mean
    are stored so dequantization can invert the transform.
    """
    shape = input.shape
    input = input.view(-1, block_size)
    abs_input = input.abs()

    amax = abs_input.amax(-1).clip(1e-12)
    # smallest non-zero magnitude per block; treat exact zeros as amax so they
    # don't collapse the dynamic range to infinity.
    amin = torch.where(abs_input > 0, abs_input, amax.view(-1, 1)).amin(-1).clip(1e-12)

    # dynamic range per block, always >= 1
    log_range = (amax / amin).log()
    # k = log(R_fp8) / log(R_x), only expand (k >= 1), never compress
    k = torch.where(
        log_range > 0,
        _LOG_RANGE_FP8_E4M3 / log_range.clip(1e-12),
        torch.ones_like(log_range),
    ).clip(1.0)
    # geometric mean of the magnitude range, used to center the block
    sqrt_minmax = (amin * amax).sqrt()

    centered = input / sqrt_minmax.view(-1, 1)
    expanded = centered.sign() * centered.abs().pow(k.view(-1, 1))

    scale = expanded.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
    codes = (expanded / scale.view(-1, 1)).to(DTYPE).view(-1)
    return codes.view(shape), scale, k, sqrt_minmax


class OptimStateCoatFp8(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale", "k", "sqrt_minmax"]

    # dtype only acts as an appearance dtype to work with the rest of PyTorch
    @staticmethod
    def __new__(
        cls,
        codes: Tensor,
        scale: Tensor,
        k: Tensor,
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
        k: Tensor,
        sqrt_minmax: Tensor,
        dtype: torch.dtype | None = None,
    ):
        """Create FP8 optimizer state quantized with COAT dynamic range expansion.

        Args
            codes: quantized FP8 E4M3FN data. Has the same shape as the original float tensor.
            scale: block-wise quantization scale (1D).
            k: block-wise dynamic range expansion exponent (1D).
            sqrt_minmax: block-wise geometric mean of the magnitude range, used to
                center each block before expansion (1D).

        NOTE: To get block-wise statistics, the original float tensor is first reshaped
        to (-1, block_size). Thus, the last dimension of the original float tensor is not
        necessarily divisible by block size. Given `codes` and `scale`, `block_size` is
        calculated as `codes.numel() // scale.numel()`.
        """
        assert codes.dtype is DTYPE
        assert scale.ndim == 1
        assert k.ndim == 1
        assert sqrt_minmax.ndim == 1
        self.codes = codes
        self.scale = scale
        self.k = k
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
        expanded = self.codes.float().view(-1, self.block_size) * self.scale.view(-1, 1)
        # invert the expansion: f^-1(x) = sign(x) * |x| ** (1 / k)
        centered = expanded.sign() * expanded.abs().pow((1.0 / self.k).view(-1, 1))
        float_data = centered * self.sqrt_minmax.view(-1, 1)
        float_data = float_data.view(self.codes.shape)

        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data

    @classmethod
    def zeros(
        cls,
        shape,
        block_size: int = 256,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        codes = torch.zeros(shape, dtype=DTYPE, device=device)
        n_blocks = codes.numel() // block_size
        scale = torch.zeros(n_blocks, device=device)
        # k=1 and sqrt_minmax=1 form an identity transform, so an all-zero state
        # round-trips to zeros.
        k = torch.ones(n_blocks, device=device)
        sqrt_minmax = torch.ones(n_blocks, device=device)
        return cls(codes, scale, k, sqrt_minmax, dtype=dtype)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )


@OptimStateCoatFp8.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimStateCoatFp8) and isinstance(src, OptimStateCoatFp8):
        assert dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        dst.k.copy_(src.k)
        dst.sqrt_minmax.copy_(src.sqrt_minmax)

    elif isinstance(dst, OptimStateCoatFp8):
        codes, scale, k, sqrt_minmax = quantize_coat_fp8(src, dst.block_size)
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)
        dst.k.copy_(k)
        dst.sqrt_minmax.copy_(sqrt_minmax)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimStateCoatFp8.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # only change the appearance dtype
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", None)
    out = OptimStateCoatFp8(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
        args[0].k.to(device=device),
        args[0].sqrt_minmax.to(device=device),
        dtype=dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@OptimStateCoatFp8.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateCoatFp8) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local()
@OptimStateCoatFp8.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args
    return OptimStateCoatFp8(x.codes.view(shape), x.scale, x.k, x.sqrt_minmax)


# Build the list of c10d operations to implement
_optim_state_coat_fp8_c10d_ops = [
    # required by DTensor.full_tensor()
    c10d_functional.all_gather_into_tensor.default,
    _c10d_functional.all_gather_into_tensor.default,
    c10d_functional.wait_tensor.default,
    _c10d_functional.wait_tensor.default,
    # required by torch.distributed.checkpoint.save
    aten.detach.default,
]
# _wrap_tensor_autograd was added in PyTorch 2.11.0.dev
if torch_version_at_least("2.11.0.dev"):
    _optim_state_coat_fp8_c10d_ops.append(
        _c10d_functional._wrap_tensor_autograd.default
    )


@OptimStateCoatFp8.implements(_optim_state_coat_fp8_c10d_ops)
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, OptimStateCoatFp8):
        raise ValueError(f"expecting a OptimStateCoatFp8 but found {type(x)}")

    return OptimStateCoatFp8(
        func(x.codes, *args[1:], **kwargs),
        func(x.scale, *args[1:], **kwargs),
        func(x.k, *args[1:], **kwargs),
        func(x.sqrt_minmax, *args[1:], **kwargs),
    )


# required by torch.distributed.checkpoint.save
# note that we don't actually implement pin memory for this tensor subclass
# (pin_memory argument is ignored in aten._to_copy)
@OptimStateCoatFp8.implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    return (
        args[0].codes.is_pinned()
        and args[0].scale.is_pinned()
        and args[0].k.is_pinned()
        and args[0].sqrt_minmax.is_pinned()
    )


# required by torch.distributed.checkpoint.load when world size changes i.e. re-sharding
@OptimStateCoatFp8.implements(aten.slice.Tensor)
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

    block_start = start * stride // block_size
    block_end = end * stride // block_size
    return OptimStateCoatFp8(
        x.codes[start:end],
        x.scale[block_start:block_end],
        x.k[block_start:block_end],
        x.sqrt_minmax[block_start:block_end],
    )


add_safe_globals([OptimStateCoatFp8])
