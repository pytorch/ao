import torch
from torch import Tensor
from typing import Optional
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, TorchAOBaseTensor

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

DTYPE = torch.float8_e4m3fn


def quantize_fp8(input: Tensor, block_size: int, dynamic_range_expansion: bool):

    shape = input.shape
    input = input.view(-1, block_size)
    k =  aten.ones(input.shape[0], dtype=DTYPE, device=input.device) if dynamic_range_expansion else None

    if dynamic_range_expansion:
        Rdtype = torch.finfo(DTYPE).max / torch.finfo(DTYPE).min # calculate the range of the dtype
        Rx = input.abs().amax(-1).clip(1e-12) / input.abs().amin(-1).clip(1e-12) # range of input max and min
        k = torch.log(Rdtype) / torch.log(Rx) # k calcuated
        input = input.sign() * (input.abs() ** k.view(-1, 1))

    scale = input.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
    input = input / scale.view(-1, 1)
    codes = input.to(DTYPE).view(-1)

    return codes.view(shape), scale, k


# NOTE: FP8 sign bit is redundant for unsigned optim state.
# we may investigate how to use it to increase range/precision for unsigned optim state.
# https://arxiv.org/abs/2409.12517 uses FP8 E5M2 for 2nd Adam buffer
class OptimStateFp8(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor):
        return Tensor._make_wrapper_subclass(cls, codes.shape, device=codes.device)

    def __init__(self, codes: Tensor, scale: Tensor, k: Optional[Tensor] =None):
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
        self.k = k
        self.block_size = codes.numel() // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, []

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes
        )

    def dequantize(self, output_dtype=None):
        float_data = self.codes.float()
        float_data = float_data.view(-1, self.block_size) * self.scale.view(-1, 1)

        if self.dynamic_range_expansion:
            float_data = float_data.view(-1, self.block_size)
            float_data =  float_data ** (1 / self.k.view(-1, 1))
        
        if output_dtype is not None:
            float_data = float_data.to(output_dtype)

        return float_data.view(self.codes.shape)

    @classmethod
    def zeros(cls, shape, block_size: int = 256, device=None):
        codes = torch.zeros(shape, dtype=DTYPE, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        return cls(codes, scale)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )


@OptimStateFp8.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimStateFp8) and isinstance(src, OptimStateFp8):
        assert dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        dst.k.copy_(src.k)

    elif isinstance(dst, OptimStateFp8):

        codes, scale, k = quantize_fp8(codes, dst.block_size, dst.dynamic_range_expansion)
        
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)
        dst.k.copy_(k)
    else:
        dst.copy_(src.dequantize())

    return dst


@OptimStateFp8.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # ignore dtype
    device = kwargs.get("device", None)
    out = OptimStateFp8(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@OptimStateFp8.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateFp8) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local()
@OptimStateFp8.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args
    return OptimStateFp8(x.codes.view(shape), x.scale)


# this is needed for DTensor.full_tensor()
@OptimStateFp8.implements(
    [
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
        c10d_functional.wait_tensor.default,
        _c10d_functional.wait_tensor.default,
    ]
)
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, OptimStateFp8):
        raise ValueError(f"expecting a OptimStateFp8 but found {type(x)}")

    # assume tensors from all ranks have the same signedness
    return OptimStateFp8(
        func(x.codes, *args[1:], **kwargs),
        func(x.scale, *args[1:], **kwargs),
    )
