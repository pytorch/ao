import math

import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _dispatch__torch_dispatch__

from .quant_utils import create_dynamic_map, scale_tensor, quantize_4bit_with_qmap, dequant_with_qmap


aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional

# https://github.com/thu-ml/low-bit-optimizers/blob/e3e2854728e498c2a606e3fdb88daa27ae94f9a6/lpmm/configs/2nd_moment_group_128.yml
# NOTE: power-1 is linear
# TODO: since QMAP_UNSIGNED is linear, perhaps doing affine quantize is faster?
QMAP_SIGNED = create_dynamic_map(True, 3, 4)
QMAP_UNSIGNED = torch.linspace(0, 1, 17)[1:].tolist()  # no zero


class OptimState4bit(Tensor):
    implements = classmethod(_implements)
    tensor_attrs = ["codes", "scale", "qmap"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool, shape):
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=codes.device,
            requires_grad=False,
        )

    def __init__(self, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool, shape):
        """Create quantized 4-bit optimizer state as proposed in https://arxiv.org/abs/2309.01507

        Args
            codes: quantized and packed 4-bit data stored as uint8.
            scale: scale data for block-wise quantization.
            qmap: lookup table that maps between quantized value (code) and float value.
            signed: whether the tensor is signed or unsigned.
            shape: shape of original float tensor.

        NOTE: To get block-wise scale, the original float tensor is first reshape to (-1, block_size).
        Thus, the last dimension of the original float tensor is not necessarily divisible by block size.
        Given `codes` and `scale`, `block_size` is calculated as `codes.numel() * 2 // scale.numel()`.
        The extra `* 2` is because `codes` is 4-bit data packed in 8-bit storage.
        """
        assert codes.dtype is torch.uint8
        assert codes.ndim == 1  # flattened buffer
        assert scale.ndim == 1
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed
        self._shape = shape
        self.block_size = codes.numel() * 2 // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.signed, self._shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    def dequantize(self, output_dtype=None):
        codes = torch.stack([self.codes >> 4, self.codes & 0b1111], dim=-1)  # unpack
        float_data = dequant_with_qmap(codes, self.qmap, self.scale)
        dtype = output_dtype or torch.get_default_dtype()
        return float_data.view(self._shape).to(dtype)

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 128, device=None):
        shape = (shape,) if isinstance(shape, int) else shape
        n_elems = math.prod(shape)

        codes = torch.zeros(n_elems // 2, dtype=torch.uint8, device=device)
        scale = torch.zeros(n_elems // block_size, device=device)
        qmap = torch.tensor(QMAP_SIGNED if signed else QMAP_UNSIGNED, device=device)
        return cls(codes, scale, qmap, signed, shape)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(signed={self.signed}, block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )

    __torch_dispatch__ = classmethod(_dispatch__torch_dispatch__)


@OptimState4bit.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimState4bit) and isinstance(src, OptimState4bit):
        assert (
            dst.signed == src.signed
            and dst.block_size == src.block_size
            and dst._shape == src._shape
        )
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        # qmap should be the same, don't need to copy

    elif isinstance(dst, OptimState4bit):
        scaled_src, scale = scale_tensor(src.view(-1), dst.block_size)
        codes = quantize_4bit_with_qmap(scaled_src, dst.qmap)
        dst.codes.copy_((codes[::2] << 4) | codes[1::2])  # packing
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimState4bit.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimState4bit) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local() and for flattening tensor
@OptimState4bit.implements(aten.view.default)
def _(func, types, args, kwargs):
    x, shape = args

    if tuple(x.shape) == tuple(shape):
        return OptimState4bit(x.codes, x.scale, x.qmap, x.signed, x._shape)

    if len(shape) == 1 and shape[0] == -1:
        return OptimState4bit(x.codes, x.scale, x.qmap, x.signed, (x.numel(),))

    raise ValueError(f"{x.__class__.__name__} only supports .view() with same shape or shape=[-1]")


# this is needed for DTensor.full_tensor()
@OptimState4bit.implements([
    c10d_functional.all_gather_into_tensor.default,
    _c10d_functional.all_gather_into_tensor.default,
    c10d_functional.wait_tensor.default,
    _c10d_functional.wait_tensor.default,
])
def _(func, types, args, kwargs):
    x = args[0]
    if not isinstance(x, OptimState4bit):
        raise ValueError(f"expecting a OptimState4bit but found {type(x)}")

    codes = func(x.codes, *args[1:], **kwargs)
    scale = func(x.scale, *args[1:], **kwargs)

    # adjust the first dim
    shape = (x._shape[0] * codes.numel() // x.codes.numel(),) + x._shape[1:]

    # assume tensors from all ranks have the same signedness
    return OptimState4bit(codes, scale, x.qmap.clone(), x.signed, shape)
