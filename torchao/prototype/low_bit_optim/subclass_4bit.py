import math

import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE

from .quant_utils import create_dynamic_map, scale_tensor, quantize_4bit_with_qmap, dequant_with_qmap


aten = torch.ops.aten


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
        assert codes.dtype is torch.uint8
        assert codes.ndim == 1  # flattened buffer
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed
        self._shape = shape

    @property
    def block_size(self):
        return self.codes.numel() * 2 // self.scale.numel()

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
        dst.codes.copy_((codes[::2] << 4) & codes[1::2])  # packing
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimState4bit.implements(aten.lerp.Scalar)
def _(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, OptimState4bit) else x for x in args]
    return func(*args, **kwargs)
