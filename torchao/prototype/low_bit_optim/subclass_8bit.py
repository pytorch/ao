import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE

from .quant_utils import create_dynamic_map, scale_tensor, quantize_8bit_with_qmap, dequant_with_qmap


aten = torch.ops.aten

QMAP_SIGNED = create_dynamic_map(signed=True)
QMAP_UNSIGNED = create_dynamic_map(signed=False)


# dynamic tree quantization
# https://arxiv.org/pdf/1511.04561
# https://arxiv.org/abs/2110.02861
class OptimState8bit(Tensor):
    implements = classmethod(_implements)
    tensor_attrs = ["codes", "scale", "qmap"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool):
        return Tensor._make_wrapper_subclass(
            cls,
            codes.shape,
            device=codes.device,
            requires_grad=False,
        )

    def __init__(self, codes: Tensor, scale: Tensor, qmap: Tensor, signed: bool):
        assert codes.dtype is torch.uint8
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed

    @property
    def block_size(self):
        return self.codes.numel() // self.scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.signed]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    def dequantize(self, output_dtype=None):
        dtype = output_dtype or torch.get_default_dtype()
        return dequant_with_qmap(self.codes, self.qmap, self.scale).to(dtype)

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 2048, device=None):
        codes = torch.zeros(shape, dtype=torch.uint8, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        qmap = torch.tensor(QMAP_SIGNED if signed else QMAP_UNSIGNED, device=device)
        return cls(codes, scale, qmap, signed)

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


@OptimState8bit.implements(aten.copy_.default)
def _(func, *args, **kwargs):
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


@OptimState8bit.implements(aten.lerp.Scalar)
def _(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, OptimState8bit) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local()
@OptimState8bit.implements(aten.view.default)
def _(func, *args, **kwargs):
    x, shape = args
    return OptimState8bit(x.codes.view(shape), x.scale, x.qmap, x.signed)
