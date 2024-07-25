import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE


aten = torch.ops.aten
DTYPE = torch.float8_e4m3fn


def quantize_fp8(input: Tensor, block_size: int):
    shape = input.shape
    input = input.view(-1, block_size)
    scale = input.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max
    input = input / scale.view(-1, 1)
    codes = input.to(DTYPE).view(-1)
    return codes.view(shape), scale


# NOTE: FP8 sign bit is redundant for unsigned optim state.
# we may investigate how to use it to increase range/precision for unsigned optim state.
class OptimStateFp8(Tensor):
    implements = classmethod(_implements)
    tensor_attrs = ["codes", "scale"]

    @staticmethod
    def __new__(cls, codes: Tensor, scale: Tensor):
        return Tensor._make_wrapper_subclass(
            cls,
            codes.shape,
            device=codes.device,
            requires_grad=False,
        )

    def __init__(self, codes: Tensor, scale: Tensor):
        assert codes.dtype is DTYPE
        self.codes = codes
        self.scale = scale

    @property
    def block_size(self):
        return self.codes.numel() // self.scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes)

    def dequantize(self, output_dtype=None):
        float_data = self.codes.float()
        float_data = float_data.view(-1, self.block_size) * self.scale.view(-1, 1)

        dtype = output_dtype or torch.get_default_dtype()
        return float_data.view(self.codes.shape).to(dtype)

    @classmethod
    def zeros(cls, shape, block_size: int = 2048, device=None):
        codes = torch.zeros(shape, dtype=DTYPE, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        return cls(codes, scale)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, device={self.device}, requires_grad={self.requires_grad})"
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


@OptimStateFp8.implements(aten.copy_.default)
def _(func, *args, **kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimStateFp8) and isinstance(src, OptimStateFp8):
        assert dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)

    elif isinstance(dst, OptimStateFp8):
        codes, scale = quantize_fp8(src, dst.block_size)
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimStateFp8.implements(aten.lerp.Scalar)
def _(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, OptimStateFp8) else x for x in args]
    return func(*args, **kwargs)


# this is needed for DTensor.from_local()
@OptimStateFp8.implements(aten.view.default)
def _(func, *args, **kwargs):
    x, shape = args
    return OptimStateFp8(x.codes.view(shape), x.scale)
