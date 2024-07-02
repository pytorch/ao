import torch
from torch import Tensor
from torchao.dtypes.utils import _implements, _ATEN_OP_OR_TORCH_FN_TABLE


aten = torch.ops.aten


# https://github.com/TimDettmers/bitsandbytes/blob/dada530149212d64d4b69534716202659ef37ec8/bitsandbytes/functional.py#L339-L391
def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 1)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return data


QMAP_SIGNED = create_dynamic_map(signed=True)
QMAP_UNSIGNED = create_dynamic_map(signed=False)

ZERO_CODE_SIGNED = QMAP_SIGNED.index(0)
ZERO_CODE_UNSIGNED = QMAP_UNSIGNED.index(0)


def quantize_8bit_with_qmap(input: Tensor, qmap: Tensor, block_size: int, implementation: int = 1):
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
        codes = torch.where(input >= qmap[128], 128, 0)
        codes += torch.where(input >= qmap[codes + 64], 64, 0)
        codes += torch.where(input >= qmap[codes + 32], 32, 0)
        codes += torch.where(input >= qmap[codes + 16], 16, 0)
        codes += torch.where(input >= qmap[codes + 8], 8, 0)
        codes += torch.where(input >= qmap[codes + 4], 4, 0)
        codes += torch.where(input >= qmap[codes + 2], 2, 0)
        codes += torch.where(input >= qmap[codes + 1], 1, 0)

        # rounding
        codes_up = (codes + 1).clip(max=255)
        val_down = qmap[codes]
        val_up = qmap[codes_up]
        residual = input - val_down
        codes = torch.where(residual >= (val_up - val_down) * 0.5, codes_up, codes)

        codes = codes.to(torch.uint8)

    else:
        raise ValueError(f"Unsupported implementation={implementation}")

    return codes, scale


# dynamic tree quantization
# https://arxiv.org/pdf/1511.04561
# https://arxiv.org/abs/2110.02861
class DTQ8bit(Tensor):
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

    @classmethod
    def from_float(cls, input_float: Tensor, signed: bool = True, block_size: int = 2048):
        qmap = torch.tensor(QMAP_SIGNED if signed else QMAP_UNSIGNED, device=input_float.device)
        codes, scale = quantize_8bit_with_qmap(input_float, qmap, block_size)
        return cls(codes.view(input_float.shape), scale, qmap, signed)

    def dequantize(self, output_dtype=None):
        # torch.compile() cannot use uint8 as index
        float_data = self.qmap[self.codes.int()]
        float_data = float_data.view(-1, self.block_size) * self.scale.view(-1, 1)

        dtype = output_dtype or torch.get_default_dtype()
        return float_data.view(self.codes.shape).to(dtype)

    @classmethod
    def zeros(cls, shape, signed: bool = True, block_size: int = 2048, device=None):
        shape = (shape,) if isinstance(shape, int) else shape
        codes = torch.full(shape, ZERO_CODE_SIGNED if signed else ZERO_CODE_UNSIGNED, dtype=torch.uint8, device=device)
        qmap = torch.tensor(QMAP_SIGNED if signed else QMAP_UNSIGNED, device=device)
        scale = torch.ones(codes.numel() // block_size, device=device)
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


@DTQ8bit.implements(aten.copy_.default)
def _(func, *args, **kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, DTQ8bit) and isinstance(src, DTQ8bit):
        assert dst.signed == src.signed
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        # qmap should be the same, don't need to copy

    elif isinstance(dst, DTQ8bit):
        codes, scale = quantize_8bit_with_qmap(src, dst.qmap, dst.block_size)
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@DTQ8bit.implements(aten.lerp.Scalar)
def _(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, DTQ8bit) else x for x in args]
    return func(*args, **kwargs)
