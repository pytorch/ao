from functools import reduce
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.custom_fp_utils import _f32_to_floatx_unpacked, _floatx_unpacked_to_f32, _n_ones
from torchao.dtypes.utils import (
    Layout,
)
from torchao.quantization.quant_api import _get_linear_subclass_inserter
from dataclasses import dataclass
from torchao.dtypes.affine_quantized_tensor import AQTTensorImpl, register_layout


aten = torch.ops.aten
_ONES_TABLE = [_n_ones(i) for i in range(8)]


def _pack(x: Tensor, n_bits: int) -> Tensor:
    return reduce(torch.bitwise_or, [x[..., i::(8 // n_bits)] << (8 - (i + 1) * n_bits) for i in range(8 // n_bits)])


def _unpack(x: Tensor, n_bits: int) -> Tensor:
    return torch.stack([(x >> (8 - (i + 1) * n_bits)) & ((1 << n_bits) - 1) for i in range(8 // n_bits)], dim=-1).flatten(-2)


# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h#L87-L116
def _bit_interleave(x: Tensor, n_bits: int, undo: bool = False) -> Tensor:
    # the original code unpacks/packs the values from/to uint32 while we unpack/pack the values from/to uint8
    # thus, we need to reverse byte order within a uint32 word.
    x = x.reshape(-1, 4).flip(1)

    x = _unpack(x, n_bits)
    x = x.view(-1, 4 * (8 // n_bits))

    if not undo:
        bit_order = {
            1: [1, 5, 9, 13, 17, 21, 25, 29, 3, 7, 11, 15, 19, 23, 27, 31,
                0, 4, 8, 12, 16, 20, 24, 28, 2, 6, 10, 14, 18, 22, 26, 30],
            2: [1, 5, 9, 13, 3, 7, 11, 15, 0, 4, 8, 12, 2, 6, 10, 14],
            4: [1, 5, 3, 7, 0, 4, 2, 6],
        }[n_bits]

    else:
        # this is inverse of the above, obtained by running
        # [v.index(i) for i in range(len(v))]
        bit_order = {
            1: [16, 0, 24, 8, 17, 1, 25, 9, 18, 2, 26, 10, 19, 3, 27, 11,
                20, 4, 28, 12, 21, 5, 29, 13, 22, 6, 30, 14, 23, 7, 31, 15],
            2: [8, 0, 12, 4, 9, 1, 13, 5, 10, 2, 14, 6, 11, 3, 15, 7],
            4: [4, 0, 6, 2, 5, 1, 7, 3],
        }[n_bits]

    x = x[:, bit_order]
    x = _pack(x, n_bits)

    # reverse byte order within a uint32 word again.
    x = x.reshape(-1, 4).flip(1)
    return x.flatten()


# this is a literal adaptation of FP6-LLM ahead-of-time bit-level pre-packing
# https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/fp6_llm/csrc/utils/weight_prepacking.h
def _pack_tc_floatx(tensor: Tensor, nbits: int) -> Tensor:
    assert tensor.ndim == 2, tensor.dtype == torch.uint8
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    # Pass 1 from original code
    tensor = tensor.view(M // 64, 4, 2, 8, N // 16, 2, 8)
    tensor = tensor.permute(0, 4, 1, 5, 2, 3, 6)
    tensor = tensor.reshape(-1, 32, 2)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor.flatten()

    used_bits = 0
    fragments = []

    for y in [1, 2, 4]:
        if nbits & y:
            mask = (1 << y) - 1
            tensor_ybit = (tensor >> (nbits - used_bits - y)) & mask
            tensor_ybit = _pack(tensor_ybit, y)

            tensor_ybit = tensor_ybit.view(32, -1, 4).permute(1, 0, 2).flip(2)  # Pass 2 from original code
            tensor_ybit = _bit_interleave(tensor_ybit.flatten(), y)             # Pass 3 from original code
            fragments.append(tensor_ybit)
            used_bits += y

    return torch.cat(fragments, dim=0).view(M, -1)


# more optimized version of _pack_tc_floatx() for FP6 by merging ops
def _pack_tc_fp6(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2, tensor.dtype == torch.uint8
    M, N = tensor.shape
    assert (M % 64 == 0) and (N % 64 == 0)

    tensor = tensor.view(M // 64, 2, 2, 2, 8, N // 16, 2, 8)
    tensor = tensor.flip(3)

    tensor_2bit = (tensor >> 4) & 0b11
    tensor_2bit = tensor_2bit.permute(0, 5, 1, 4, 7, 3, 2, 6)
    tensor_2bit = _pack(tensor_2bit.flatten(), 2)

    tensor_4bit = tensor & 0b1111
    tensor_4bit = tensor_4bit.permute(0, 5, 1, 2, 4, 7, 3, 6)
    tensor_4bit = _pack(tensor_4bit.flatten(), 4)

    return torch.cat([tensor_2bit, tensor_4bit], dim=0).view(M, -1)


# currently only optimize for TC-FP6 packing
def pack_tc_floatx(tensor: Tensor, nbits: int) -> Tensor:
    if nbits == 6:
        return _pack_tc_fp6(tensor)
    return _pack_tc_floatx(tensor, nbits)


def to_scaled_tc_floatx(tensor: Tensor, ebits: int, mbits: int) -> Tuple[Tensor, Tensor]:
    # _n_ones() is not compatible with torch.compile() due to << operator
    # https://github.com/pytorch/pytorch/issues/119152
    # exp_bias = _n_ones(ebits - 1)
    # max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2 ** mbits))

    # workaround: global lookup table
    exp_bias = _ONES_TABLE[ebits - 1]
    max_normal = 2 ** (_ONES_TABLE[ebits] - exp_bias) * (_ONES_TABLE[mbits + 1] / (2 ** mbits))

    tensor = tensor.float()
    scale = tensor.abs().amax(1).clamp(min=1e-12) / max_normal
    tensor_floatx = _f32_to_floatx_unpacked(tensor / scale.view(-1, 1), ebits, mbits)
    tensor_tc_floatx = pack_tc_floatx(tensor_floatx, 1 + ebits + mbits)
    return tensor_tc_floatx, scale.half()


# inverse of _pack_tc_floatx()
def _unpack_tc_floatx(tensor: Tensor, nbits: int) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    size = tensor.numel()
    tensor = tensor.flatten()
    offset = 0
    used_bits = 0

    tensor_floatx = None

    for y in [1, 2, 4]:
        if nbits & y:
            size_ybit = size // nbits * y
            tensor_ybit = tensor[offset : offset + size_ybit]
            offset += size_ybit

            tensor_ybit = _bit_interleave(tensor_ybit, y, undo=True)            # undo Pass 3
            tensor_ybit = tensor_ybit.view(-1, 32, 4).flip(2).permute(1, 0, 2)  # undo Pass 2

            tensor_ybit = _unpack(tensor_ybit.flatten(), y)
            tensor_ybit = tensor_ybit << (nbits - used_bits - y)
            used_bits += y

            if tensor_floatx is None:
                tensor_floatx = tensor_ybit
            else:
                tensor_floatx |= tensor_ybit

    # undo Pass 1
    tensor_floatx = tensor_floatx.view(32, -1, 2).permute(1, 0, 2)
    tensor_floatx = tensor_floatx.reshape(M // 64, -1, 4, 2, 2, 8, 8)
    tensor_floatx = tensor_floatx.permute(0, 2, 4, 5, 1, 3, 6)
    tensor_floatx = tensor_floatx.reshape(M, -1)
    return tensor_floatx


# more optimized version of _unpack_tc_floatx() for FP6 by merging ops
# inverse of _unpack_tc_fp6()
def _unpack_tc_fp6(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2 and tensor.dtype == torch.uint8
    M = tensor.shape[0]
    N = tensor.shape[1] // 3 * 4
    assert (M % 64 == 0) and (N % 64 == 0)
    size_2bit = M * N // 4
    size_4bit = M * N // 2
    tensor = tensor.view(-1)
    assert tensor.numel() == size_2bit + size_4bit

    tensor_2bit, tensor_4bit = tensor.split([size_2bit, size_4bit])

    tensor_2bit = _unpack(tensor_2bit, 2)
    tensor_2bit = tensor_2bit.view(M // 64, N // 16, 2, 8, 8, 2, 2, 2)
    tensor_2bit = tensor_2bit.permute(0, 2, 6, 5, 3, 1, 7, 4)

    tensor_4bit = _unpack(tensor_4bit, 4)
    tensor_4bit = tensor_4bit.view(M // 64, N // 16, 2, 2, 8, 8, 2, 2)
    tensor_4bit = tensor_4bit.permute(0, 2, 3, 6, 4, 1, 7, 5)

    tensor_fp6 = (tensor_2bit << 4) | tensor_4bit
    tensor_fp6 = tensor_fp6.flip(3).reshape(M, N)
    return tensor_fp6


def unpack_tc_floatx(tensor: Tensor, nbits: int) -> Tensor:
    if nbits == 6:
        return _unpack_tc_fp6(tensor)
    return _unpack_tc_floatx(tensor, nbits)


def from_scaled_tc_floatx(tensor: Tensor, ebits: int, mbits: int, scale=None) -> Tensor:
    floatx_unpacked = unpack_tc_floatx(tensor, 1 + ebits + mbits)
    tensor = _floatx_unpacked_to_f32(floatx_unpacked, ebits, mbits)
    if scale is not None:
        tensor = tensor * scale.float().view(-1, 1)
    return tensor


# https://github.com/microsoft/DeepSpeed/blob/3a3a6db3332e339cc9fd94efd4982f6d60635a3d/deepspeed/inference/v2/kernels/core_ops/cuda_linear/cuda_linear.py
_SPLIT_K_MAP = [
    {  # tokens: [1, 64]
        3072: 18,
        4096: 13,
        5120: 10,
        6144: 9,
        8192: 6,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 7
    },
    {  # tokens: [65:128]
        3072: 9,
        4096: 6,
        5120: 5,
        6144: 9,
        8192: 3,
        10240: 5,
        14336: 7,
        28672: 7,
        57344: 6
    },
    {  # tokens: [129:192]
        3072: 6,
        4096: 4,
        5120: 7,
        6144: 3,
        8192: 2,
        10240: 5,
        14336: 5,
        28672: 5,
        57344: 4
    },
    {  # tokens: [193:256]
        3072: 9,
        4096: 3,
        5120: 5,
        6144: 2,
        8192: 5,
        10240: 4,
        14336: 8,
        28672: 6,
        57344: 4
    },
    {  # tokens: [257:320]
        3072: 7,
        4096: 5,
        5120: 2,
        6144: 5,
        8192: 4,
        10240: 1,
        14336: 3,
        28672: 3,
        57344: 4
    },
    {  # tokens: [321:384]
        3072: 3,
        4096: 2,
        5120: 5,
        6144: 3,
        8192: 1,
        10240: 8,
        14336: 3,
        28672: 4,
        57344: 3
    },
    {  # tokens: [385:448]
        3072: 5,
        4096: 7,
        5120: 3,
        6144: 5,
        8192: 7,
        10240: 3,
        14336: 1,
        28672: 1,
        57344: 3
    },
    {  # tokens: [449:512]
        3072: 2,
        4096: 5,
        5120: 4,
        6144: 1,
        8192: 5,
        10240: 2,
        14336: 6,
        28672: 4,
        57344: 1
    },
    {  # tokens: [513:576]
        3072: 2,
        4096: 3,
        5120: 1,
        6144: 1,
        8192: 3,
        10240: 3,
        14336: 3,
        28672: 1,
        57344: 1
    },
    {  # tokens: [577:640]
        3072: 5,
        4096: 4,
        5120: 1,
        6144: 4,
        8192: 2,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [641:704]
        3072: 3,
        4096: 1,
        5120: 2,
        6144: 2,
        8192: 1,
        10240: 2,
        14336: 1,
        28672: 1,
        57344: 1
    },
    {  # tokens: [705:768]
        3072: 3,
        4096: 1,
        5120: 3,
        6144: 2,
        8192: 1,
        10240: 1,
        14336: 1,
        28672: 1,
        57344: 1
    }
]


# quantization api integrations

@dataclass(frozen=True)
class FloatxTensorCoreLayout(Layout):
    """Layout type for FloatxTensorCoreAQTTensorImpl
    """
    ebits: int
    mbits: int

@register_layout(FloatxTensorCoreLayout)
class FloatxTensorCoreAQTTensorImpl(AQTTensorImpl):
    """FloatxTensorCoreAQTTensorImpl represents a Tensor with dtype floatx(ebits=a, mbits=b),
    it has a internal tensor field of "packed_floatx_data", which is packed from the
    uint8 unpacked data (the output of `quantize_affine_floatx` operator)

    The packing is optimized for TensorCore, from the fp6-llm paper: https://arxiv.org/abs/2401.14112
    github repo: https://github.com/usyd-fsalab/fp6_llm, now renamed to quant-llm

    At a high level packing is done by grouping bits into 1 bit fragments (shards), 2 bit fragments and
    4 bit fragments each fragments are packed separately and concatenated together.
    For example for 6 bit dtype, we can extract the first 4 bits for all elements and pack them together
    in a fragment, and extract the last 2 bits for all elements and pack them into fragment, in the end
    we concatenate the fragments together.

    If original Tensor shape is (M, N), and the data is in nbit, the shape of the packed data will be
    (M, N // 8 * nbit)

    FloatxTensorCoreAQTTensorImpl.from_plain takes an unpacked uint8 floatx Tensor of shape (M, N), with format of
    (zero padding bits + sign bit + exponent bits + mantissa bits), e.g. 00SEEEMM for fp6_e3_m2
    it will then pack the weight and instantiate the FloatxTensorCoreAQTTensorImpl tensor
    FloatxTensorCoreAQTTensorImpl.__init__() takes a packed floatx Tensor of shape (M, N // 8 * nbit)
    """
    def __new__(
        cls,
        packed_floatx_data: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        assert packed_floatx_data.ndim == 2
        assert packed_floatx_data.dtype == torch.uint8
        shape = (packed_floatx_data.shape[0], packed_floatx_data.shape[1] // (1 + _layout.ebits + _layout.mbits) * 8)
        kwargs = {}
        kwargs["device"] = packed_floatx_data.device
        kwargs["layout"] = (
            kwargs.get("layout") if kwargs.get("layout", False) else packed_floatx_data.layout
        )
        kwargs["dtype"] = packed_floatx_data.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        packed_floatx_data: torch.Tensor,
        scale: torch.Tensor,
        _layout: Layout,
    ):
        self.packed_floatx_data = packed_floatx_data
        self.scale = scale
        self._layout = _layout

    def __tensor_flatten__(self):
        return ["packed_floatx_data", "scale"], [self._layout]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        packed_floatx_data, scale = tensor_data_dict["packed_floatx_data"], tensor_data_dict["scale"]
        _layout, = tensor_attributes
        return cls(packed_floatx_data, scale, _layout)

    def get_plain(self) -> Tuple[torch.Tensor, torch.Tensor]:
        unpacked_floatx_data = unpack_tc_floatx(self.packed_floatx_data, 1 + self._layout.ebits + self._layout.mbits)
        return unpacked_floatx_data, self.scale

    @classmethod
    def from_plain(
        cls,
        unpacked_floatx_data: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        _layout: Layout,
    ):
        """
        Format for `unpacked_floatx_data` will be:
        zero padding bits | sign bit | exponent bits | mantissa bits

        For example for fp6_e3_m2, the format will be: `00SEEEMM`, where S is sign bit, E is exponent
        bit, M is mantissa bit
        """
        assert isinstance(_layout, FloatxTensorCoreLayout)
        packed_floatx_data = pack_tc_floatx(unpacked_floatx_data, 1 + _layout.ebits + _layout.mbits)
        return cls(packed_floatx_data, scale, _layout)

    def __repr__(self):
        unpacked_floatx_data, scale = self.get_plain()
        _layout = self.get_layout()
        return f"{self.__class__.__name__}(unpacked_floatx_data={unpacked_floatx_data}, scale={scale}, _layout={_layout})"

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.packed_floatx_data),
            fn(self.scale),
            self._layout,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        device = kwargs.pop("device")
        return self.__class__(
            self.packed_floatx_data.to(device),
            self.scale.to(device),
            self._layout,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        if func is aten.detach.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )
        elif func is aten.clone.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
            )
        elif func is aten._to_copy.default:
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(lambda x: x.to(device=kwargs.pop("device", None))),
            )

        raise NotImplementedError(
            f"FloatxTensorCoreAQTTensorImpl dispatch: attempting to run {func}, this is not supported"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def get_layout(self) -> Layout:
        return self._layout
