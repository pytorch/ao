import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch.library import Library, impl

from torchao.utils import fill_defaults


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


# from
# https://github.com/drisspg/transformer_nuggets/blob/9ad3a7fc552a954eb702ade0e276b8d8e09c3db6/transformer_nuggets/quant/qlora.py#L233


def unpack_uint4(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    shape = uint8_data.shape
    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=-1).view(up_size(shape))


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size(shape))


qtensor_lib = Library("qtensors", "DEF")
qtensor_lib.define(
    "quantize_per_tensor_uint4(Tensor input, float scale, int zero_point) -> Tensor"
)


@impl(qtensor_lib, "quantize_per_tensor_uint4", "CompositeExplicitAutograd")
def quantize_per_tensor_uint4(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return pack_uint4(
        torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 15).to(torch.uint8)
    )


qtensor_lib.define(
    "dequantize_per_tensor_uint4(Tensor input, float scale, int zero_point) -> Tensor"
)


@impl(qtensor_lib, "dequantize_per_tensor_uint4", "CompositeExplicitAutograd")
def dequantize_per_tensor_uint4(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    input = unpack_uint4(input)
    return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale


class UInt4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(
            cls, up_size(elem.shape), dtype=torch.uint4, **kwargs
        )

    def __init__(self, elem, **kwargs):
        self.elem = elem

    @classmethod
    def from_unpacked(cls, unpacked):
        return UInt4Tensor(pack_uint4(unpacked))

    def tolist(self):
        return self.to(torch.uint8).tolist()

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        assert meta is None
        elem = flattened["elem"]
        return UInt4Tensor(elem)

    def __hash__(self):
        return hash(self.elem)

    def __eq__(self, other):
        return torch.equal(self.elem, other.elem)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.view.default:
            self, size = args
            size = utils.infer_size(size, self.numel())
            assert not kwargs
            # WARNING: views not preserved
            return UInt4Tensor(self.elem.reshape(down_size(size)))
        elif func is torch.ops.aten.view.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return unpack_uint4(self.elem).view(torch.uint8)
            return NotImplementedError(f"view {args}")
        elif func is torch.ops.aten.to.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return unpack_uint4(self.elem).view(torch.uint8)
            return NotImplementedError(f"to {args}")
        elif func is torch.ops.aten.eq.Tensor:
            args = pytree.tree_map_only(
                UInt4Tensor, lambda x: x.elem.view(torch.uint8), args
            )
            kwargs = pytree.tree_map_only(
                UInt4Tensor, lambda x: x.elem.view(torch.uint8), kwargs
            )
            return torch.ops.aten.eq.Tensor(*args, **kwargs)
        elif func is torch.ops.aten._to_copy.default:
            (self,) = args
            if kwargs == {"dtype": torch.uint8}:
                return unpack_uint4(self.elem).view(self.shape)  # no wrap
            else:
                raise NotImplementedError(f"_to_copy {kwargs}")
        elif func is torch.ops.aten.unbind.int:
            # This is tricky.  Given torch.tensor([0, 1, 2, 3]) we want to
            # create four tensors containing one element each.  But we can't
            # do this with uint4 because such a tensor's size is not divisible
            # by bytes.  What I am going to do instead is promote to uint8
            # when this happens
            self, dim = fill_defaults(args, 2, [0])
            if dim != self.dim() - 1:
                raise NotImplementedError(f"unbind dim={dim}")
            else:
                # We're unbinding the last dimension, need to promote
                return torch.ops.aten._to_copy.default(self, dtype=torch.uint8).unbind(
                    dim
                )
        elif func is torch.ops.aten.select.int:
            self, dim, index = args
            if dim != self.dim() - 1:
                return UInt4Tensor(torch.ops.aten.select.int(self.elem, dim, index))
            else:
                raise NotImplementedError(f"select dim={dim}")
        elif func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == self.dim() - 1:
                # hard case
                if step != 1:
                    raise NotImplementedError(f"slice step={step}")
                assert start % 2 == 0, start
                assert end >= self.shape[dim] or end % 2 == 0, end
                return UInt4Tensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start // 2, end // 2, 1)
                )
            else:
                # easy case
                return UInt4Tensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step)
                )
        elif func is torch.ops.aten.t.default:
            # assert False, "transpose is not properly implemented currently"
            (self,) = args
            unpacked = unpack_uint4(self.elem)
            transposed = torch.ops.aten.t.default(unpacked)
            transposed_and_packed = pack_uint4(transposed)
            return UInt4Tensor(transposed_and_packed)
        elif func is torch.ops.aten.transpose_copy.int:
            self, dim0, dim1 = args
            unpacked = unpack_uint4(self.elem).view(self.shape)
            transposed = torch.ops.aten.transpose_copy.int(unpacked, dim0, dim1)
            transposed_and_packed = pack_uint4(transposed)
            return UInt4Tensor(transposed_and_packed)
        elif func is torch.ops.aten.as_strided.default:
            # size, stride, storage_offset are referring to tensor elements, not physical bytes
            self, size, stride, storage_offset = args
            size = down_size(size)

            new_stride = []
            for s in stride:
                if s != 1:
                    # since two int4 equals to 1 uint8
                    new_stride.append(s // 2)
                else:
                    new_stride.append(s)
            stride = new_stride

            storage_offset //= 2
            return UInt4Tensor(
                torch.ops.aten.as_strided.default(
                    self.elem, size, stride, storage_offset
                )
            )

        raise NotImplementedError(f"{func}")

    __torch_function__ = torch._C._disabled_torch_function_impl


def _dynamically_quantize_per_channel_int4(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scale and zero point based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scale is the same dtype as the original tensor
    scale = torch.clamp(scale, min=eps).to(x.dtype)
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scale/zp
    # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x.transpose(0, 1) / scale
    x_round = torch.round(x_div)
    x_zp = x_round + zero_point
    x_zp = x_zp.transpose(0, 1)
    quant = torch.clamp(x_zp, quant_min, quant_max)

    if target_dtype == torch.uint4:
        # TODO: simplify (maybe implement to)
        quant = PerChannelSymmetricWeightUInt4Tensor.from_unpacked(
            quant.to(torch.uint8), scale
        )
    else:
        quant = quant.to(target_dtype)

    return quant, scale, zero_point


class PerChannelSymmetricWeightUInt4Tensor(UInt4Tensor):
    @staticmethod
    def __new__(cls, elem, scales, **kwargs):
        return super().__new__(cls, elem, **kwargs)

    def __init__(self, elem, scales, **kwargs):
        super().__init__(elem, **kwargs)

        self.scales = scales

    def __tensor_flatten__(self):
        return ["elem", "scales"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        assert meta is None
        elem = flattened["elem"]
        scales = flattened["scales"]
        return PerChannelSymmetricWeightUInt4Tensor(elem, scales)

    @classmethod

    #  inconsistently.

    def from_unpacked(cls, unpacked, scales):
        return cls(pack_uint4(unpacked), scales)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.addmm.default:
            bias, x, weight = args
            x_view = x.view(-1, x.shape[-1])
            y = torch.mm(x_view, weight.to(torch.uint8).to(x.dtype)) * weight.scales
            y = y.reshape(*x.shape[:-1], -1)
            if bias is not None:
                y += bias
            return y
        elif func is torch.ops.aten.t.default:
            # TODO: add proper support for transpose
            (self,) = args
            unpacked = unpack_uint4(self.elem)
            transposed = torch.ops.aten.t.default(unpacked)
            return PerChannelSymmetricWeightUInt4Tensor.from_unpacked(
                transposed, self.scales
            )
        elif func is torch.ops.aten.detach.default:
            (self,) = args
            return self
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def from_float(cls, w_fp32):
        w_int4, scales, _zp = _dynamically_quantize_per_channel_int4(
            w_fp32, 0, 15, torch.uint4
        )
        w_int4 = w_int4.to(device=w_fp32.device)
        return w_int4
