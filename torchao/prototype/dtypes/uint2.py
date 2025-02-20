from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch._prims_common as utils

from torchao.utils import fill_defaults

UINT2_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    def decorator(fn):
        for op in aten_ops:
            UINT2_OPS_TABLE[op] = fn
        return fn

    return decorator


def down_size(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by 4"
    return (*size[:-1], size[-1] // 4)


def up_size(size):
    return (*size[:-1], size[-1] * 4)


def unpack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    uint8_data = uint8_data.to(torch.uint8)
    first_elements = (uint8_data >> 6) & 0b11
    second_elements = (uint8_data >> 4) & 0b11
    third_elements = (uint8_data >> 2) & 0b11
    fourth_elements = uint8_data & 0b11
    return torch.stack(
        (first_elements, second_elements, third_elements, fourth_elements), dim=-1
    ).view(up_size(shape))


def pack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0, f"{shape}, last dim not divisible by 4"
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (
        uint8_data[::4] << 6
        | uint8_data[1::4] << 4
        | uint8_data[2::4] << 2
        | uint8_data[3::4]
    ).view(down_size(shape))
    return packed_data


@dataclass
class SubclassTensorArgs:
    original_shape: torch.Size
    original_strides: Tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


class UInt2Tensor(torch.Tensor):
    def __new__(cls, input_tensor: torch.Tensor):
        assert input_tensor.dtype == torch.uint8
        tensor_meta = SubclassTensorArgs(
            input_tensor.size(),
            input_tensor.stride(),
            input_tensor.storage_offset(),
            cls,
            input_tensor.device,
            input_tensor.requires_grad,
        )
        uint2i_tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            up_size(tensor_meta.original_shape),
            tensor_meta.original_strides,
            tensor_meta.storage_offset,
            dtype=torch.uint8,  # Not sure if this is correct
            device=tensor_meta.device,
            requires_grad=tensor_meta.requires_grad,
        )
        return uint2i_tensor

    def __init__(self, input_tensor: torch.Tensor, **kwargs):
        self.elem = input_tensor

    @classmethod
    def from_packed(cls, unpacked):
        return UInt2Tensor(pack_uint2(unpacked))

    def tolist(self):
        return unpack_uint2(self.elem).tolist()

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta):
        assert meta is None
        elem = flattened["elem"]
        return UInt2Tensor(elem)

    def __hash__(self):
        return hash(self.elem)

    def __eq__(self, other):
        return torch.equal(self.elem, other.elem)

    def __repr__(self):
        data = unpack_uint2(self.elem).tolist()
        return f"UInt2Tensor({data}, dtype=torch.uint2)"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Up to the next one to handle")

        if func in UINT2_OPS_TABLE:
            return UINT2_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(
            f"UINT2 dispatch: attempting to run {func}, this is not supported"
        )


@implements([torch.ops.aten.view.default])
def uint2_view(func, args, kwargs):
    tensor, size = args
    size = utils.infer_size(size, tensor.numel())
    assert not kwargs
    dsize = down_size(size)
    reshaped_elem = tensor.elem.view(dsize)
    return UInt2Tensor(reshaped_elem)


@implements([torch.ops.aten.view.dtype])
def view_dtype(func, args, kwargs):
    tensor, dtype = args
    if dtype is torch.uint8:
        return unpack_uint2(tensor.elem).to(torch.uint8)
    raise NotImplementedError(f"view {dtype} not supported")


@implements([torch.ops.aten.clone.default])
def clone(func, args, kwargs):
    tensor = args[0]
    return UInt2Tensor(tensor.elem.clone())


@implements([torch.ops.aten._unsafe_view.default])
def unsafe_view(func, args, kwargs):
    tensor, size = args
    size = utils.infer_size(size, tensor.numel())
    assert not kwargs
    dsize = down_size(size)
    reshaped_elem = tensor.elem.view(dsize)
    return UInt2Tensor(reshaped_elem)


@implements([torch.ops.aten.unbind.int])
def unbind(func, args, kwargs):
    tensor, dim = fill_defaults(args, 2, [0])
    if dim != tensor.dim() - 1:
        raise NotImplementedError(f"unbind dim={dim}")
    else:
        x = tensor.elem.to(torch.uint8).unbind(dim)
        return x


@implements([torch.ops.aten._to_copy.default])
def to_copy(func, args, kwargs):
    (tensor,) = args
    dtype = kwargs["dtype"]
    if dtype == torch.uint8:
        return unpack_uint2(tensor.elem).view(tensor.shape).view(torch.uint8)
    elif dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return tensor.to(torch.uint8).to(dtype)
    elif isinstance(tensor, UInt2Tensor):
        return tensor
    raise NotImplementedError(f"to_copy {dtype} not supported")


@implements([torch.ops.aten.select.int])
def select(func, args, kwargs):
    tensor, dim, index = args
    if dim != tensor.dim() - 1:
        selected_elem = tensor.elem.select(dim, index)
        return UInt2Tensor(selected_elem)
    else:
        raise NotImplementedError(f"select dim={dim}")


@implements([torch.ops.aten.reshape.default])
def reshape(func, args, kwargs):
    tensor, size = args
    size = utils.infer_size(size, tensor.numel())
    assert not kwargs
    dsize = down_size(size)
    reshaped_elem = tensor.elem.view(dsize)
    return UInt2Tensor(reshaped_elem)


def slice_tensor(func, args, kwargs):
    tensor, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    if dim == tensor.dim() - 1:
        if step != 1:
            raise NotImplementedError(f"slice step={step}")
        assert start % 4 == 0, start
        assert end is None or end % 4 == 0, end
        end = end if end is not None else tensor.shape[dim]
        sliced_elem = tensor.elem[..., start // 4 : end // 4 : step]
        return UInt2Tensor(sliced_elem)
    else:
        sliced_elem = tensor.elem[..., start:end:step]
        return UInt2Tensor(sliced_elem)


@implements([torch.ops.aten.equal.default])
def equal(func, args, kwargs):
    tensor, other = args
    return torch.equal(tensor.elem, other.elem)


@implements([torch.ops.aten.detach.default])
def detach(func, args, kwargs):
    (tensor,) = args
    detached_elem = tensor.elem.detach()
    return UInt2Tensor(detached_elem)


@implements([torch.ops.aten.to.dtype])
def to_dtype(func, args, kwargs):
    (tensor, dtype) = args
    if dtype == torch.uint8:
        return unpack_uint2(tensor.elem).view(torch.uint8)
    elif dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return unpack_uint2(tensor.elem).to(torch.uint8).to(dtype)
    elif isinstance(tensor, UInt2Tensor):
        return tensor.elem

    raise NotImplementedError(f"to {dtype} not supported")


@implements([torch.ops.aten.t.default])
def t(func, args, kwargs):
    (tensor,) = args
    unpacked = unpack_uint2(tensor.elem).to(tensor.device)
    transposed = unpacked.t()
    return UInt2Tensor(pack_uint2(transposed))


@implements([torch.ops.aten.allclose.default])
def allclose(func, args, kwargs):
    tensor, other = args
    return torch.allclose(tensor.elem, other.elem)
