import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch.library import impl, Library
from typing import Dict, Any, Tuple

UINT4_OPS_TABLE: Dict[Any, Any] = {}

def implements(aten_ops):
    def decorator(fn):
        for op in aten_ops:
            UINT4_OPS_TABLE[op] = fn
        return fn
    return decorator

def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)

def up_size(size):
    return (*size[:-1], size[-1] * 2)

def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.
    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list
    Example:
        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i - n + len(defaults_tail)])
    return r

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

        def allowed_subclasses(type):
            return (
                issubclass(cls, type) or
                issubclass(torch._subclasses.fake_tensor.FakeTensor, type) or 
                issubclass(torch._subclasses.functional_tensor.FunctionalTensor, type)
            )
        
        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Up to the next one to handle")

        if func in UINT4_OPS_TABLE:
            return UINT4_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"UINT4 dispatch: attempting to run {func}, this is not supported")

@implements([torch.ops.aten.view.default])
def view_uint4(func, args, kwargs):
    tensor, size = args
    size = utils.infer_size(size, tensor.numel())
    assert not kwargs
    # WARNING: views not preserved
    return UInt4Tensor(tensor.elem.reshape(down_size(size)))

@implements([torch.ops.aten.view.dtype])
def view_as_uint4(func, args, kwargs):
    tensor, dtype = args
    if dtype == torch.uint8:
        return unpack_uint4(tensor.elem)
    return NotImplemented(f"view {dtype} not supported")

@implements([torch.ops.aten.to.dtype])
def to_uint4(func, args, kwargs):
    tensor, dtype = args
    if dtype == torch.uint8:
        return unpack_uint4(tensor.elem)
    return NotImplemented(f"to {dtype} not supported")

@implements([torch.ops.aten.eq.Tensor])
def eq_uint4(func, args, kwargs):
    args = pytree.trees_map_only(UInt4Tensor, lambda x: x.elem.view(torch.uint8), kwargs)
    kwargs = pytree.tree_map_only(lambda x: x.elem.view(torch.uint8), kwargs)
    return torch.ops.aten.eq.Tensor(*args, **kwargs)

@implements([torch.ops.aten._to_copy])
def to_copy_uin4(func, args, kwargs):
    (tensor, ) = args
    if kwargs == {"dtype": torch.uint8}:
        return unpack_uint4(tensor.elem).view(tensor.shape)  # no wrap
    else:
        raise NotImplementedError(f"_to_copy {kwargs}")
    
@implements([torch.ops.aten.unbind.int])
def unbind_uint4(func, args, kwargs):
    # This is tricky.  Given torch.tensor([0, 1, 2, 3]) we want to
    # create four tensors containing one element each.  But we can't
    # do this with uint4 because such a tensor's size is not divisible
    # by bytes.  What I am going to do instead is promote to uint8
    # when this happens
    tensor, dim = fill_defaults(args, 2, [0])
    if dim != tensor.dim() - 1:
        raise NotImplementedError(f"unbind dim={dim}")
    else:
        # We're unbinding the last dimension, need to promote
        return torch.ops.aten._to_copy.default(tensor, dtype=torch.uint8).unbind(
            dim
        )

@implements([torch.ops.aten.select.int])
def select_uint4(func, args, kwargs):
    tensor, dim, index = args
    if dim != tensor.dim() - 1:
        return UInt4Tensor(torch.ops.aten.select.int(tensor.elem, dim, index))
    else:
        raise NotImplementedError(f"select dim={dim}")
    
@implements([torch.ops.aten.slice.Tensor])
def slice_uint4(func, args, kwargs):
    tensor, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
    if dim == tensor.dim() - 1:
        # hard case
        if step != 1:
            raise NotImplementedError(f"slice step={step}")
        assert start % 2 == 0, start
        assert end >= tensor.shape[dim] or end % 2 == 0, end
        return UInt4Tensor(
            torch.ops.aten.slice.Tensor(tensor.elem, dim, start // 2, end // 2, 1)
        )
    else:
        # easy case
        return UInt4Tensor(
            torch.ops.aten.slice.Tensor(tensor.elem, dim, start, end, step)
        )

@implements([torch.ops.aten.t.default])
def t_uint4(func, args, kwargs):
    # assert False, "transpose is not properly implemented currently"
    (tensor,) = args
    unpacked = unpack_uint4(tensor.elem)
    transposed = torch.ops.aten.t.default(unpacked)
    transposed_and_packed = pack_uint4(transposed)
    return UInt4Tensor(transposed_and_packed)

@implements([torch.ops.aten.transpose_copy.int])
def transpose_copy_uint4(func, args, kwargs):
    tensor, dim0, dim1 = args
    unpacked = unpack_uint4(tensor.elem).view(tensor.shape)
    transposed = torch.ops.aten.transpose_copy.int(unpacked, dim0, dim1)
    transposed_and_packed = pack_uint4(transposed)
    return UInt4Tensor(transposed_and_packed)

@implements([torch.ops.aten.as_strided.default])
def as_strided_uint4(func, args, kwargs):
    # size, stride, storage_offset are referring to tensor elements, not physical bytes
    tensor, size, stride, storage_offset = args
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
            tensor.elem, size, stride, storage_offset
        )
    )