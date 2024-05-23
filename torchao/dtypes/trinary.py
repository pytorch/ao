import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch.library import impl, Library
import lovely_tensors as lt
lt.monkey_patch()

def down_size(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by four"
    return (*size[:-1], size[-1] // 4)

def up_size(size):
    return (*size[:-1], size[-1] * 4)

def roundclip(x, a, b):
    return torch.max(torch.tensor(a), torch.min(torch.tensor(b), torch.round(x)))

def quantize_per_tensor_trinary(weights):
    # Compute the average absolute value of the weight tensor
    gamma = torch.mean(torch.abs(weights))
    
    # Scale the weight tensor by the average absolute value
    scaled_weights = weights / (gamma + 1e-8)
    
    # Round each scaled weight to the nearest integer in {-1, 0, +1} and shift to {0, 1, 2}
    quantized_weights = roundclip(scaled_weights, -1, 1) + 1

    return quantized_weights.to(torch.uint8)

def unpack_trinary(uint8_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 4 entries per byte
    shape = uint8_data.shape
    unpacked_data = torch.empty((*shape, 4), dtype=torch.int8)

    #shift back to {-1, 0, 1} while unpacking
    unpacked_data[..., 0] = ((uint8_data >> 6) & 0b11).to(torch.int8) - 1.0
    unpacked_data[..., 1] = ((uint8_data >> 4) & 0b11).to(torch.int8) - 1.0
    unpacked_data[..., 2] = ((uint8_data >> 2) & 0b11).to(torch.int8) - 1.0
    unpacked_data[..., 3] = (uint8_data & 0b11).to(torch.int8) - 1.0
    return unpacked_data.view(up_size(shape))

def pack_trinary(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (uint8_data[::4] << 6 | uint8_data[1::4] << 4 | uint8_data[2::4] << 2 | uint8_data[3::4]).view(down_size(shape))
    return packed_data

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

class TrinaryTensor(torch.Tensor):
    def __new__(cls, data, *args, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        
        return torch.Tensor._make_wrapper_subclass(
            cls, up_size(elem.shape), dtype=torch.trinary, **kwargs
        )
        
    def __init__(self, elem, **kwargs):
        self.elem = elem

    @classmethod
    def from_unpacked(cls, unpacked):
        return TrinaryTensor(pack_trinary(unpacked))

    def tolist(self):
        return self.to(torch.uint8).tolist()

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        assert meta is None
        elem = flattened["elem"]
        return TrinaryTensor(elem)

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
            return TrinaryTensor(self.elem.reshape(down_size(size)))
        elif func is torch.ops.aten.view.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return unpack_trinary(self.elem).view(torch.uint8)
            return NotImplementedError(f"view {args}")
        elif func is torch.ops.aten.to.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return unpack_trinary(self.elem).view(torch.uint8)
            return NotImplementedError(f"to {args}")
        elif func is torch.ops.aten.eq.Tensor:
            args = pytree.tree_map_only(
                TrinaryTensor, lambda x: x.elem.view(torch.uint8), args
            )
            kwargs = pytree.tree_map_only(
                TrinaryTensor, lambda x: x.elem.view(torch.uint8), kwargs
            )
            return torch.ops.aten.eq.Tensor(*args, **kwargs)
        elif func is torch.ops.aten._to_copy.default:
            (self,) = args
            if kwargs == {"dtype": torch.uint8}:
                return unpack_trinary(self.elem).view(self.shape)  # no wrap
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
                return TrinaryTensor(torch.ops.aten.select.int(self.elem, dim, index))
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
                return TrinaryTensor(
                    # Not sure about this one
                    torch.ops.aten.slice.Tensor(self.elem, dim, start // 4, end // 4, 1) 
                )
            else:
                # easy case
                return TrinaryTensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step)
                )
        elif func is torch.ops.aten.t.default:
            # assert False, "transpose is not properly implemented currently"
            (self,) = args
            unpacked = unpack_trinary(self.elem)
            transposed = torch.ops.aten.t.default(unpacked)
            transposed_and_packed = pack_trinary(transposed)
            return TrinaryTensor(transposed_and_packed)
        elif func is torch.ops.aten.transpose_copy.int:
            self, dim0, dim1 = args
            unpacked = unpack_trinary(self.elem).view(self.shape)
            transposed = torch.ops.aten.transpose_copy.int(unpacked, dim0, dim1)
            transposed_and_packed = pack_trinary(transposed)
            return TrinaryTensor(transposed_and_packed)
        
        elif func is torch.ops.aten.as_strided.default:
            # size, stride, storage_offset are referring to tensor elements, not physical bytes
            self, size, stride, storage_offset = args
            size = down_size(size)

            new_stride = []
            for s in stride:
                if s != 1:
                    # since four trinary values equals to 1 uint8
                    new_stride.append(s // 4)
                else:
                    new_stride.append(s)
            stride = new_stride

            storage_offset //= 4
            return TrinaryTensor(
                torch.ops.aten.as_strided.default(
                    self.elem, size, stride, storage_offset
                )
            )

        raise NotImplementedError(f"{func}")

    __torch_function__ = torch._C._disabled_torch_function_impl