import torch
import torch._prims_common as utils

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
def unpack_uint4(quantized_data) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using uint8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    first_elements = (quantized_data >> 4).to(torch.uint8)
    second_elements = (quantized_data & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=-1)

class UInt4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        # TODO: uint64 here is wrong, need a real dtype.  Don't try to(int64)
        # weird shit will happen
        assert elem.dtype is torch.uint8
        return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.int64)

    def __init__(self, elem):
        self.elem = elem

    def tolist(self):
        return self.to(torch.uint8).tolist()

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.view.default:
            self, size = args
            size = utils.infer_size(size, self.numel())
            assert not kwargs
            # WARNING: views not preserved
            return UInt4Tensor(self.elem.reshape(down_size(size)))
        elif func is torch.ops.aten._to_copy.default:
            self, = args
            if kwargs == {'dtype': torch.uint8}:
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
                return torch.ops.aten._to_copy.default(self, dtype=torch.uint8).unbind(dim)
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
                return UInt4Tensor(torch.ops.aten.slice.Tensor(self.elem, dim, start // 2, end // 2, 1))
            else:
                # easy case
                return UInt4Tensor(torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step))
        raise NotImplementedError(f"{func}")

    def __eq__(self, other):
        return torch.equal(self.elem, other.elem)

    __torch_function__ = torch._C._disabled_torch_function_impl
