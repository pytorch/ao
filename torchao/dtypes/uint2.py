import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch.library import impl, Library
from .uint4 import qtensor_lib


def down_size(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by four"
    return (*size[:-1], size[-1] // 4)

def up_size(size):
    return (*size[:-1], size[-1] * 4)

if torch.cuda.is_available() and torch.utils._triton.has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def triton_unpack_uint8_to_trinary2(uint8_data, output, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        uint8_value = tl.load(uint8_data + offsets, mask=mask)
        
        first_elements = ((uint8_value >> 6) & 0b11).to(tl.int8) - 1
        second_elements = ((uint8_value >> 4) & 0b11).to(tl.int8) - 1
        third_elements = ((uint8_value >> 2) & 0b11).to(tl.int8) - 1
        fourth_elements = (uint8_value & 0b11).to(tl.int8) - 1

        tl.store(output + offsets * 4 + 0, first_elements, mask=mask)
        tl.store(output + offsets * 4 + 1, second_elements, mask=mask)
        tl.store(output + offsets * 4 + 2, third_elements, mask=mask)
        tl.store(output + offsets * 4 + 3, fourth_elements, mask=mask)

    def unpack_uint8_to_trinary2(uint8_data: torch.Tensor) -> torch.Tensor:
        uint8_data = uint8_data.to('cuda')
        shape = uint8_data.shape
        output = torch.empty(up_size(shape), dtype=torch.int8, device='cuda')
        n_elements = uint8_data.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        triton_unpack_uint8_to_trinary2[grid](uint8_data, output, n_elements, BLOCK_SIZE=1024)
        return output

    @triton.jit
    def triton_unpack_uint2(uint8_data, output, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        uint8_value = tl.load(uint8_data + offsets, mask=mask)

        first_elements = (uint8_value >> 6) & 0b11
        second_elements = (uint8_value >> 4) & 0b11
        third_elements = (uint8_value >> 2) & 0b11
        fourth_elements = uint8_value & 0b11

        tl.store(output + offsets * 4 + 0, first_elements, mask=mask)
        tl.store(output + offsets * 4 + 1, second_elements, mask=mask)
        tl.store(output + offsets * 4 + 2, third_elements, mask=mask)
        tl.store(output + offsets * 4 + 3, fourth_elements, mask=mask)

    def unpack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
        uint8_data = uint8_data.to('cuda')
        shape = uint8_data.shape
        output = torch.empty(up_size(shape), dtype=torch.uint8, device='cuda')
        n_elements = uint8_data.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        triton_unpack_uint2[grid](uint8_data, output, n_elements, BLOCK_SIZE=1024)
        return output

    @triton.jit
    def triton_pack_uint2(uint8_data, output, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets * 4 < n_elements

        first_elements = tl.load(uint8_data + offsets * 4 + 0, mask=mask)
        second_elements = tl.load(uint8_data + offsets * 4 + 1, mask=mask)
        third_elements = tl.load(uint8_data + offsets * 4 + 2, mask=mask)
        fourth_elements = tl.load(uint8_data + offsets * 4 + 3, mask=mask)

        packed_data = (first_elements << 6) | (second_elements << 4) | (third_elements << 2) | fourth_elements

        tl.store(output + offsets, packed_data, mask=mask)

    def pack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
        uint8_data = uint8_data.to('cuda')
        shape = uint8_data.shape
        assert shape[-1] % 4 == 0
        n_elements = uint8_data.numel()
        packed_shape = down_size(shape)
        output = torch.empty(packed_shape, dtype=torch.uint8, device='cuda')
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 4),)
        triton_pack_uint2[grid](uint8_data, output, n_elements, BLOCK_SIZE=1024)
        return output

else:
    #@torch.compile
    def unpack_uint8_to_trinary2(uint8_data: torch.Tensor) -> torch.Tensor:
        # since we are using uint8 we will decode 4 entries per byte
        shape = uint8_data.shape
        first_elements = ((uint8_data >> 6) & 0b11).to(torch.int8) - 1
        second_elements = ((uint8_data >> 4) & 0b11).to(torch.int8) - 1
        third_elements = ((uint8_data >> 2) & 0b11).to(torch.int8) - 1
        fourth_elements = (uint8_data & 0b11).to(torch.int8) - 1
        return torch.stack([first_elements, second_elements, third_elements, fourth_elements], dim=-1).view(up_size(shape))

    #@torch.compile
    def unpack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
        # since we are using uint8 we will decode 4 entries per byte
        shape = uint8_data.shape
        uint8_data = uint8_data.to(torch.uint8)
        first_elements = ((uint8_data >> 6) & 0b11)
        second_elements = ((uint8_data >> 4) & 0b11)
        third_elements = ((uint8_data >> 2) & 0b11)
        fourth_elements = (uint8_data & 0b11)
        return torch.stack((first_elements, second_elements, third_elements, fourth_elements), dim=-1).view(up_size(shape))

    #packing uint8
    #@torch.compile
    def pack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
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


#qtensor_lib = Library("qtensors", "DEF")
qtensor_lib.define(
    "quantize_per_tensor_uint2(Tensor input, float scale, int zero_point) -> Tensor"
)


@impl(qtensor_lib, "quantize_per_tensor_uint2", "CompositeExplicitAutograd")
def quantize_per_tensor_uint2(
    input: torch.Tensor,
    scale: float = 1.0,
    zero_point: int = 1,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return pack_uint2(
        torch.clamp(torch.round(input * inv_scale) + zero_point, 0, 2).to(torch.uint8)
    )


qtensor_lib.define(
    "dequantize_per_tensor_uint2(Tensor input, float scale, int zero_point) -> Tensor"
)


@impl(qtensor_lib, "dequantize_per_tensor_uint2", "CompositeExplicitAutograd")
def dequantize_per_tensor_uint2(
    input: torch.Tensor,
    scale: float = 1.0,
    zero_point: int = 1,
) -> torch.Tensor:
    input = unpack_uint2(input)
    return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale


class UInt2Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False

        return torch.Tensor._make_wrapper_subclass(
            cls, up_size(elem.shape), dtype=torch.uint2, **kwargs
        )

    def __init__(self, elem, **kwargs):
        self.elem = elem

    @classmethod
    def from_unpacked(cls, unpacked):
        return UInt2Tensor(pack_uint2(unpacked))

    def tolist(self):
        return self.to(torch.uint8).tolist()

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        assert meta is None
        elem = flattened["elem"]
        return UInt2Tensor(elem)

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
            return UInt2Tensor(self.elem.reshape(down_size(size)))
        elif func is torch.ops.aten.view.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return unpack_uint2(self.elem).view(torch.uint8)
            return NotImplementedError(f"view {args}")
        elif func is torch.ops.aten.to.dtype:
            self, dtype = args
            if dtype == torch.uint8:
                return unpack_uint2(self.elem).view(torch.uint8)
            return NotImplementedError(f"to {args}")
        elif func is torch.ops.aten.eq.Tensor:
            args = pytree.tree_map_only(
                UInt2Tensor, lambda x: x.elem.view(torch.uint8), args
            )
            kwargs = pytree.tree_map_only(
                UInt2Tensor, lambda x: x.elem.view(torch.uint8), kwargs
            )
            return torch.ops.aten.eq.Tensor(*args, **kwargs)
        elif func is torch.ops.aten._to_copy.default:
            (self,) = args
            dtype = kwargs["dtype"]
            if dtype == torch.uint8:
                return unpack_uint2(self.elem).view(self.shape)
            if dtype in (torch.uint16, torch.uint32, torch.uint64):
                return self.to(torch.uint8).to(dtype)
            if dtype == torch.uint2:
                return self
            else:
                raise NotImplementedError(f"_to_copy {kwargs}")
        elif func is torch.ops.aten.unbind.int:
            # This is tricky.  Given torch.tensor([0, 1, 2, 3]) we want to
            # create four tensors containing one element each.  But we can't
            # do this with uint2 because such a tensor's size is not divisible
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
                return UInt2Tensor(torch.ops.aten.select.int(self.elem, dim, index))
            else:
                raise NotImplementedError(f"select dim={dim}")
        elif func is torch.ops.aten.slice.Tensor:
            self, dim, start, end, step = fill_defaults(args, 5, [0, None, None, 1])
            if dim == self.dim() - 1:
                # hard case
                if step != 1:
                    raise NotImplementedError(f"slice step={step}")
                assert start % 4 == 0, start
                assert end >= self.shape[dim] or end % 4 == 0, end
                return UInt2Tensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start // 4, end // 4, 1)
                )
            else:
                # easy case
                return UInt2Tensor(
                    torch.ops.aten.slice.Tensor(self.elem, dim, start, end, step)
                )
        elif func is torch.ops.aten.t.default:
            # assert False, "transpose is not properly implemented currently"
            (self,) = args
            unpacked = unpack_uint2(self.elem)
            transposed = torch.ops.aten.t.default(unpacked)
            transposed_and_packed = pack_uint2(transposed)
            return UInt2Tensor(transposed_and_packed)
        elif func is torch.ops.aten.transpose_copy.int:
            self, dim0, dim1 = args
            unpacked = unpack_uint2(self.elem).view(self.shape)
            transposed = torch.ops.aten.transpose_copy.int(unpacked, dim0, dim1)
            transposed_and_packed = pack_uint2(transposed)
            return UInt2Tensor(transposed_and_packed)
        elif func is torch.ops.aten.as_strided.default:
            # size, stride, storage_offset are referring to tensor elements, not physical bytes
            self, size, stride, storage_offset = args
            size = down_size(size)

            new_stride = []
            for s in stride:
                if s != 1:
                    # since two int4 equals to 1 uint8
                    new_stride.append(s // 4)
                else:
                    new_stride.append(s)
            stride = new_stride

            storage_offset //= 4
            return UInt2Tensor(
                torch.ops.aten.as_strided.default(
                    self.elem, size, stride, storage_offset
                )
            )

        raise NotImplementedError(f"{func}")

    __torch_function__ = torch._C._disabled_torch_function_impl


def _quantize_int2(x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    quant = x.sign() + 1

    if target_dtype == torch.uint2:
        quant = BitnetTensor.from_unpacked(
            quant.to(torch.uint8),
        )
    else:
        quant = quant.to(target_dtype)

    return quant


class BitnetTensor(UInt2Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        return super().__new__(cls, elem, **kwargs)

    def __init__(self, elem, **kwargs):
        super().__init__(elem, **kwargs)

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(flattened, meta, outer_size, outer_stride):
        assert meta is None
        elem = flattened["elem"]
        return BitnetTensor(elem)

    @classmethod
    #  inconsistently.
    def from_unpacked(cls, unpacked: torch.Tensor) -> "BitnetTensor":
        return cls(pack_uint2(unpacked))

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.mm.default:
            x, weight = args
            y = torch.mm(x, weight.to(torch.int8).to(x.device).to(x.dtype))
            return y
        elif func is torch.ops.aten.addmm.default:
            bias, x, weight = args
            y = torch.mm(x, weight.to(torch.int8).to(x.device).to(x.dtype))
            if bias is not None:
                y += bias
            return y
        elif func is torch.ops.aten.t.default:
            # TODO: add proper support for transpose
            (self,) = args
            unpacked = unpack_uint2(self.elem).to(self.device)
            transposed = torch.ops.aten.t.default(unpacked)
            return BitnetTensor.from_unpacked(transposed)
        elif func is torch.ops.aten.detach.default:
            (self,) = args
            return self
        elif func is torch.ops.aten.to.dtype:
            self, dtype = args
            if dtype == torch.int8:
                return unpack_uint2(self.elem).view(torch.int8) - 1
        elif func is torch.ops.aten._to_copy.default:
            (self,) = args
            dtype = kwargs["dtype"]
            if dtype == torch.int8:
                return unpack_uint2(self.elem).view(self.shape).view(torch.int8) - 1
            elif dtype in (torch.float, torch.float16, torch.bfloat16, torch.int16, torch.int32, torch.int64):
                return self.to(torch.int8).to(dtype)
            elif dtype == torch.uint2:
                return self
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def from_float(cls, w: torch.Tensor):
        w_int2 = _quantize_int2(
            w, torch.uint2
        ).to(device=w.device)
        return w_int2

