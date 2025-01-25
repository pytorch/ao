import torch

from torchao.prototype.dtypes.uint2 import UInt2Tensor, pack_uint2, unpack_uint2

BITNET_OPS_TABLE = {}


def implements(aten_ops):
    def decorator(fn):
        for op in aten_ops:
            BITNET_OPS_TABLE[op] = fn
        return fn

    return decorator


def _quantize_int2(x: torch.Tensor) -> torch.Tensor:
    # Quantize the input tensor to int2
    quant = x.sign() + 1
    quant = BitnetTensor.from_unpacked(quant.to(torch.uint8))
    return quant


class BitnetTensor(UInt2Tensor):
    def __new__(cls, input_tensor: torch.Tensor, **kwargs):
        return super(BitnetTensor, cls).__new__(cls, input_tensor, **kwargs)

    def __init__(self, input_tensor: torch.Tensor, **kwargs):
        super(BitnetTensor, self).__init__(input_tensor, **kwargs)

    @staticmethod
    def __tensor_unflatten__(flattened, *meta):
        # TODO - meta is not None, is it ok?
        elem = flattened["elem"]
        return BitnetTensor(elem)

    @classmethod
    def from_unpacked(cls, unpacked: torch.Tensor) -> "BitnetTensor":
        return cls(pack_uint2(unpacked))

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
            return NotImplemented("Bitnet, Up to the next one to handle")

        if func in BITNET_OPS_TABLE:
            return BITNET_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(
            f"Bitnet dispatch: attempting to run {func}, this is not supported"
        )

    @classmethod
    def from_float(cls, w: torch.Tensor):
        w_intq = _quantize_int2(w)
        w_int2 = w_intq.to(device=w.device)
        return w_int2

    def clone(self):
        return BitnetTensor(self.elem.clone())

    def copy_(self, src):
        self.elem.copy_(src.elem)
        return self

    def tolist(self):
        data = unpack_uint2(self.elem).tolist()
        return data

    def __repr__(self):
        try:
            data = unpack_uint2(self.elem).tolist()
        except AssertionError:
            data = f"Tensor of shape {self.shape} and dtype {self.elem.dtype}"
        return f"BitnetTensor({data}, dtype={self.elem.dtype})"

    def to(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]
            if dtype == torch.int8:
                return unpack_uint2(self.elem).view(self.shape).view(torch.int8)
            elif dtype in (
                torch.float,
                torch.float16,
                torch.bfloat16,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                return unpack_uint2(self.elem).to(torch.int8).to(dtype)
            elif dtype == torch.uint8:
                return unpack_uint2(self.elem).view(torch.uint8)
            elif isinstance(self, BitnetTensor):
                return self
        if "device" in kwargs:
            device = kwargs["device"]
            return BitnetTensor(self.elem.to(device=device))

        return super().to(*args, **kwargs)


@implements([torch.ops.aten.mm.default])
def mm(func, args, kwargs):
    x, weight = args
    if isinstance(x, BitnetTensor):
        x = unpack_uint2(x.elem).to(torch.float32)
    if isinstance(weight, BitnetTensor):
        weight = unpack_uint2(weight.elem).to(torch.float32)
    y = torch.mm(x, weight)
    return y


@implements([torch.ops.aten.addmm.default])
def addmm(func, args, kwargs):
    bias, x, weight = args
    if isinstance(x, BitnetTensor):
        x = unpack_uint2(x.elem).to(torch.float32)
    if isinstance(weight, BitnetTensor):
        weight = unpack_uint2(weight.elem).to(torch.float32)
    if bias is not None:
        bias = bias.to(torch.float32)
    y = torch.addmm(bias, x, weight)
    return y


@implements([torch.ops.aten.t.default])
def t(func, args, kwargs):
    (tensor,) = args
    unpacked = unpack_uint2(tensor.elem).to(tensor.device)
    transposed = unpacked.t()
    return BitnetTensor(pack_uint2(transposed))


@implements([torch.ops.aten.detach.default])
def detach(func, args, kwargs):
    (tensor,) = args
    return tensor


@implements([torch.ops.aten.to.dtype])
def to_dtype(func, args, kwargs):
    (tensor, dtype) = args
    if dtype == torch.int8:
        return unpack_uint2(tensor.elem).view(torch.uint8) - 1
    elif dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return unpack_uint2(tensor.elem).to(torch.int8).to(dtype)
    elif dtype == torch.uint8:
        return unpack_uint2(tensor.elem).view(torch.uint8)
    elif isinstance(tensor, BitnetTensor):
        return tensor.elem
    raise NotImplementedError(f"to {dtype} not supported")


@implements([torch.ops.aten._to_copy.default])
def _to_copy(func, args, kwargs):
    (tensor,) = args
    dtype = kwargs["dtype"]
    if dtype == torch.int8:
        return BitnetTensor(
            unpack_uint2(tensor).view(tensor.shape).view(torch.int8) - 1
        )
    elif dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        return BitnetTensor(tensor.to(torch.int8).to(dtype))
    elif isinstance(tensor, BitnetTensor):
        return BitnetTensor(tensor)
    raise NotImplementedError(f"to {dtype} not supported")


@implements([torch.ops.aten.clone.default])
def clone(func, args, kwargs):
    (tensor,) = args
    return tensor.clone()


@implements([torch.ops.aten.allclose.default])
def allclose(func, args, kwargs):
    (a, b) = args
    return torch.allclose(a.elem, b.elem, **kwargs)
