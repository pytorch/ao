import torch
from torchao.prototype.dtypes.uint2 import UInt2Tensor, unpack_uint2, pack_uint2

BITNET_OPS_TABLE = {}

def implements(aten_ops):
    def decorator(fn):
        for op in aten_ops:
            BITNET_OPS_TABLE[op] = fn
        return fn
    return decorator

def _quantize_int2(x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    # Quantize the input tensor to int2
    quant = x.sign() + 1

    if target_dtype == torch.uint2:
        quant = BitnetTensor.from_unpacked(
            quant.to(torch.uint8),
        )
    else:
        quant = quant.to(target_dtype)

    return quant

class BitnetTensor(UInt2Tensor):
    def __new__(cls, input_tensor: torch.Tensor, **kwargs):
        return super(BitnetTensor, cls).__new__(cls, input_tensor, **kwargs)
    
    def __init__(self, input_tensor: torch.Tensor, **kwargs):
        super(BitnetTensor, self).__init__(input_tensor, **kwargs)

    @staticmethod
    def __tensor_unflatten__(flattened, meta):
        assert meta is None
        elem = flattened["elem"]
        return BitnetTensor(elem)

    @classmethod
    def from_unpacked(cls, unpacked: torch.Tensor) -> "BitnetTensor":
        return cls(pack_uint2(unpacked))
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        def allowed_subclasses(type):
            return (
                issubclass(cls, type) or
                issubclass(torch._subclasses.fake_tensor.FakeTensor, type) or 
                issubclass(torch._subclasses.functional_tensor.FunctionalTensor, type)
            )
        
        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Bitnet, Up to the next one to handle")

        if func in BITNET_OPS_TABLE:
            return BITNET_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"Bitnet dispatch: attempting to run {func}, this is not supported")
    
    @classmethod
    def from_float(cls, w: torch.Tensor):
        w_int2 = _quantize_int2(w, torch.uint2).to(device=w.device)
        return w_int2

@implements([torch.ops.aten.mm.default])
def mm(func, args, kwargs):
    x, weight = args
    y = torch.mm(x, weight.to(torch.int8).to(x.device).to(x.dtype))
    return y

@implements([torch.ops.aten.addmm.default])
def addmm(func, args, kwargs):
    bias, x, weight = args
    y = torch.addmm(bias, x, weight.to(torch.int8).to(x.device).to(x.dtype))
    if bias is not None:
        y += bias
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
    elif dtype in (torch.float, torch.float16, torch.bfloat16, torch.int16, torch.int32, torch.int64):
        return unpack_uint2(tensor.elem).to(torch.int8).to(dtype)
    elif dtype == torch.uint8:
        return unpack_uint2(tensor.elem).view(torch.uint8)
    elif dtype == torch.uint2:
        return tensor.elem
    raise NotImplementedError(f"to {dtype} not supported")

@implements([torch.ops.aten._to_copy.default])
def _to_copy(func, args, kwargs):
    (tensor,) = args
    dtype = kwargs["dtype"]
    if dtype == torch.int8:
        return BitnetTensor(unpack_uint2(tensor).view(tensor.shape).view(torch.int8) - 1)
    elif dtype in (torch.float, torch.float16, torch.bfloat16, torch.int16, torch.int32, torch.int64):
        return BitnetTensor(tensor.to(torch.int8).to(dtype))
    elif dtype == torch.uint2:
        return BitnetTensor(tensor)
    raise NotImplementedError(f"to {dtype} not supported")

if __name__ == "__main__":
    # Test case using BitnetTensor
    a = torch.randint(0, 15, (2, 8), dtype=torch.uint8)
    a_bitnet = BitnetTensor(a)
    a_bitnet = a_bitnet.to(torch.uint2)
    print(f"a_bitnet: {a_bitnet}")

