import torch
from .uint4i import pack_uint4, unpack_uint4
from .uint4i import UInt4Tensor
from typing import Dict, Any

SYMMETRIC_WEIGHT_OPS_TABLE: Dict[Any, Any] = {}

def implements(aten_ops):
    def decorator(fn):
        for op in aten_ops:
            SYMMETRIC_WEIGHT_OPS_TABLE[op] = fn
        return fn
    return decorator

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
        return super(UInt4Tensor, cls).__new__(cls, elem, **kwargs)

    def __init__(self, elem, scales, **kwargs):
        super(UInt4Tensor, self).__init__(elem, **kwargs)

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
        def allowed_subclasses(type):
            return (
                issubclass(cls, type) or
                issubclass(torch._subclasses.fake_tensor.FakeTensor, type) or 
                issubclass(torch._subclasses.functional_tensor.FunctionalTensor, type)
            )
        
        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented("Up to the next one to handle")

        if func in SYMMETRIC_WEIGHT_OPS_TABLE:
            return SYMMETRIC_WEIGHT_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"UINT4 dispatch: attempting to run {func}, this is not supported")

        
    @classmethod
    def from_float(cls, w_fp32):
        w_int4, scales, _zp = _dynamically_quantize_per_channel_int4(
            w_fp32, 0, 15, torch.uint4
        )
        w_int4 = w_int4.to(device=w_fp32.device)
        return w_int4

@implements([torch.ops.aten.addmm.default])
def addmm(func, args, kwargs):
    bias, x, weight = args
    x_view = x.view(-1, x.shape[-1])
    y = torch.mm(x_view, weight.to(torch.uint8).to(x.dtype)) * weight.scales
    y = y.reshape(*x.shape[:-1], -1)
    if bias is not None:
        y += bias
    return y

@implements([torch.ops.aten.t.default])
def t(func, args, kwargs):
    # TODO: add proper support for transpose
    (tensor,) = args
    unpacked = unpack_uint4(tensor.elem)
    transposed = torch.ops.aten.t.default(unpacked)
    return PerChannelSymmetricWeightUInt4Tensor.from_unpacked(
        transposed, tensor.scales
    )

@implements([torch.ops.aten.detach.default])
def detach(func, args, kwargs):
    (tensor,) = args
    return tensor