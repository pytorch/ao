import torch
from torchao.dtypes.uint4 import pack_uint4, unpack_uint4
from torchao.dtypes import UInt4Tensor
from typing import Dict, Any
from torchao.dtypes.utils import _implements
from torchao.dtypes.utils import _ATEN_OP_OR_TORCH_FN_TABLE

SYMMETRIC_WEIGHT_OPS_TABLE: Dict[Any, Any] = {}

from torchao.dtypes.utils import _implements

def implements(aten_ops_or_torch_fns):
    return _implements(PerChannelSymmetricWeightUInt4Tensor, aten_ops_or_torch_fns)

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
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in _ATEN_OP_OR_TORCH_FN_TABLE[cls]:
            return _ATEN_OP_OR_TORCH_FN_TABLE[cls][func](func, *args, **kwargs)

        raise NotImplementedError(
            f"PerChannelSymmetricWeightUInt4Tensor dispatch: attempting to run {func}, this is not supported"
        )

        
    @classmethod
    def from_float(cls, w_fp32):
        w_int4, scales, _zp = _dynamically_quantize_per_channel_int4(
            w_fp32, 0, 15, torch.uint4
        )
        w_int4 = w_int4.to(device=w_fp32.device)
        return w_int4

@implements([torch.ops.aten.addmm.default])
def _(func, args, kwargs):
    bias, x, weight = args
    x_view = x.view(-1, x.shape[-1])
    y = torch.mm(x_view, weight.to(torch.uint8).to(x.dtype)) * weight.scales
    y = y.reshape(*x.shape[:-1], -1)
    if bias is not None:
        y += bias
    return y

@implements([torch.ops.aten.t.default])
def _(func, args, kwargs):
    # TODO: add proper support for transpose
    (tensor,) = args
    unpacked = unpack_uint4(tensor.elem)
    transposed = torch.ops.aten.t.default(unpacked)
    return PerChannelSymmetricWeightUInt4Tensor.from_unpacked(
        transposed, tensor.scales
    )

@implements([torch.ops.aten.detach.default])
def _(func, args, kwargs):
    (tensor,) = args
    return 
    
if __name__ == "__main__":
    # test
    x = torch.randn(2, 3, 4)
    w = torch.randn(5, 4)
    b = torch.randn(5)
    y = PerChannelSymmetricWeightUInt4Tensor.from_float(w)
    # print(y)
