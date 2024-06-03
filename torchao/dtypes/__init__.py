import torch
from .nf4tensor import NF4Tensor, to_nf4
from .uint4 import UInt4Tensor


def to_aqt(tensor):
    from torchao.dtypes.aqt import AffineQuantizedTensor
    from torchao.quantization.quant_primitives import MappingType

    assert tensor.dim() > 1

    block_size = (tensor.size(-2), 1)
    # TODO: Add block_size broadcast support to choose_qparams_affine
    block_size = ((1,) * (tensor.dim() - 2)) + block_size
    return AffineQuantizedTensor.from_float(tensor,
                                            MappingType.ASYMMETRIC,
                                            block_size,
                                            torch.int8)

def to_aqt_with_scales(tensor, scales):
    pass

def to_aqt_with_scales_zeropoint(tensor, scales, zeropoint):
    pass


__all__ = [
    "NF4Tensor",
    "to_nf4",
    "UInt4Tensor",
    "to_aqt",
]
