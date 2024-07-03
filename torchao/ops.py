import torch
from torch import Tensor
from torchao.utils import TORCH_VERSION_AFTER_2_4


def register_custom_op(name):
    def decorator(func):
        if TORCH_VERSION_AFTER_2_4:
            return torch.library.register_fake(f"{name}")(func)
        else:
            return torch.library.impl_abstract(f"{name}")(func)
    return decorator


def quant_llm_linear(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = 1,
) -> Tensor:
    """
    Quant-LLM linear layer A @ W.T. See https://arxiv.org/abs/2401.14112 for more details.

    Arguments
        EXPONENT: number of exponent bits
        MANTISSA: number of mantissa bits
        _in_feats: input activations in FP16
        _weights: packed FPx weights
        _scales: scale
        splitK: split K

    Returns
        output of linear layer
    """
    return torch.ops.torchao.quant_llm_linear.default(EXPONENT, MANTISSA, _in_feats, _weights, _scales, splitK)


@register_custom_op("torchao::quant_llm_linear")
def _(EXPONENT, MANTISSA, _in_feats, _weights, _scales, splitK = 1):
    torch._check(_in_feats.dim() == 2, lambda: f"input should be a 2d tensor, got {_in_feats.dim()}D")
    torch._check(_in_feats.dtype is torch.float16, lambda: f"weight must be FP16, got {_in_feats.dtype}")
    torch._check(_weights.dim() == 2, lambda: f"weight should be a 2d tensor, got {_weights.dim()}D")
    torch._check(_weights.dtype is torch.uint8, lambda: f"weight must be UINT8, got {_weights.dtype}")
    torch._check(_scales.dim() == 1, lambda: f"scale should be a 2d tensor, got {_scales.dim()}D")
    torch._check(_scales.dtype is torch.float16, lambda: f"scale must be FP16, got {_scales.dtype}")

    BS, IC = _in_feats.shape
    OC, _ = _weights.shape
    N_BITS = 1 + EXPONENT + MANTISSA
    torch._check(IC // 8 * N_BITS == _weights.shape[1], lambda: "Dimensions mismatched")
    torch._check(OC == _scales.shape[0], lambda: "Dimensions mismatched")

    return _in_feats.new_empty((BS, OC))


def int4mv(A: Tensor, B: Tensor, groupSize: int, scalesAndZeros: Tensor) -> Tensor:
    return torch.ops.torchao.int4mv.default(A, B, groupSize, scalesAndZeros)


@register_custom_op("torchao::int4mv")
def _(A, B, groupSize, scalesAndZeros):
    M = A.size(0)
    N = B.size(0)
    return torch.empty(M, N, dtype=A.dtype, device=A.device)