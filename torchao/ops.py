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



def unpack_tensor_core_tiled_layout(packed_w: Tensor, inner_k_tiles: int) -> Tensor:
    """
    Unpacks weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`.

    Assumes that the packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `inner_k_tiles = 2 | 4 | 8`"

    Args:
        packed_w: torch.tensor: 4D tensor with shape (N / 8) x (K / (inner_k_tiles * 16)) x 32 x inner_k_tiles, dtype is torch.int32
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.int32

    """
    return torch.ops.torchao.unpack_tensor_core_tiled_layout.default(
        packed_w=packed_w, inner_k_tiles=inner_k_tiles
    )
    

@register_custom_op(f"torchao::unpack_tensor_core_tiled_layout")
def _(packed_w: Tensor, inner_k_tiles: int) -> Tensor:
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 42d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        lambda: "inner_k_tiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == inner_k_tiles / 2,
        lambda: "packed weight must have inner_k_tiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * inner_k_tiles * 16

    return torch.empty((N, K), dtype=torch.int32, device=packed_w.device)

def dequantize_tensor_core_tiled_layout(packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int) -> Tensor:
    """
    Dequantizes by:
    - Unpacking weights that were packed with `torch.ops.aten._convert_weight_to_int4pack` to original tensor of shape `N x K`
    - Upcasting to bfloat16
    - Dequantizing with the scales_and_zeros that were packed with `torchao.quantization.utils.pack_tinygemm_scales_and_zeros`

    Assumes:
    - packed weights were generated with `torch.ops.aten._convert_weight_to_int4pack` with `inner_k_tiles = 2 | 4 | 8`"
    - packed scales_and_zeros were generated with `torchao.quantization.utils.pack_tinygemm_scales_and_zeros`
    - qGroupSize is 32 | 64 | 128 | 256
    
    Args:
        packed_w: torch.tensor: 4D tensor with shape `(N / 8) x (K / (inner_k_tiles * 16)) x 32 x inner_k_tiles / 2`, dtype is torch.int32
        scales_and_zeros: torch.tensor: 3D tensor with shape `numQGroups x N x 2`, dtype is torch.bfloat16 where numQGroups is K / qGroupSize
        qGroupSize: int
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.bfloat16

    """
    return torch.ops.torchao.dequantize_tensor_core_tiled_layout.default(
        packed_w, scales_and_zeros, group_size, inner_k_tiles
    )
    

@register_custom_op(f"torchao::dequantize_tensor_core_tiled_layout")
def _(packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int) -> Tensor:
    # packed_w preconditions
    torch._check(
        packed_w.dim() == 4,
        lambda: f"packed weight should be a 4d tensor, got {packed_w.dim()}D",
    )
    torch._check(
        packed_w.dtype is torch.int32,
        lambda: f"weight must be INT32, got {packed_w.dtype}",
    )
    torch._check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        lambda: "inner_k_tiles must be 2, 4, or 8",
    )
    torch._check(packed_w.size(2) == 32, lambda: "packed weight must have 32 at dim 2")
    torch._check(
        packed_w.size(3) == inner_k_tiles / 2,
        lambda: "packed weight must have inner_k_tiles/2 at dim 3",
    )
    N = packed_w.size(0) * 8
    K = packed_w.size(1) * inner_k_tiles * 16

    # scales_and_zeros preconditions
    torch._check(scales_and_zeros.dtype is torch.bfloat16, lambda: "scales_and_zeros must be bfloat16")
    torch._check(scales_and_zeros.dim() == 3, lambda: "scales_and_zeros must be 3D, got {scales_and_zeros.dim()}")
    torch._check(group_size == 32 or group_size == 64 or group_size == 128 or group_size == 256, lambda: "qGroupSize must be 32, 64, 128, or 256")
    torch._check(scales_and_zeros.size(0) == K // group_size, lambda: "scales_and_zeros must have K // qGroupSize at dim 0")
    torch._check(scales_and_zeros.size(1) == N, lambda: "scales_and_zeros must have N at dim 1")
    torch._check(scales_and_zeros.size(2) == 2, lambda: "scales_and_zeros must have 2 at dim 2")
    
    return torch.empty((N, K), dtype=torch.bfloat16, device=packed_w.device)


def marlin_24_mm(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    thread_k: int = -1, 
    thread_m: int = -1, 
    sms: int = -1, 
    max_par: int = 16,
) -> Tensor:
    """
    Sparse Marlin 2:4 matrix multiplication. Reference: https://github.com/IST-DASLab/Sparse-Marlin/tree/main

    Args:
        x: input matrix of shape `(n, k/2)` in column-major layout.
        weight_marlin: weight matrix of original shape `(m, k)` in Marlin format; see `Layer.pack()`.
        meta: metadata information for 2:4 sparsity.
        s: scales of shape `(n / groupsize / 2, m)`.
        workspace: tensor with at least `m / 128 * max_par` entries that are all zero.
        thread_k:  size of a thread_tile in `A` (can usually be left as auto -1).
        thread_m: size of a thread_tile in `A` (can usually be left as auto -1).
        sms: number of SMs to use for the kernel (can usually be left as auto -1).
        max_par: maximum number of batch 64 problems to solve in parallel for large input sizes.

    Returns:
        output matrix of shape `(n, m)` in column-major layout.
    """
    out = torch.empty((x.size(0), s.size(1)), dtype=x.dtype, device=x.device)

    # From: https://github.com/IST-DASLab/Sparse-Marlin/blob/c2ffa2395a3ada26c8cb7f910a5ec65bd3ce288a/marlin/marlin_cuda.cpp#L66
    prob_n = x.size(0)
    prob_m = out.size(1)
    prob_k = x.size(1)
    group_size = -1 if s.size(0) == 1 else prob_k / 2 / s.size(0)
    device = torch.cuda.current_device()
    cuda_stream = torch.cuda.current_stream(device)

    torch.ops.torchao.marlin_24_mm.default(
        x, weight_marlin, meta, out, 
        s, prob_m, prob_n, prob_k, 
        workspace, group_size, device, cuda_stream, 
        thread_k, thread_m, sms, max_par
    )

    return out


@register_custom_op(f"torchao::dequantize_tensor_core_tiled_layout")
def _(
    weight_marlin: Tensor,
    x: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    thread_k: int = -1, 
    thread_m: int = -1, 
    sms: int = -1, 
    max_par: int = 16,
) -> Tensor:
    
    # Dimensions check
    torch._check(weight_marlin.dim() == 2, lambda: f"weight_marlin should be a 2d tensor, got {weight_marlin.dim()}D")
    torch._check(x.dim() == 2, lambda: f"x should be a 2d tensor, got {x.dim()}D")
    torch._check(s.dim() == 2, lambda: f"s should be a 2d tensor, got {s.dim()}D")

    # Check if the dimensions are compatible
    torch._check(
        x.size(1) * 2 == weight_marlin.size(1),
        lambda: f"weight_marlin should have 2x the number of columns as x, got {weight_marlin.size(1)} and {x.size(1)}",
    )

    # TODO(diogo): Add more checks here

    return torch.empty((x.size(0), s.size(1)), dtype=x.dtype, device=x.device)
