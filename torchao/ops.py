# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Optional

import torch
from torch import Tensor

from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

lib = torch.library.Library("torchao", "FRAGMENT")
lib.define(
    "quant_llm_linear(int EXPONENT, int MANTISSA, Tensor _in_feats, Tensor _weights, Tensor _scales, int splitK) -> Tensor"
)
lib.define(
    "unpack_tensor_core_tiled_layout(Tensor packed_w, int inner_k_tiles) -> Tensor"
)
lib.define(
    "dequantize_tensor_core_tiled_layout(Tensor packed_w, Tensor scales_and_zeros, int group_size, int inner_k_tiles) -> Tensor"
)
lib.define(
    "marlin_24_gemm(Tensor x, Tensor weight_marlin, Tensor meta, Tensor s, Tensor workspace, int bits, int size_m, int size_n, int size_k) -> Tensor"
)
lib.define(
    "marlin_qqq_gemm(Tensor x, Tensor weight_marlin, Tensor s_tok, Tensor s_ch, Tensor s_group, Tensor workspace, int size_m, int size_n, int size_k) -> Tensor"
)
lib.define(
    "rowwise_scaled_linear_cutlass_s8s4(Tensor input, Tensor input_scale, Tensor weight, Tensor weight_scale, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor"
)
lib.define(
    "rowwise_scaled_linear_cutlass_s4s4(Tensor input, Tensor input_scale, Tensor weight, Tensor weight_scale, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor"
)
lib.define(
    "rowwise_scaled_linear_sparse_cutlass_f8f8(Tensor input, Tensor input_scale, Tensor weight, Tensor weight_meta, Tensor weight_scale, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor"
)
lib.define(
    "to_sparse_semi_structured_cutlass_sm9x_f8(Tensor weight) -> (Tensor, Tensor)"
)
lib.define(
    "sparse24_sm90_sparsify(Tensor input, str metadata_fmt, str activation, str sp_selection_algo, *, ScalarType? dtype = None, Tensor? scale=None) -> (Tensor, Tensor)"
)
lib.define(
    "swizzle_mm(Tensor mat1, Tensor mat2, bool mat1_is_swizzled, bool mat2_is_swizzled) -> Tensor"
)
lib.define(
    "swizzle_scaled_mm(Tensor mat1, Tensor mat2, bool mat1_is_swizzled, bool mat2_is_swizzled, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None) -> Tensor"
)
# Note: we need to add the `torch._C.Tag.needs_fixed_stride_order` tag in order for inductor
# to honor the layout constraints for `b` in the two ops below.
lib.define(
    "mx_fp8_bf16(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale) -> Tensor",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)
lib.define(
    "mx_fp4_bf16(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale) -> Tensor",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)
lib.define(
    "qscaled_dot_product(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, float? scale=None, float q_scale=1.0, int q_zp=0, float k_scale=1.0, int k_zp=0, float v_scale=1.0, int v_zp=0, float a_scale=1.0, int a_zp=0, float o_scale=1.0, int o_zp=0) -> Tensor"
)


def register_custom_op(name):
    def decorator(func):
        if TORCH_VERSION_AT_LEAST_2_4:
            return torch.library.register_fake(f"{name}")(func)
        else:
            return torch.library.impl_abstract(f"{name}")(func)

    return decorator


def register_custom_op_impl(name):
    def decorator(func):
        if TORCH_VERSION_AT_LEAST_2_4:
            return torch.library.custom_op(f"{name}", mutates_args=())(func)
        else:
            return torch.library.impl(f"{name}", "CUDA")(func)

    return decorator


@functools.lru_cache
def cached_compute_capability():
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    compute_capability = device_props.major * 10 + device_props.minor
    return compute_capability


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
        _weights: packed Floatx weights
        _scales: scale
        splitK: split K

    Returns
        output of linear layer
    """
    # Check if we're on a supported architecture (sm7.5 or higher)
    compute_capability = cached_compute_capability()
    torch._check(
        compute_capability >= 75,
        lambda: f"quant_llm_linear requires sm7.5+ GPU architecture, but current device has sm{compute_capability}",
    )
    return torch.ops.torchao.quant_llm_linear.default(
        EXPONENT, MANTISSA, _in_feats, _weights, _scales, splitK
    )


@register_custom_op("torchao::quant_llm_linear")
def _(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = 1,
) -> Tensor:
    torch._check(
        _in_feats.dim() == 2,
        lambda: f"input should be a 2d tensor, got {_in_feats.dim()}D",
    )
    torch._check(
        _in_feats.dtype in (torch.float16, torch.bfloat16),
        lambda: f"weight must be FP16 or BF16, got {_in_feats.dtype}",
    )
    torch._check(
        _weights.dim() == 2,
        lambda: f"weight should be a 2d tensor, got {_weights.dim()}D",
    )
    torch._check(
        _weights.dtype is torch.uint8,
        lambda: f"weight must be UINT8, got {_weights.dtype}",
    )
    torch._check(
        _scales.dim() == 1, lambda: f"scale should be a 2d tensor, got {_scales.dim()}D"
    )
    torch._check(
        _scales.dtype in (torch.float16, torch.bfloat16),
        lambda: f"scale must be FP16 or BF16, got {_scales.dtype}",
    )

    BS, IC = _in_feats.shape
    OC, _ = _weights.shape
    N_BITS = 1 + EXPONENT + MANTISSA
    torch._check(IC // 8 * N_BITS == _weights.shape[1], lambda: "Dimensions mismatched")
    torch._check(OC == _scales.shape[0], lambda: "Dimensions mismatched")

    return _in_feats.new_empty((BS, OC))


def qscaled_dot_product(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    q_scale: float = 1.0,
    q_zp: int = 0,
    k_scale: float = 1.0,
    k_zp: int = 0,
    v_scale: float = 1.0,
    v_zp: int = 0,
    a_scale: float = 1.0,
    a_zp: int = 0,
    o_scale: float = 1.0,
    o_zp: int = 0,
) -> Tensor:
    """
    Quantized SDPA with quantized inputs and outputs.
    Arguments
        query: input query tensor,
        key: input key tensor,
        value: input value tensor,
        attn_mask: attention mask tensor,
        dropout_p: dropout probability,
        is_causal: causal flag,
        scale: scaling factor applied prior to softmax,
        q_scale: scale for query from linear quantization,
        q_zp: zero point for query from linear quantization,
        k_scale: scale for key from linear quantization,
        k_zp: zero point of key from linear quantization,
        v_scale: zero point for value from linear quantization,
        v_zp: zero point of value from linear quantization,
        a_scale: scale for attention from softmax quantization,
        a_zp: zero point for attention from softmax quantization,
        o_scale: scale for output from linear quantization,
        o_zp: zero point for output from linear quantization,
    Returns
        output of quantized SDPA
    """
    return torch.ops.torchao.qscaled_dot_product.default(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        q_scale,
        q_zp,
        k_scale,
        k_zp,
        v_scale,
        v_zp,
        a_scale,
        a_zp,
        o_scale,
        o_zp,
    )


@register_custom_op("torchao::qscaled_dot_product")
def _(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    q_scale: float = 1.0,
    q_zp: int = 0,
    k_scale: float = 1.0,
    k_zp: int = 0,
    v_scale: float = 1.0,
    v_zp: int = 0,
    a_scale: float = 1.0,
    a_zp: int = 0,
    o_scale: float = 1.0,
    o_zp: int = 0,
) -> Tensor:
    return query


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


@register_custom_op("torchao::unpack_tensor_core_tiled_layout")
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


def dequantize_tensor_core_tiled_layout(
    packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int
) -> Tensor:
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
        group_size: int
        inner_k_tiles: int

    Returns:
        torch.tensor of shape is N x K, dtype is torch.bfloat16

    """
    return torch.ops.torchao.dequantize_tensor_core_tiled_layout.default(
        packed_w, scales_and_zeros, group_size, inner_k_tiles
    )


@register_custom_op("torchao::dequantize_tensor_core_tiled_layout")
def _(
    packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int
) -> Tensor:
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
    torch._check(
        scales_and_zeros.dtype is torch.bfloat16,
        lambda: "scales_and_zeros must be bfloat16",
    )
    torch._check(
        scales_and_zeros.dim() == 3,
        lambda: "scales_and_zeros must be 3D, got {scales_and_zeros.dim()}",
    )
    torch._check(
        group_size == 32 or group_size == 64 or group_size == 128 or group_size == 256,
        lambda: "qGroupSize must be 32, 64, 128, or 256",
    )
    torch._check(
        scales_and_zeros.size(0) == K // group_size,
        lambda: "scales_and_zeros must have K // qGroupSize at dim 0",
    )
    torch._check(
        scales_and_zeros.size(1) == N, lambda: "scales_and_zeros must have N at dim 1"
    )
    torch._check(
        scales_and_zeros.size(2) == 2, lambda: "scales_and_zeros must have 2 at dim 2"
    )

    return torch.empty((N, K), dtype=torch.bfloat16, device=packed_w.device)


def marlin_24_gemm(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    """
    Sparse Marlin 2:4 matrix multiplication. Reference: https://github.com/IST-DASLab/Sparse-Marlin/tree/main
    Args:
        x: input matrix of shape `(n, k/2)` in column-major layout.
        weight_marlin: weight matrix of original shape `(m, k)` in Marlin format; see `Layer.pack()`.
        meta: metadata information for 2:4 sparsity.
        s: scales of shape `(n / groupsize / 2, m)`.
        workspace: tensor with at least `m / 128 * max_par` entries that are all zero.
        bits: number of bits for quantization.
        size_m: number of rows in input matrix.
        size_n: number of columns in weight matrix.
        size_k: number of columns in input matrix.
    Returns:
        output matrix of shape `(n, m)` in column-major layout.
    """
    return torch.ops.torchao.marlin_24_gemm.default(
        x, weight_marlin, meta, s, workspace, bits, size_m, size_n, size_k
    )


@register_custom_op("torchao::marlin_24_gemm")
def _(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    TILE_SIZE = 16
    MIN_THREAD_N = 128
    MAX_PARALLELISM = 64

    # Verify num_bits
    torch._check(
        bits == 4 or bits == 8, lambda: f"num_bits must be 4 or 8. Got = {bits}"
    )
    pack_factor = 32 // bits

    # Verify M
    torch._check(
        size_m == x.size(0),
        lambda: f"Shape mismatch: x.size(0) = {x.size(0)}, size_m = {size_m}",
    )

    # Verify K
    torch._check(
        size_k == x.size(1),
        lambda: f"Shape mismatch: x.size(1) = {x.size(1)}, size_k = {size_k}",
    )
    torch._check(
        size_k % TILE_SIZE == 0,
        lambda: f"size_k = {size_k} is not divisible by tile_size = {TILE_SIZE}",
    )
    torch._check(
        (size_k // TILE_SIZE // 2) == weight_marlin.size(0),
        lambda: f"Shape mismatch: weight_marlin.size(0) = {weight_marlin.size(0)}, size_k = {size_k}, tile_size = {TILE_SIZE}",
    )

    # Verify N
    torch._check(
        s.size(1) == size_n, lambda: f"s.size(1) = {s.size(1)}, size_n = {size_n}"
    )
    torch._check(
        weight_marlin.size(1) % TILE_SIZE == 0,
        lambda: f"weight_marlin.size(1) = {weight_marlin.size(1)} is not divisible by tile_size = {TILE_SIZE}",
    )

    actual_size_n = (weight_marlin.size(1) // TILE_SIZE) * pack_factor
    torch._check(
        size_n == actual_size_n,
        lambda: f"size_n = {size_n}, actual_size_n = {actual_size_n}",
    )

    # Verify meta
    torch._check(
        meta.size(0) == size_k // 8 // 2 // 2,
        lambda: f"meta.size(0) = {meta.size(0)} is not size_k / 8 / 2 / 2 = {size_k // 8 // 2 // 2}",
    )
    torch._check(
        meta.size(1) == size_n * 2,
        lambda: f"meta.size(1) = {meta.size(1)} is not size_n * 2 = {size_n * 2}",
    )

    # Verify A device and strides
    torch._check(x.is_cuda, lambda: "x is not on GPU")
    torch._check(x.is_contiguous(), lambda: "x is not contiguous")

    # Verify B device and strides
    torch._check(weight_marlin.is_cuda, lambda: "weight_marlin is not on GPU")
    torch._check(
        weight_marlin.is_contiguous(), lambda: "weight_marlin is not contiguous"
    )

    # Verify meta device and strides
    torch._check(meta.is_cuda, lambda: "meta is not on GPU")
    torch._check(meta.is_contiguous(), lambda: "meta is not contiguous")

    # Verify scales device and strides
    torch._check(s.is_cuda, lambda: "s is not on GPU")
    torch._check(s.is_contiguous(), lambda: "s is not contiguous")

    # Verify groupsize
    groupsize = -1
    if s.size(0) > 1:
        torch._check(
            size_k % s.size(0) == 0,
            lambda: f"size_k = {size_k} is not divisible by s.size(0) = {s.size(0)}",
        )
        groupsize = size_k // s.size(0)
        groupsize //= 2  # Because of 24
    torch._check(
        groupsize == -1 or groupsize == 64,
        lambda: f"Unexpected groupsize = {groupsize}",
    )

    # Verify workspace size
    torch._check(
        size_n % MIN_THREAD_N == 0,
        lambda: f"size_n = {size_n} is not divisible by min_thread_n = {MIN_THREAD_N}",
    )
    min_workspace_size = (size_n // MIN_THREAD_N) * MAX_PARALLELISM
    torch._check(
        workspace.numel() >= min_workspace_size,
        lambda: f"workspace.numel = {workspace.numel()} is below min_workspace_size = {min_workspace_size}",
    )

    return torch.empty((x.size(0), s.size(1)), dtype=x.dtype, device=x.device)


def marlin_qqq_gemm(
    x: Tensor,
    weight_marlin: Tensor,
    s_tok: Tensor,
    s_ch: Tensor,
    s_group: Tensor,
    workspace: Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    """
    Marlin for W4A8 mixed precision matrix multiplication.
    See https://arxiv.org/pdf/2406.09904 for more details.
    Reference: https://github.com/HandH1998/QQQ/tree/main
    Args:
        x: `torch.int8` input matrix of shape `(m, k)` in standard row-major layout.
        weight_marlin: `torch.int32` weight matrix of original shape `(k, n)` in the specified format.
        s_tok: `torch.float32` activation per-token quantization scales of shape `(m, 1)`.
        s_ch: `torch.float32` weight per-channel quantization scales of shape `(1, n)`.
        s_group: `torch.half` weight per-group quantization scales of shape `(m / groupsize, n)`, it should be empty when group_size != -1.
        workspace: `torch.int32` tensor with at least `n / 128 * max_par` entries that are all zero.
        size_m: number of rows in input matrix.
        size_n: number of columns in weight matrix.
        size_k: number of columns in input matrix.
    Returns:
        `torch.half` out matrix of shape `(m, n)` in standard row-major layout.
    """
    return torch.ops.torchao.marlin_qqq_gemm.default(
        x, weight_marlin, s_tok, s_ch, s_group, workspace, size_m, size_n, size_k
    )


@register_custom_op("torchao::marlin_qqq_gemm")
def _(
    x: Tensor,
    weight_marlin: Tensor,
    s_tok: Tensor,
    s_ch: Tensor,
    s_group: Tensor,
    workspace: Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor:
    TILE_SIZE = 16
    MIN_THREAD_N = 64
    MAX_PARALLELISM = 16
    PACK_FACTOR = 32 // 4

    # Verify M
    torch._check(
        size_m == x.size(0),
        lambda: f"Shape mismatch: x.size(0) = {x.size(0)}, size_m = {size_m}",
    )
    torch._check(
        size_m == s_tok.numel(),
        lambda: f"Shape mismatch: s_tok.numel() = {s_tok.numel()}, size_m = {size_m}",
    )

    # Verify K
    torch._check(
        size_k == x.size(1),
        lambda: f"Shape mismatch: x.size(1) = {x.size(1)}, size_k = {size_k}",
    )
    torch._check(
        size_k % TILE_SIZE == 0,
        lambda: f"size_k = {size_k} is not divisible by tile_size = {TILE_SIZE}",
    )
    torch._check(
        (size_k // TILE_SIZE) == weight_marlin.size(0),
        lambda: f"Shape mismatch: weight_marlin.size(0) = {weight_marlin.size(0)}, size_k = {size_k}, tile_size = {TILE_SIZE}",
    )

    # Verify groupsize
    groupsize = -1 if s_group.numel() == 0 else size_k // s_group.size(0)
    torch._check(groupsize in [-1, 128], lambda: f"Unexpected groupsize = {groupsize}")

    # Verify N
    torch._check(
        s_ch.numel() == size_n,
        lambda: f"Shape mismatch: s_ch.numel() = {s_ch.numel()}, size_n = {size_n}",
    )
    torch._check(
        weight_marlin.size(1) % TILE_SIZE == 0,
        lambda: f"weight_marlin.size(1) = {weight_marlin.size(1)} is not divisible by tile_size = {TILE_SIZE}",
    )
    if groupsize != -1:
        torch._check(
            s_group.size(1) == size_n,
            lambda: f"Shape mismatch: s_group.size(1) = {s_group.size(1)}, size_n = {size_n}",
        )
        torch._check(
            size_k % s_group.size(0) == 0,
            lambda: f"size_k = {size_k} is not divisible by s_group.size(0) = {s_group.size(0)}",
        )

    actual_size_n = (weight_marlin.size(1) // TILE_SIZE) * PACK_FACTOR
    torch._check(
        size_n == actual_size_n,
        lambda: f"Shape mismatch: size_n = {size_n}, actual_size_n = {actual_size_n}",
    )

    # Verify A device and strides
    torch._check(x.is_cuda, lambda: "x is not on GPU")
    torch._check(x.is_contiguous(), lambda: "x is not contiguous")

    # Verify B device and strides
    torch._check(weight_marlin.is_cuda, lambda: "weight_marlin is not on GPU")
    torch._check(
        weight_marlin.is_contiguous(), lambda: "weight_marlin is not contiguous"
    )

    # Verify s_tok device, strides and dtype
    torch._check(s_tok.is_cuda, lambda: "s_tok is not on GPU")
    torch._check(s_tok.is_contiguous(), lambda: "s_tok is not contiguous")
    torch._check(s_tok.dtype == torch.float32, lambda: "s_tok's dtype is not float32")

    # Verify s_ch device, strides and dtype
    torch._check(s_ch.is_cuda, lambda: "s_ch is not on GPU")
    torch._check(s_ch.is_contiguous(), lambda: "s_ch is not contiguous")
    torch._check(s_ch.dtype == torch.float32, lambda: "s_ch's dtype is not float32")

    # Verify s_group device, strides and dtype
    torch._check(s_group.is_cuda, lambda: "s_group is not on GPU")
    torch._check(s_group.is_contiguous(), lambda: "s_group is not contiguous")
    torch._check(s_group.dtype == torch.float16, "s_group's dtype is not float16")

    # Verify workspace size
    torch._check(
        size_n % MIN_THREAD_N == 0,
        lambda: f"size_n = {size_n} is not divisible by min_thread_n = {MIN_THREAD_N}",
    )
    min_workspace_size = (size_n // MIN_THREAD_N) * MAX_PARALLELISM
    torch._check(
        workspace.numel() >= min_workspace_size,
        lambda: f"workspace.numel() = {workspace.numel()} is below min_workspace_size = {min_workspace_size}",
    )

    return torch.empty((size_m, size_n), dtype=torch.float16, device=x.device)


def rowwise_scaled_linear_cutlass_s8s4(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    CUTLASS-based row-wise scaled W4A8 linear operator.
    Args:
        input: quantized input tensor, in row-major layout.
        input_scale: scale factors for input tensor, has to be tensor of the same shape as the input tensor, minus the last dimension.
        weight: quantized weight matrix, in row-major layout.
        weight_scale: scale factors for weight tensor, one value per row of weight matrix (thus also tensor of the same shape as the weight tensor, minus the last dimension).
        bias: an optional vector of size equal to number of rows of weight tensor, or None.
        out_dtype: optional data type for output tensor.
    Returns:
        output: result tensor, in row-major layout.
    """

    return torch.ops.torchao.rowwise_scaled_linear_cutlass_s8s4.default(
        input,
        input_scale,
        weight,
        weight_scale,
        bias,
        out_dtype,
    )


@register_custom_op("torchao::rowwise_scaled_linear_cutlass_s8s4")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    # No checks here, as detailed checks are performed by the
    # operator itself.

    dtype = out_dtype if out_dtype is not None else input_scale.dtype
    device = input.device
    return torch.empty((*input.shape[:-1], weight.shape[0]), dtype=dtype, device=device)


def rowwise_scaled_linear_cutlass_s4s4(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    CUTLASS-based row-wise scaled W4A4 linear operator.
    Args:
        input: quantized input tensor, in row-major layout.
        input_scale: scale factors for input tensor, has to be tensor of the same shape as the input tensor, minus the last dimension.
        weight: quantized weight matrix, in row-major layout.
        weight_scale: scale factors for weight tensor, one value per row of weight matrix (thus also tensor of the same shape as the weight tensor, minus the last dimension).
        bias: an optional vector of size equal to number of rows of weight tensor, or None.
        out_dtype: optional data type for output tensor.
    Returns:
        output: result tensor, in row-major layout.
    """

    return torch.ops.torchao.rowwise_scaled_linear_cutlass_s4s4.default(
        input, input_scale, weight, weight_scale, bias, out_dtype
    )


@register_custom_op("torchao::rowwise_scaled_linear_cutlass_s4s4")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    # No checks here, as detailed checks are performed by the
    # operator itself.

    dtype = out_dtype if out_dtype is not None else input_scale.dtype
    device = input.device
    return torch.empty((*input.shape[:-1], weight.shape[0]), dtype=dtype, device=device)


def rowwise_scaled_linear_sparse_cutlass_f8f8(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_meta: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    CUTLASS-based row-wise scaled F8F8 linear operator, for sparsified weight case.
    Args:
        input: quantized input tensor, in row-major layout.
        input_scale: scale factors for input tensor, has to be tensor of the same shape as the input tensor, minus the last dimension.
        weight: sparsified quantized weight matrix, in row-major layout.
        weight_meta: sparsify metadata for weight tensor.
        weight_scale: scale factors for weight tensor, one value per row of weight matrix (thus also tensor of the same shape as the weight tensor, minus the last dimension).
        bias: an optional vector of size equal to number of rows of weight tensor, or None.
        out_dtype: optional data type for output tensor.
    Returns:
        output: result tensor, in row-major layout.
    """

    return torch.ops.torchao.rowwise_scaled_linear_sparse_cutlass_f8f8.default(
        input, input_scale, weight, weight_meta, weight_scale, bias, out_dtype
    )


@register_custom_op("torchao::rowwise_scaled_linear_sparse_cutlass_f8f8")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_meta: Tensor,
    weight_scale: Tensor,
    bias: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    # No checks here, as detailed checks are performed by the
    # operator itself.

    dtype = out_dtype if out_dtype is not None else input_scale.dtype
    device = input.device
    return torch.empty((*input.shape[:-1], weight.shape[0]), dtype=dtype, device=device)


def to_sparse_semi_structured_cutlass_sm9x_f8(
    weight: Tensor,
) -> (Tensor, Tensor):
    """
    CUTLASS-based conversion from sparsified input tensor to corresponding compressed tensor, along with corresponding metadata tensor.
    Args:
        weight: input tensor, in row-major layout.
    Returns:
        weight_compressed: compressed weight tensor, with sparsity eliminated, in row-major layout.
        weight_meta: metadata tensor, describing the sparsity structure of the input tensor, also in row-major layout.
    """

    return torch.ops.torchao.to_sparse_semi_structured_cutlass_sm9x_f8.default(weight)


@register_custom_op("torchao::to_sparse_semi_structured_cutlass_sm9x_f8")
def _(
    weight: Tensor,
) -> (Tensor, Tensor):
    # No checks here, as detailed checks are performed by the
    # operator itself.

    return (
        weight.new_empty(weight[0], weight[1] // 2),
        weight.new_empty(weight[0], max(weight[1] // 8, 16), dtype=torch.char),
    )


def sparse24_sm90_sparsify(
    input_tensor: Tensor,
    metadata_format: str,
    activation: str,
    algorithm: str,
    dtype=None,
    scale=None,
) -> (Tensor, Tensor):
    return torch.ops.torchao.sparse24_sm90_sparsify(
        input_tensor, metadata_format, activation, algorithm, dtype=dtype, scale=scale
    )


def swizzle_mm(
    mat1: Tensor, mat2: Tensor, mat1_is_swizzled: bool, mat2_is_swizzled: bool
) -> Tensor:
    """
    Similar to torch.mm but Tensor inputs can be SwizzleTensor instances.

    """
    return torch.ops.torchao.swizzle_mm.default(
        mat1, mat2, mat1_is_swizzled, mat2_is_swizzled
    )


@register_custom_op("torchao::swizzle_mm")
def _(
    mat1: Tensor, mat2: Tensor, mat1_is_swizzled: bool, mat2_is_swizzled: bool
) -> Tensor:
    return mat1.new_empty(mat1.shape[0], mat2.shape[1])


def swizzle_scaled_mm(
    mat1: Tensor,
    mat2: Tensor,
    mat1_is_swizzled: bool,
    mat2_is_swizzled: bool,
    scale_a: Tensor,
    scale_b: Tensor,
    bias: Optional[Tensor],
    scale_result: Optional[Tensor],
    out_dtype: Optional[torch.dtype],
) -> Tensor:
    """
    Similar to torch.mm but Tensor inputs can be SwizzleTensor instances.

    """
    return torch.ops.torchao.swizzle_scaled_mm.default(
        mat1,
        mat2,
        mat1_is_swizzled,
        mat2_is_swizzled,
        scale_a,
        scale_b,
        bias,
        scale_result,
        out_dtype,
    )


@register_custom_op("torchao::swizzle_scaled_mm")
def _(
    mat1: Tensor,
    mat2: Tensor,
    mat1_is_swizzled: bool,
    mat2_is_swizzled: bool,
    scale_a: Tensor,
    scale_b: Tensor,
    bias: Optional[Tensor],
    scale_result: Optional[Tensor],
    out_dtype: Optional[torch.dtype],
) -> Tensor:
    return mat1.new_empty(mat1.shape[0], mat2.shape[1])


@functools.lru_cache()
def _get_dtypes():
    """TODO: when e8m0 is hardened and major release lets remove uint8 support"""
    if hasattr(torch, "float8_e8m0fnu"):
        return (torch.uint8, torch.float8_e8m0fnu)
    return (torch.uint8,)


def _check_scale_dtypes(A_scale, B_scale):
    allowed_dtypes = _get_dtypes()

    torch._check(
        A_scale.dtype in allowed_dtypes,
        lambda: f"A_scale tensor must be uint8 or float8_e8m0fnu, got {A_scale.dtype}",
    )
    torch._check(
        B_scale.dtype in allowed_dtypes,
        lambda: f"B_scale tensor must be uint8 or float8_e8m0fnu, got {B_scale.dtype}",
    )


@register_custom_op("torchao::mx_fp8_bf16")
def meta_mx_fp8_bf16(A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor):
    """Meta impl for mx_fp8_bf16"""
    return torch.empty((A.size(0), B.size(1)), dtype=torch.bfloat16, device=A.device)


def mx_fp4_bf16(A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor):
    """Defines a matmul between two fp4 tensors w/ MX scales in E8MO and returns a bf16 tensor.

    The expected format is fp4_e2m1 specified:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final.pdf (Section 5.3.3)

    Note: The mx scales are E8MO tensors stored in uint8 tensors (for now).
        The layout of the scales is very particular, see:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout


    Args:
        A: fp4 tensor (2 fp4 elements are packed into 1 byte -> elem0|elem1)
        B: fp4 tensor (2 fp4 elements are packed into 1 byte -> elem0|elem1)
        A_scale: E8M0 scale tensor for A with groupsize=32 in swizzled layout
        B_scale: E8M0 scale tensor for B with groupsize=32 in swizzled layout

    Returns:
        MXN bf16 Tensor

    """
    _check_scale_dtypes(A_scale, B_scale)
    return torch.ops.torchao.mx_fp4_bf16.default(A, B, A_scale, B_scale)


@register_custom_op("torchao::mx_fp4_bf16")
def meta_mx_fp4_bf16(A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor):
    """Meta impl for mx_fp4_bf16"""
    # Assume that the contraction happens in the K dim thus M,N are perserved post bit pack
    return torch.empty((A.size(0), B.size(1)), dtype=torch.bfloat16, device=A.device)
