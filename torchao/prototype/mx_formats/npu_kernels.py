# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Ascend NPU (torch_npu) MX kernels for ``KernelPreference.ASCEND``.

The implementations mirror the behavior already validated on Ascend NPU
hardware in the torchao_npu patch layer: quantize dispatches to
``torch_npu.npu_dynamic_mx_quant``, MXFP8 gemm dispatches to
``torch._scaled_mm`` (routed to NPU by torch_npu), and MXFP4 gemm dispatches
to ``torch_npu.npu_quant_matmul``.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Union

import torch

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.utils import (
    from_blocked,
    hp_data_dims_to_swizzled_scale_dims_mx,
    to_blocked,
)

# torch_npu expects the target MX dtype as an integer code.
_NPU_MX_DST_TYPE = {
    torch.float4_e2m1fn_x2: 296,
    torch.float8_e4m3fn: 292,
    torch.float8_e5m2: 291,
}

# Map TorchAO scale modes to torch_npu's per-dtype scale algorithms.
_NPU_SCALE_ALG = {
    ScaleCalculationMode.FLOOR: 0,
    ScaleCalculationMode.RCEIL: 1,
}
_NPU_MXFP4_SCALE_ALG = {
    ScaleCalculationMode.FLOOR: 0,
    ScaleCalculationMode.RCEIL: 2,
}


def _require_torch_npu():
    """Import torch_npu, raising a clear error when it is not installed."""
    try:
        import torch_npu
    except ImportError as e:
        raise AssertionError(
            "KernelPreference.ASCEND requires the torch_npu package to be "
            "installed (see https://github.com/ascend/pytorch)."
        ) from e
    return torch_npu


def _npu_scale_alg(elem_dtype: Any, scaling_mode: ScaleCalculationMode) -> int:
    mapping = (
        _NPU_MXFP4_SCALE_ALG
        if elem_dtype == torch.float4_e2m1fn_x2
        else _NPU_SCALE_ALG
    )
    return mapping[scaling_mode]


def _pad_to_mx_blocksize_x2(
    tensor: torch.Tensor, axis: int, mx_blocksize: int = 32
) -> torch.Tensor:
    """Pad ``axis`` to a multiple of ``2 * mx_blocksize`` with zeros."""
    mx_blocksize_x2 = 2 * mx_blocksize
    dim_size = tensor.shape[axis]
    remainder = dim_size % mx_blocksize_x2
    if remainder == 0:
        return tensor
    pad_amount = mx_blocksize_x2 - remainder

    ndim = tensor.dim()
    pos_axis = axis if axis >= 0 else ndim + axis
    pad = [0] * (2 * ndim)
    right_pad_slot = 2 * (ndim - 1 - pos_axis) + 1
    pad[right_pad_slot] = pad_amount
    return torch.nn.functional.pad(tensor, tuple(pad), mode="constant", value=0)


def _swizzle_mx_scale_as_e8m0(scale_2d_e8m0: torch.Tensor) -> torch.Tensor:
    """Swizzle E8M0 scales through uint8 because NPU lacks E8M0 tensor ops."""
    blocked_u8 = to_blocked(scale_2d_e8m0.contiguous().view(torch.uint8))
    return blocked_u8.view(torch.float8_e8m0fnu)


def _unswizzle_mx_scale_as_e8m0(
    blocked_scale_e8m0: torch.Tensor,
    original_rows: int,
    original_cols: int,
) -> torch.Tensor:
    unblocked_u8 = from_blocked(
        blocked_scale_e8m0.contiguous().view(torch.uint8),
        original_rows,
        original_cols,
    )
    return unblocked_u8.view(torch.float8_e8m0fnu)


def _get_mx_scale(mx_tensor: Any) -> torch.Tensor:
    """Return the MX scale in the natural ``[M, K / block_size]`` layout."""
    scale = mx_tensor.scale
    if not mx_tensor.is_swizzled_scales:
        return scale

    is_transposed = mx_tensor.qdata.stride(-2) < mx_tensor.qdata.stride(-1)
    if is_transposed:
        leading_dims, m_dim, k_dim = (
            mx_tensor.shape[:-2],
            mx_tensor.shape[-1],
            mx_tensor.shape[-2],
        )
        scale = scale.transpose(-2, -1)
    else:
        leading_dims, m_dim, k_dim = (
            mx_tensor.shape[:-2],
            mx_tensor.shape[-2],
            mx_tensor.shape[-1],
        )

    scale = _unswizzle_mx_scale_as_e8m0(
        scale, math.prod(leading_dims) * m_dim, k_dim // mx_tensor.block_size
    )
    scale = scale.view(*leading_dims, m_dim, k_dim // mx_tensor.block_size)
    if is_transposed:
        scale = scale.transpose(-2, -1)
    return scale


def to_mx_npu(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode,
    is_swizzled_scales: bool = False,
):
    """Quantize a high-precision tensor to MX via ``torch_npu.npu_dynamic_mx_quant``.

    Returns ``(scale_e8m0_biased, data_lp)``, mirroring
    ``torchao.prototype.mx_formats.mx_tensor.to_mx``.
    """
    torch_npu = _require_torch_npu()

    data_hp = data_hp.contiguous()
    data_hp = _pad_to_mx_blocksize_x2(data_hp, -1)

    dst_type = _NPU_MX_DST_TYPE[elem_dtype]
    scale_alg = _npu_scale_alg(elem_dtype, scaling_mode)
    qdata, packed_scale = torch_npu.npu_dynamic_mx_quant(
        data_hp,
        axis=-1,
        round_mode="rint",
        dst_type=dst_type,
        block_size=block_size,
        scale_alg=scale_alg,
    )

    block_count = data_hp.shape[-1] // block_size
    scale = packed_scale.contiguous().reshape(*data_hp.shape[:-1], -1)[..., :block_count]
    scale = scale.contiguous().view(torch.float8_e8m0fnu)

    if is_swizzled_scales:
        leading_dims, m_dim, k_dim = (
            data_hp.shape[:-2],
            data_hp.shape[-2],
            data_hp.shape[-1],
        )
        scale_shape = (math.prod(leading_dims) * m_dim, k_dim // block_size)
        scale = _swizzle_mx_scale_as_e8m0(scale.view(scale_shape)).flatten()
        scale_m, scale_k = hp_data_dims_to_swizzled_scale_dims_mx(m_dim, k_dim)
        scale = scale.view(*leading_dims, scale_m, scale_k)

    return scale, qdata


def _build_mxfp8_scaled_mm_scale_args(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    device: torch.device,
):
    """Pack natural MXFP8 scales for NPU ``torch._scaled_mm``.

    ``a_scale`` and ``b_scale`` use ``[rows, K / 32]``. The NPU operator
    consumes ``[M, K / 64, 2]`` and ``[K / 64, N, 2]`` respectively.
    """
    if a_scale.shape[-1] != b_scale.shape[-1]:
        raise RuntimeError(
            "MXFP8 activation and weight scales must have the same K blocks: "
            f"got {a_scale.shape[-1]} and {b_scale.shape[-1]}."
        )
    if a_scale.shape[-1] % 2 != 0:
        raise RuntimeError(
            "MXFP8 NPU scale packing requires an even number of K / 32 blocks; "
            f"got {a_scale.shape[-1]}."
        )

    k64 = a_scale.shape[-1] // 2
    pertoken_scale = a_scale.reshape(-1, k64, 2).contiguous()
    weight_scale = b_scale.reshape(-1, k64, 2).contiguous().transpose(0, 1)
    if pertoken_scale.device != device:
        pertoken_scale = pertoken_scale.to(device)
    if weight_scale.device != device:
        weight_scale = weight_scale.to(device)
    return weight_scale, pertoken_scale


def _build_mxfp4_npu_quant_matmul_scale_args(
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    k_dim: int,
    block_size: int,
    device: torch.device,
):
    """Pack MXFP4 scales for ``npu_quant_matmul``."""
    k64 = k_dim // block_size

    # Keep [K / 64, N, 2] as a transpose view over [N, K / 64, 2].
    weight_scale_base = b_scale.t().reshape(-1, k64, 2).contiguous()
    if weight_scale_base.device != device:
        weight_scale_base = weight_scale_base.to(device)
    weight_scale = weight_scale_base.transpose(0, 1)

    pertoken_scale = a_scale.view(-1, k64, 2)
    if pertoken_scale.device != device:
        pertoken_scale = pertoken_scale.to(device)

    return weight_scale, pertoken_scale


def _mxfp8_scaled_mm(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
    block_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    leading_shape = tuple(a_data.shape[:-1])
    k_dim = a_data.size(-1)
    rows_m = math.prod(leading_shape) if leading_shape else 1

    if len(leading_shape) <= 1 and a_data.dim() == 2 and a_data.is_contiguous():
        a_data_prepared = a_data
    else:
        a_data_prepared = a_data.contiguous().view(rows_m, k_dim)
    # Keep x2 as a transpose view matching the packed weight scale layout.
    b_data_prepared = b_data.transpose(-2, -1).contiguous().transpose(-2, -1)

    if block_size != 32:
        raise RuntimeError(
            f"NPU torch._scaled_mm requires MXFP8 block_size=32, got {block_size}."
        )
    scale, pertoken_scale = _build_mxfp8_scaled_mm_scale_args(
        a_scale=a_scale.reshape(rows_m, -1),
        b_scale=b_scale.t(),
        device=a_data.device,
    )
    out = torch._scaled_mm(
        a_data_prepared,
        b_data_prepared,
        scale_a=pertoken_scale,
        scale_b=scale,
        bias=bias,
        out_dtype=output_dtype,
        use_fast_accum=False,
    )
    if len(leading_shape) > 1:
        out = out.view(*leading_shape, out.size(-1))
    return out


def _mxfp4_npu_matmul(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
    block_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch_npu = _require_torch_npu()

    leading_shape = tuple(a_data.shape[:-1])
    k_dim = a_data.size(-1)
    rows_m = math.prod(leading_shape) if leading_shape else 1

    if len(leading_shape) <= 1 and a_data.dim() == 2 and a_data.is_contiguous():
        a_data_prepared = a_data
    else:
        a_data_prepared = a_data.contiguous().view(rows_m, k_dim)
    # Keep x2 as a transpose view matching the packed weight scale layout.
    b_data_prepared = b_data.transpose(-2, -1).contiguous().transpose(-2, -1)

    scale, pertoken_scale = _build_mxfp4_npu_quant_matmul_scale_args(
        a_scale=a_scale,
        b_scale=b_scale,
        k_dim=k_dim,
        block_size=block_size,
        device=a_data.device,
    )
    scale_dtype = torch_npu.float8_e8m0fnu
    mxfp4_dtype_npu = torch_npu.float4_e2m1fn_x2
    out = torch_npu.npu_quant_matmul(
        a_data_prepared,
        b_data_prepared,
        scale,
        pertoken_scale=pertoken_scale,
        scale_dtype=scale_dtype,
        pertoken_scale_dtype=scale_dtype,
        output_dtype=output_dtype,
        group_sizes=[1, 1, block_size],
        x1_dtype=mxfp4_dtype_npu,
        x2_dtype=mxfp4_dtype_npu,
    )
    if len(leading_shape) > 1:
        out = out.view(*leading_shape, out.size(-1))
    # npu_quant_matmul has no bias operand; add it separately.
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


def mx_gemm_npu(a: Any, b: Any, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Ascend NPU MX gemm for ``KernelPreference.ASCEND``.

    Routes MXFP8 to ``torch._scaled_mm`` and MXFP4 to
    ``torch_npu.npu_quant_matmul``. ``a`` and ``b`` are ``MXTensor``.
    """
    _require_torch_npu()
    assert a.is_swizzled_scales == b.is_swizzled_scales
    assert a.block_size == b.block_size

    a_scale = _get_mx_scale(a)
    b_scale = _get_mx_scale(b)

    if a.elem_dtype == torch.float4_e2m1fn_x2 and b.elem_dtype == torch.float4_e2m1fn_x2:
        return _mxfp4_npu_matmul(
            a.qdata,
            a_scale,
            b.qdata,
            b_scale,
            output_dtype=torch.bfloat16,
            block_size=a.block_size,
            bias=bias,
        )
    return _mxfp8_scaled_mm(
        a.qdata,
        a_scale,
        b.qdata,
        b_scale,
        output_dtype=torch.bfloat16,
        block_size=a.block_size,
        bias=bias,
    )
