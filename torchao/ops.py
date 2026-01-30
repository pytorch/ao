# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Optional

import torch
from torch import Tensor

lib = torch.library.Library("torchao", "FRAGMENT")
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
    "sparse24_fp8_sm90_cutlass_gemm(Tensor a, Tensor a_mdata, Tensor b, *, Tensor? a_scale = None, Tensor? b_scale = None, int swizzle_size=8, str swizzle_axis='n', int sm_count=128) -> Tensor"
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
    "qscaled_dot_product(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, float? scale=None, float q_scale=1.0, int q_zp=0, float k_scale=1.0, int k_zp=0, float v_scale=1.0, int v_zp=0, float a_scale=1.0, int a_zp=0, float o_scale=1.0, int o_zp=0) -> Tensor"
)
lib.define(
    "da8w4_linear_prepack_cpu(Tensor weight, Tensor scales, Tensor qzeros) -> (Tensor, Tensor, Tensor, Tensor)"
)
lib.define(
    "da8w4_linear_cpu(Tensor input, Tensor input_scales, Tensor input_qzeros, Tensor weight, Tensor weight_scales, Tensor weight_qzeros, Tensor compensation, Tensor? bias, ScalarType output_dtype) -> Tensor"
)
lib.define(
    "_scaled_embedding_bag(Tensor qweight, Tensor indices, Tensor offsets, Tensor weight_scale, float o_scale, int mode, bool include_last_offset, ScalarType output_dtype) -> Tensor"
)
lib.define(
    "float8_linear_prepack_cpu(Tensor weight, Tensor scales) -> (Tensor, Tensor)"
)
lib.define(
    "float8_linear_cpu(Tensor input, Tensor input_scales, Tensor weight, Tensor weight_scales, Tensor? bias, ScalarType output_dtype) -> Tensor"
)


def register_custom_op(name):
    def decorator(func):
        return torch.library.register_fake(f"{name}")(func)

    return decorator


def register_custom_op_impl(name):
    def decorator(func):
        return torch.library.custom_op(f"{name}", mutates_args=())(func)

    return decorator


@functools.lru_cache
def cached_compute_capability():
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    compute_capability = device_props.major * 10 + device_props.minor
    return compute_capability


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


@register_custom_op("torchao::sparse24_sm90_sparsify")
def _(
    input_tensor: Tensor,
    metadata_format: str,
    activation: str,
    algorithm: str,
    dtype=None,
    scale=None,
):
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    return (
        torch.empty(
            (input_tensor.shape[0], input_tensor.shape[1] // 2),
            dtype=out_dtype,
            device=input_tensor.device,
        ),
        torch.empty(
            (input_tensor.shape[0], input_tensor.shape[1] // 8),
            dtype=torch.uint8,
            device=input_tensor.device,
        ),
    )


def sparse24_fp8_sm90_cutlass_gemm(
    a: Tensor,
    meta: Tensor,
    b: Tensor,
    a_scale: Optional[Tensor] = None,
    b_scale: Optional[Tensor] = None,
    swizzle_size: int = 8,
    swizzle_axis: str = "n",
    sm_count: int = 128,
) -> Tensor:
    return torch.ops.torchao.sparse24_fp8_sm90_cutlass_gemm(
        a,
        meta,
        b,
        a_scale=a_scale,
        b_scale=b_scale,
        swizzle_size=swizzle_size,
        swizzle_axis=swizzle_axis,
        sm_count=sm_count,
    )


@register_custom_op("torchao::sparse24_fp8_sm90_cutlass_gemm")
def _(
    a: Tensor,
    meta: Tensor,
    b: Tensor,
    a_scale: Optional[Tensor] = None,
    b_scale: Optional[Tensor] = None,
    swizzle_size: int = 8,
    swizzle_axis: str = "n",
    sm_count: int = 128,
):
    return torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=a.device)


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


def da8w4_linear_prepack_cpu(
    weight: Tensor,
    scales: Tensor,
    qzeros: Tensor,
) -> Tensor:
    """
    Prepack weights for DA8W4 linear operator on CPU.
    Args:
        weight: weight tensor.
        scales: scales for weight tensor.
        qzeros: zero points for weight tensor.
    Returns:
        packed weight, scales, and zero points.
    """
    return torch.ops.torchao.da8w4_linear_prepack_cpu.default(weight, scales, qzeros)


@register_custom_op("torchao::da8w4_linear_prepack_cpu")
def _(weight: Tensor, scales: Tensor, qzeros: Tensor) -> Tensor:
    return weight, scales, qzeros, torch.Tensor()


def da8w4_linear_cpu(
    input: Tensor,
    input_scales: Tensor,
    input_qzeros: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    weight_qzeros: Tensor,
    compensation: Tensor,
    bias: Optional[Tensor],
    out_dtype: torch.dtype,
):
    """
    DA8W4 linear operator on CPU.
    Args:
        input: input tensor.
        input_scales: scales for input tensor.
        input_qzeros: zero points for input tensor.
        weight: weight tensor.
        weight_scales: scales for weight tensor.
        weight_qzeros: zero points for weight tensor.
        compensation: compensation tensor for weight.
        bias: optional bias tensor.
        out_dtype: output data type.
    Returns:
        output tensor in out_dtype.
    """
    return torch.ops.torchao.da8w4_linear_cpu.default(
        input,
        input_scales,
        input_qzeros,
        weight,
        weight_scales,
        weight_qzeros,
        compensation,
        bias,
        out_dtype,
    )


@register_custom_op("torchao::da8w4_linear_cpu")
def _(
    input: Tensor,
    input_scales: Tensor,
    input_qzeros: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    weight_qzeros: Tensor,
    compensation: Tensor,
    bias: Optional[Tensor],
    out_dtype: torch.dtype,
) -> Tensor:
    assert weight.dim() == 4
    N = weight.size(0) * weight.size(3) * 2
    return input.new_empty(*input.shape[:-1], N, dtype=out_dtype)


@register_custom_op("torchao::_scaled_embedding_bag")
def _(
    qweight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    w_scales: Tensor,
    o_scale: float,
    mode: int,
    include_last_offset: bool,
    out_dtype: torch.dtype,
) -> Tensor:
    # Only support include_last_offset == True
    assert include_last_offset == True
    batch_size = offsets.shape[0] - 1
    return qweight.new_empty(batch_size, qweight.shape[1], dtype=out_dtype)


def float8_linear_prepack_cpu(
    weight: Tensor,
    scales: Tensor,
) -> Tensor:
    """
    Prepack weights for float8 linear operator on CPU.
    Args:
        weight: weight tensor.
        scales: scales for weight tensor.
    Returns:
        packed weight, packed scales
    """
    return torch.ops.torchao.float8_linear_prepack_cpu.default(weight, scales)


@register_custom_op("torchao::float8_linear_prepack_cpu")
def _(weight: Tensor, scales: Tensor) -> Tensor:
    return weight, scales


def float8_linear_cpu(
    input: Tensor,
    input_scales: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    bias: Optional[Tensor],
    out_dtype: torch.dtype,
):
    """
    float8 linear operator on CPU.
    Args:
        input: input tensor.
        input_scales: scales for input tensor.
        weight: weight tensor.
        weight_scales: scales for weight tensor.
        bias: optional bias tensor.
        out_dtype: output data type.
    Returns:
        output tensor in out_dtype.
    """
    return torch.ops.torchao.float8_linear_cpu.default(
        input,
        input_scales,
        weight,
        weight_scales,
        bias,
        out_dtype,
    )


@register_custom_op("torchao::float8_linear_cpu")
def _(
    input: Tensor,
    input_scales: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    bias: Optional[Tensor],
    out_dtype: torch.dtype,
) -> Tensor:
    assert weight.dim() in (2, 4)
    N = weight.size(0) * weight.size(3) if weight.dim() == 4 else weight.size(0)
    return input.new_empty(*input.shape[:-1], N, dtype=out_dtype)
