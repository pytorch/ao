# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from torch._dynamo import is_compiling as dynamo_is_compiling
from torch._higher_order_ops.out_dtype import out_dtype
from torch.ao.quantization.fx._decomposed import (
    _quant_min_max_bounds_check,
    quantized_decomposed_lib,
)
from torch.library import impl

__all__ = [
    "safe_int_mm",
    "dynamically_quantize_per_tensor",
    "quantize_activation_per_token_absmax",
    "dynamically_quantize_per_channel",
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "quant_int8_dynamic_linear",
    "quant_int8_matmul",
    "quant_int8_dynamic_per_token_linear",
    "quant_int8_per_token_matmul",
    "get_groupwise_affine_qparams",
    "pack_tinygemm_scales_and_zeros",
    "unpack_tinygemm_scales_and_zeros",
    "groupwise_affine_quantize_tensor_from_qparams",
    "groupwise_affine_dequantize_tensor_from_qparams",
    "groupwise_affine_quantize_tensor",
    "groupwise_affine_dequantize_tensor",
    # TODO: need to clean up above functions
    "per_token_dynamic_quant",
    "get_group_qparams_symmetric",
]


def safe_int_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    r"""
    This function wraps torch._int_mm and avoids several undesirable behaviors of the function for certain inputs while still
    returning correct results and being torch.compiled in a performant way.

    Assumes both tensors have dimension of 2.

    Note: no error checking for torch.compiled path, if input.shape = [i, j] and j<=16 then the triton kernel
    will error.

    Args:
        input (Tensor, int8): the first tensor to be multiplied
        mat2 (Tensor, int8): the second tensor to be multiplied

    Return:
        out (Tensor, int32): the result of the matmul with device matching that of the inputs
    """

    # torch.compile path
    if dynamo_is_compiling() or "FakeTensor" in input.__repr__():
        return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)

    # error checking for cublas path
    assert (
        mat2.device == input.device
    # pyre-fixme[16]: Callable `input` has no attribute `device`.
    ), f"need both tensors to be on the same device but got {mat2.device} and {input.device}"
    device_cpu = "cpu" in [mat2.device.type, input.device.type]
    # with input.shape = [i,j] and mat2.shape = [j,k]
    i_is_strictly_greater_than_16 = input.shape[0] > 16
    j_is_nonzero_multiple_of_8 = (input.shape[1] % 8 == 0) and (input.shape[1] > 0)
    k_is_nonzero_multiple_of_8 = (mat2.shape[1] % 8 == 0) and (mat2.shape[1] > 0)
    bad_dimensions_for_cublas = not (
        i_is_strictly_greater_than_16
        and j_is_nonzero_multiple_of_8
        and k_is_nonzero_multiple_of_8
    )

    if device_cpu or bad_dimensions_for_cublas:
        # fallback path
        return torch.matmul(input.cpu().to(torch.int32), mat2.cpu().to(torch.int32)).to(
            input.device.type
        )

    # cublas paths
    if not mat2.is_contiguous():  # silently gives incorrect result without this
        mat2 = mat2.contiguous()
    if (not input.is_contiguous()) and (
        input.shape[0] % 8 != 0
    ):  # gives cryptic error without this
        input = (
            input.contiguous()
        )  # (it seems the transpose makes cublas check the above j constraint on i)
    return out_dtype(torch.ops.aten.mm.default, torch.int32, input, mat2)


# copy-pasta of https://www.internalfb.com/intern/anp/view/?id=3350736
# pyre-fixme[3]: Return type must be annotated.
def dynamically_quantize_per_tensor(
    # pyre-fixme[2]: Parameter must be annotated.
    x,
    # pyre-fixme[2]: Parameter must be annotated.
    quant_min,
    # pyre-fixme[2]: Parameter must be annotated.
    quant_max,
    # pyre-fixme[2]: Parameter must be annotated.
    target_dtype,
    # pyre-fixme[2]: Parameter must be annotated.
    qscheme=torch.per_tensor_affine,  # for now, reuse existing qscheme enum
):
    # assumes affine quantization

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    if qscheme == torch.per_tensor_affine:
        # get min and max
        # TODO(future): make torch.aminmax work on cpu-half
        # min_val, max_val = torch.aminmax(x)
        min_val = torch.min(x)
        max_val = torch.max(x)

        # calculate scale and zero point based on min and max
        # reference: https://fburl.com/code/srbiybme
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # TODO(future): make torch.clamp with scalar work on cpu-half
        scale = torch.clamp(scale, min=eps).reshape(1)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # quantize based on qmin/qmax/scale/zp
        # reference: torch/ao/quantization/fx/_decomposed.py?lines=63
        quant = torch.clamp(
            torch.round(x / scale) + zero_point, quant_min, quant_max
        ).to(target_dtype)

    else:
        assert qscheme == torch.per_tensor_symmetric, f"unsupported qscheme {qscheme}"
        # assert quant_min == -1 * quant_max, "unsupported quant_min/quant_max"
        amax = torch.max(torch.abs(x))
        scale = amax / (float(quant_max - quant_min) / 2)
        scale = torch.clamp(scale, min=eps).reshape(1)
        quant = torch.clamp(torch.round(x / scale), quant_min, quant_max).to(
            target_dtype
        )
        # do not create a tensor for zero_point as this is expensive
        zero_point = None

    return quant, scale, zero_point


# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def quantize_activation_per_token_absmax(t):
    n_bits = 8
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]

    scales = t.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max)
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
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
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scale, zero_point


# reference: https://fburl.com/code/vfsygwd0
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    y = int_repr.to(out_dtype)
    if zero_point is not None:
        y -= zero_point
    return y * scale


# reference: https://fburl.com/code/org0fmi3
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    # assumes axis is 0
    y = int_repr.transpose(0, 1)
    y = y.to(out_dtype)
    y = y - zero_points
    y = y * scales
    y = y.transpose(0, 1)
    return y


# pyre-fixme[3]: Return type must be annotated.
def quant_int8_dynamic_linear(
    # pyre-fixme[2]: Parameter must be annotated.
    x,
    # pyre-fixme[2]: Parameter must be annotated.
    x_quant_min,
    # pyre-fixme[2]: Parameter must be annotated.
    x_quant_max,
    # pyre-fixme[2]: Parameter must be annotated.
    x_q_dtype,
    # pyre-fixme[2]: Parameter must be annotated.
    w_vals_int8_t,
    # pyre-fixme[2]: Parameter must be annotated.
    w_scales,
    # pyre-fixme[2]: Parameter must be annotated.
    w_vals_int8_t_sums_int64,
    # pyre-fixme[2]: Parameter must be annotated.
    bias,
    # pyre-fixme[2]: Parameter must be annotated.
    out_dtype=torch.float32,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scale, x_zp = dynamically_quantize_per_tensor(
        x, x_quant_min, x_quant_max, x_q_dtype
    )
    # w_vals_int8_t_sums_int64 = w_vals_int8_t.sum(dim=0)
    mm_out = quant_int8_matmul(
        x_vals_int8,
        x_scale,
        x_zp,
        w_vals_int8_t,
        w_vals_int8_t_sums_int64,
        w_scales,
        out_dtype,
    )
    if bias is not None:
        mm_out += bias
    return mm_out


# pyre-fixme[3]: Return type must be annotated.
def quant_int8_matmul(
    # pyre-fixme[2]: Parameter must be annotated.
    x_vals_int8,
    # pyre-fixme[2]: Parameter must be annotated.
    x_scale,
    # pyre-fixme[2]: Parameter must be annotated.
    x_zp,
    # pyre-fixme[2]: Parameter must be annotated.
    w_vals_int8_t,
    # pyre-fixme[2]: Parameter must be annotated.
    w_vals_int8_t_sums_int64,
    # pyre-fixme[2]: Parameter must be annotated.
    w_scales,
    # pyre-fixme[2]: Parameter must be annotated.
    out_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # out_dtype. For now, this is written for approximate numerical
    # correctness, and things like aligning accumulation behaviors and
    # performance optimizations are left for a future PR.
    # Assumes that weight quantization is symmetric, i.e. w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw (dot(X_i, W_j) - zx * sum(W_j))
    #

    assert x_vals_int8.dtype in (
        torch.uint8,
        torch.int8,
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"
    assert w_scales.dtype == out_dtype, f"{w_scales.dtype} does not match {out_dtype}"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    # TODO(before land): add test case for input with bsz
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)
    y_dot_int32 = y_dot_int32.reshape(*x_vals_int8.shape[:-1], -1)

    # TODO(future): consider using integer arithmetic throughout, although
    # TBD if that is actually faster on GPUs
    # need to use 32 bits here to prevent overflow for large shapes,
    # 16 bits is not enough
    y_dot_float32 = y_dot_int32.to(torch.float32)

    #
    # 2. connect it all together
    #

    # mm_unscaled has to stay in float32 for the next two lines to prevent overflow
    mm_unscaled_float32 = y_dot_float32 - (x_zp * w_vals_int8_t_sums_int64)
    y = x_scale * w_scales * mm_unscaled_float32
    # can downcast only at the very end
    y = y.to(out_dtype)
    return y


# pyre-fixme[3]: Return type must be annotated.
def quant_int8_dynamic_per_token_linear(
    # pyre-fixme[2]: Parameter must be annotated.
    x,
    # pyre-fixme[2]: Parameter must be annotated.
    w_vals_int8_t,
    # pyre-fixme[2]: Parameter must be annotated.
    w_scales,
    # pyre-fixme[2]: Parameter must be annotated.
    bias,
    # pyre-fixme[2]: Parameter must be annotated.
    out_dtype,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = quant_int8_per_token_matmul(
        x_vals_int8, x_scales, w_vals_int8_t, w_scales, out_dtype
    )
    if bias is not None:
        mm_out += bias
    return mm_out


# pyre-fixme[3]: Return type must be annotated.
def quant_int8_per_token_matmul(
    # pyre-fixme[2]: Parameter must be annotated.
    x_vals_int8,
    # pyre-fixme[2]: Parameter must be annotated.
    x_scales,
    # pyre-fixme[2]: Parameter must be annotated.
    w_vals_int8_t,
    # pyre-fixme[2]: Parameter must be annotated.
    w_scales,
    # pyre-fixme[2]: Parameter must be annotated.
    output_dtype=torch.float32,
):
    # Quantized matmul of int8 operands that accumulates to int32 and returns
    # output_dtype. For now, this is written for approximate numerical
    # Assumes that activation and weight quantization are symmetric,
    # i.e. act_zp and w_zp is 0.
    # Assumes that weight quantization is per-channel.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming output_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    #

    assert (
        x_vals_int8.dtype == torch.int8
    ), f"x dtype {x_vals_int8.dtype} not yet supported"
    assert (
        w_vals_int8_t.dtype == torch.int8
    ), f"w dtype {w_vals_int8_t.dtype} not yet supported"

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    y_dot_int32 = safe_int_mm(tmp, w_vals_int8_t)

    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    y = (y_dot_int32 * x_scales.reshape(-1, 1) * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_int32.shape[-1]
    )

    # can downcast only at the very end
    y = y.to(output_dtype)
    return y


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_groupwise_affine_qparams(w, n_bit=4, groupsize=128):
    """ """
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def pack_tinygemm_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def unpack_tinygemm_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


# pyre-fixme[3]: Return type must be annotated.
def groupwise_affine_quantize_tensor_from_qparams(
    # pyre-fixme[2]: Parameter must be annotated.
    w, scales, zeros, n_bit=4, groupsize=128
):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int4x8 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int4x8


# pyre-fixme[3]: Return type must be annotated.
def groupwise_affine_dequantize_tensor_from_qparams(
    # pyre-fixme[2]: Parameter must be annotated.
    w_int4x8, scales, zeros, n_bit=4, groupsize=128
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int4x8.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int4x8.shape[-1]
    assert w_int4x8.shape[-1] % groupsize == 0
    assert w_int4x8.dim() == 2

    w_int4x8_grouped = w_int4x8.reshape(-1, groupsize)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    w_dq = (
        w_int4x8_grouped.sub(2 ** (n_bit - 1))
        .mul(scales)
        .add(zeros)
        .reshape_as(w_int4x8)
    )
    return w_dq


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_groupwise_affine_qparams(w, n_bit, groupsize)
    w_int4x8 = groupwise_affine_quantize_tensor_from_qparams(
        w, scales, zeros, n_bit, groupsize
    )
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    return w_int4x8, scales_and_zeros


# pyre-fixme[3]: Return type must be annotated.
def groupwise_affine_dequantize_tensor(
    # pyre-fixme[2]: Parameter must be annotated.
    w_int4x8, scales_and_zeros, n_bit=4, groupsize=128
):
    scales, zeros = unpack_tinygemm_scales_and_zeros(scales_and_zeros)
    return groupwise_affine_dequantize_tensor_from_qparams(
        w_int4x8, scales, zeros, n_bit, groupsize
    )


# TODO: need some cleanups
# TODO: move this to https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """

    scales = input.abs().amax(dim=-1, keepdim=True)
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    if dtype == torch.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        raise Exception(f"unsupported dtype in choose_qparams_per_token: {dtype}")

    scales = scales.clamp(min=1e-5).div(quant_max)
    zero_points = torch.zeros_like(scales)
    return scales, zero_points


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = (1, input.size(-1))
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


# TODO: move this to https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # Based on https://github.com/google/XNNPACK/blob/df156f0cf3db5a4576cc711123eeb54915f82ffc/src/xnnpack/quantization.h#L18
    qmin, qmax = -128, 127
    min_val, max_val = torch.aminmax(input, dim=-1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    return scale.to(torch.float32), zero_point.to(torch.float32)


@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = (1, input.size(-1))
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    num_tokens = math.prod(list(input.size())[:-1])
    assert (
        num_tokens == scales.numel()
    ), f"num_tokens: {num_tokens} scales: {scales.size()}"
    assert (
        num_tokens == zero_points.numel()
    ), f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"


quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
# pyre-fixme[3]: Return type must be annotated.
def quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    input = (
        torch.round(input / scales + zero_points).clamp(quant_min, quant_max).to(dtype)
    )
    return input


@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
# pyre-fixme[3]: Return type must be annotated.
def quantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
# pyre-fixme[3]: Return type must be annotated.
def dequantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    """Per token dequantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): quantized Tensor (uint8, int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """
    input = input - zero_points
    input = input.to(output_dtype) * scales
    return input


@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
# pyre-fixme[3]: Return type must be annotated.
def dequantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: support fp16
    return torch.empty_like(input, dtype=output_dtype)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_group_qparams_symmetric(w, n_bit=4, groupsize=128, precision=torch.float32):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    max_val_abs = torch.max(-min_val_neg, max_val_pos)
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))

    scales = max_val_abs / (float(max_int - min_int) / 2)
    scales = torch.max(scales, torch.full_like(scales, torch.finfo(torch.float32).eps))
    # TODO: make sure abs(scales) is not too small?
    zeros = torch.full_like(scales, 0)
    return scales.to(precision).reshape(w.shape[0], -1), zeros.to(precision).reshape(
        w.shape[0], -1
    )


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def pack_scales_and_zeros(scales, zeros, precision=torch.float16):
    assert scales.shape == zeros.shape
    assert scales.dtype == precision
    assert zeros.dtype == precision
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# TODO: dtype is ignored for now
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
# pyre-fixme[3]: Return type must be annotated.
def quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    # pyre-fixme[2]: Parameter must be annotated.
    group_size=128,
):
    assert group_size > 1
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2

    # TODO: check for dtype, currently we can't express torch.int4 so it's omitted
    to_quant = input.reshape(-1, group_size)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    input_int8 = (
        to_quant.div(scales)
        .add(zero_points)
        .round()
        .clamp_(quant_min, quant_max)
        .to(dtype)
        .reshape_as(input)
    )

    return input_int8


@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
# pyre-fixme[3]: Return type must be annotated.
def quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    # pyre-fixme[2]: Parameter must be annotated.
    group_size=128,
):
    """Groupwise quantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    assert group_size > 1
    # needed for GPTQ single column quantize
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    return torch.empty_like(input, dtype=dtype)


# pyre-fixme[3]: Return type must be annotated.
def group_quantize_tensor_symmetric(
    # pyre-fixme[2]: Parameter must be annotated.
    w, n_bit=4, group_size=128, precision=torch.float32
):
    scales, zeros = get_group_qparams_symmetric(w, n_bit, group_size, precision)
    n_bit = 4
    max_int = 2 ** (n_bit - 1) - 1
    min_int = -(2 ** (n_bit - 1))
    # TODO: currently we don't know how to express torch.int4, we'll
    # add torch.int4 to core later
    w_int8 = torch.ops.quantized_decomposed.quantize_per_channel_group(
        w, scales, zeros, min_int, max_int, torch.int8, group_size
    )

    return w_int8, scales, zeros


quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
# pyre-fixme[3]: Return type must be annotated.
def dequantize_per_channel_group(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size: int = 128,
    output_dtype: torch.dtype = torch.float32,
):
    """Groupwise dequantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): quantized Tensor (uint8/int8 etc.)
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for input Tensor
       quant_max (int): maximum quantized value for input Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor
       output_dtype (torch.dtype): dtype (e.g. torch.float32) for output Tensor

    Returns:
       dequantized Tensor with dtype `output_dtype`
    """

    assert group_size > 1
    # needed for GPTQ single column dequantize
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    assert w_int8.shape[-1] % group_size == 0
    assert w_int8.dim() == 2

    w_int8_grouped = w_int8.reshape(-1, group_size)
    scales = scales.reshape(-1, 1)
    zp = zero_points.reshape(-1, 1) if zero_points is not None else 0
    w_dq = w_int8_grouped.sub(zp).mul(scales).reshape_as(w_int8).to(output_dtype)
    return w_dq


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def up_size(size):
    return (*size[:-1], size[-1] * 2)


quantized_decomposed_lib.define("pack_int4_from_int8(Tensor int8_data) -> Tensor")


@impl(quantized_decomposed_lib, "pack_int4_from_int8", "CompositeExplicitAutograd")
def pack_int4_from_int8(int8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
    shape = int8_data.shape
    assert shape[-1] % 2 == 0
    int8_data = int8_data.contiguous().view(-1)
    return (int8_data[::2] << 4 | int8_data[1::2]).view(down_size(shape))


quantized_decomposed_lib.define("unpack_int4_to_int8(Tensor int8_data) -> Tensor")


@impl(quantized_decomposed_lib, "unpack_int4_to_int8", "CompositeExplicitAutograd")
def unpack_int4_to_int8(int8_data: torch.Tensor) -> torch.Tensor:
    """Get the original weight from the normalized float weight format"""
    # since we are using int8 we will decode 2 entries per byte
    # Shift elements down 4 and select out the bottom 4 bits
    shape = int8_data.shape
    first_elements = (int8_data >> 4).to(torch.int8)
    second_elements = (int8_data & 0b1111).to(torch.int8)
    return torch.stack([first_elements, second_elements], dim=-1).view(up_size(shape))


def per_token_dynamic_quant(input: torch.Tensor) -> torch.Tensor:
    orig_dtype = input.dtype
    (
        scales,
        zero_points,
    ) = torch.ops.quantized_decomposed.choose_qparams_per_token(input, torch.int8)

    # TODO: get these from torch.int8
    quant_min = -128
    quant_max = 127
    input = torch.ops.quantized_decomposed.quantize_per_token(
        input, scales, zero_points, quant_min, quant_max, torch.int8
    )
    input = torch.ops.quantized_decomposed.dequantize_per_token(
        input, scales, zero_points, quant_min, quant_max, torch.int8, orig_dtype
    )
    return input
