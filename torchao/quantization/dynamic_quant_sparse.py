import torch
import torch.nn as nn
from typing import Tuple, Optional

from torchao.quantization.quant_primitives import (
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
    quantize_activation_per_token_absmax
)

from torch.sparse import SparseSemiStructuredTensor

# Quant + Sparse helper functinos

def sparse_quant_int8_dynamic_cutlass_linear(
    x,
    w_vals_int8,
    w_meta_int32,
    w_scales,
    bias,
    out_dtype,
):
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = sparse_quant_int8_cutlass_matmul(
        x_vals_int8, x_scales, w_vals_int8, w_meta_int32, w_scales, out_dtype)

    if bias is not None:
        mm_out += bias
    return mm_out

def sparse_quant_int8_dynamic_cslt_linear(
    x,
    w_vals_int8,
    w_scales,
    bias,
    out_dtype,
):
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = sparse_quant_int8_cslt_matmul(
        x_vals_int8, x_scales, w_vals_int8, w_scales, out_dtype)

    if bias is not None:
        mm_out += bias
    return mm_out


def sparse_quant_int8_cslt_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8,
    w_scales,
    out_dtype,
):

    assert x_vals_int8.dtype == torch.int8, f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8.dtype == torch.int8, f'w dtype {w_vals_int8.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, f'{w_scales.dtype} does not match {out_dtype}'

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).contiguous()

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    y_dot_bf16_w_scales_fused = torch._cslt_sparse_mm(w_vals_int8, tmp.t(), alpha=w_scales, out_dtype=torch.bfloat16).t()
    y = (y_dot_bf16_w_scales_fused* x_scales.reshape(-1, 1)).reshape(
        *x_vals_int8.shape[:-1], y_dot_bf16_w_scales_fused.shape[-1]
    )
    y = y.to(out_dtype)
    return y

def sparse_quant_int8_cutlass_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8,
    w_meta_int32,
    w_scales,
    out_dtype,
):
    assert x_vals_int8.dtype == torch.int8, f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8.dtype == torch.int8, f'w dtype {w_vals_int8.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, f'{w_scales.dtype} does not match {out_dtype}'
    assert w_meta_int32.dtype == torch.int32, f'{w_meta_int32.dtype} not yet supported'

    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).contiguous()

    assert x_scales.dtype in [
        torch.float,
        torch.bfloat16,
    ], f"x_scales needs to be a torch.float32 or torch.bfloat16 but got {x_scales.dtype}"

    y_dot_int32 = torch._sparse_semi_structured_linear(tmp, w_vals_int8, w_meta_int32.view(torch.int32), out_dtype=torch.int32)
    y = (y_dot_int32 * x_scales.reshape(-1, 1) * w_scales).reshape(
        *x_vals_int8.shape[:-1], y_dot_int32.shape[-1]
    )
    y = y.to(out_dtype)
    return y
