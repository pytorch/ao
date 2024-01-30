import torch
import torch.nn as nn
from typing import Tuple, Optional

from functools import partial

from torchao.quantization.quant_primitives import (
    dynamically_quantize_per_channel,
    quant_int8_dynamic_per_token_linear,
    quantize_activation_per_token_absmax
)
from torchao.quantization import quant_api
from torchao.sparsity import apply_fake_sparsity

# Quant + Sparse helper functinos
def sparse_quant_int8_dynamic_per_token_linear(
    x,
    w_vals_int8,
    w_scales,
    bias,
    out_dtype=torch.float32,
    fuse_dequant=True,
):
    # like F.linear, but with int8 dynamic quantization of activation,
    # and a quantized weight
    x_vals_int8, x_scales = quantize_activation_per_token_absmax(x)
    mm_out = sparse_quant_int8_per_token_matmul(
        x_vals_int8, x_scales, w_vals_int8, w_scales, out_dtype, fuse_dequant=fuse_dequant)
    if bias is not None:
        mm_out += bias
    return mm_out

def sparse_quant_int8_per_token_matmul(
    x_vals_int8,
    x_scales,
    w_vals_int8,
    w_scales,
    out_dtype=torch.float32,
    fuse_dequant=True,
):
    # Quantized sparse matmul of int8 operands that accumulates to fp16 and returns
    # out_dtype. This matmul uses cuSPARSELt as a backend.

    # Assumes that activation and weight quantization are symmetric,
    # i.e. act_zp and w_zp is 0.
    # Assumes that weight quantization is per-channel.
    # NOTE: sparsity is only compatible with symmetric (zero-preserving) quantization techniques.

    # see
    # https://github.com/google/gemmlowp/blob/master/doc/quantization.md
    # for an overview of quantized matmul compute

    # in scalar form, assuming out_dtype is fp32 and zw == 0:
    #
    #   Y_i_j_fp32 = sx * sw dot(X_i, W_j)
    #

    assert x_vals_int8.dtype == torch.int8, \
        f'x dtype {x_vals_int8.dtype} not yet supported'
    assert w_vals_int8.dtype == torch.int8, \
        f'w dtype {w_vals_int8.dtype} not yet supported'
    assert w_scales.dtype == out_dtype, \
        f'{w_scales.dtype} does not match {out_dtype}'

    #
    # 1. do the matrix form of dot(X_i, W_j)
    #

    # For sparse matmul, we need one of the input operands to be transposed.
    # This is because cuSPARSELt only supports int8 matmul for specific formats:
    # https://docs.nvidia.com/cuda/cusparselt/functions.html#matmul-descriptor-functions
    # Because we currently only support the first input to the operand being sparse,
    # we cannot transpose w_vals_int8, so instead we transpose x_vals_int8.
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1]).contiguous()
    # Since cuSPARSELt does not have support for int32 output, we instead use the fp16 kernel
    # instead, by setting out_dtype.
    # y_dot_fp16 = torch._sparse_semi_structured_linear(tmp, w_vals_int8, out_dtype=torch.float16)
    y_dot_fp16 = torch._cslt_sparse_mm(w_vals_int8, tmp.t(), out_dtype=torch.float16).t()
    y_dot_fp32 = y_dot_fp16.reshape(*x_vals_int8.shape[:-1], -1).to(out_dtype)

    #
    # 2. rescale the output
    #
    # in cases with large matrices, y_dot_int32 can grow sufficiently
    # large that y_dot_int32 * a float16 scale is greater than the maximum
    # value of a float 16, (which results in a value of inf even if multiplying
    # by the other scale would bring it within the expected range)

    # assert x_scales.dtype == torch.float, f"x_scales needs to be a torch.float32 but got {x_scales.dtype}"

    y = y_dot_fp32 * x_scales * w_scales

    # can downcast only at the very end
    y = y.to(out_dtype)
    return y

class SparseDynamicallyPerAxisQuantizedLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`, implementing sparse dynamic quantization on
    the input across all axes except for the last axis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__(in_features, out_features, bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the sparse quantized linear layer.

        This method applies dynamic quantization to the input tensor across all axes except
        the last axis using the `quant_int8_dynamic_per_token_linear` function.

        We artifically limit the quantization value to int4 range to ensure we stay within the range of fp16.
        This method will use cuSPASRELt to perform sparse matmul.

        Args:
            X (torch.Tensor): The input tensor to the sparse quantized linear layer.
        Returns:
            torch.Tensor: The output tensor after the sparse quantized matmul and rescale.
        """
        Y = sparse_quant_int8_dynamic_per_token_linear(
            X, self.W_int_repr, self.W_scales, self.bias, X.dtype, fuse_dequant=self.fuse_dequant)
        return Y

    @classmethod
    def from_float(cls, mod: torch.nn.Linear, fuse_dequant=True) -> 'SparseDynamicallyPerAxisQuantizedLinear':
        """
        Converts a `mod` of class `torch.nn.Linear` to the sparse dynamically quantized version of it.
        Note: this class does not require calibration.
        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.
        Returns:
            SparseDynamicallyPerAxisQuantizedLinear: The converted sparse quantized linear module.
        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias=mod.bias is not None)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        # NOTE: We artifically clamp the values to int4 quantization to ensure we stay within the
        # dynamic range of fp16
        W_int_repr, W_scales, _W_zps = dynamically_quantize_per_channel(
            mod.weight, -8, 7, torch.int8)
        new_mod.register_buffer('W_int_repr', torch._cslt_compress(W_int_repr.contiguous()))
        new_mod.register_buffer('W_scales', W_scales)
        new_mod.bias = mod.bias
        new_mod.fuse_dequant = fuse_dequant
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

def apply_int4_dynamic_quant_sparse(model, fuse_dequant=False):
    apply_fake_sparsity(model)
    quant_api._replace_with_custom_fn_if_matches_filter(
        model,
        partial(SparseDynamicallyPerAxisQuantizedLinear.from_float, fuse_dequant=fuse_dequant),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))
