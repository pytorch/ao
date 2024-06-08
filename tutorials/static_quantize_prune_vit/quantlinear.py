import torch
import torch
import torch.nn as nn
from typing import Tuple, Optional

import torch
from torch._dynamo import is_compiling as dynamo_is_compiling
from torchao.quantization.quant_primitives import quant_int8_per_token_matmul
from torchao.kernel.intmm import int_scaled_2_bias_matmul

def quantize_activation_per_token(t, scales):
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t

def quantize_activation_per_token_absmax(t, scales): # absmax):
    # n_bits = 8
    # # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
    # # want float scales to avoid overflows
    # scales = absmax.float()
    # q_max = 2 ** (n_bits - 1) - 1
    # scales.clamp_(min=1e-5).div_(q_max)

    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    t = torch.round(t / scales).clamp(-127, 127).to(torch.int8)
    return t, scales

def quant_int8_dynamic_per_token_linear(
    x_vals_int8,
    x_scales,
    w_vals_int8_t,
    w_scales,
    bias,
    out_dtype,
):
    assert x_scales.dtype == out_dtype
    assert w_scales.dtype == out_dtype
    tmp = x_vals_int8.reshape(-1, x_vals_int8.shape[-1])
    res = int_scaled_2_bias_matmul(tmp,
                                   w_vals_int8_t,
                                   x_scales.reshape(-1, 1),
                                   w_scales.reshape(1, -1),
                                   out_dtype,
                                   bias)
    return res.reshape(*x_vals_int8.shape[:-1], -1)

class StaticallyPerAxisQuantizedLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`, implementing static quantization on
    the input across all axes except for the last axis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__(in_features, out_features, bias)
        self.calibration_limit = 10
        self.x_absmax_tmp = None
        self.x_absmax = None
        self.calibration_count = 0
        # torch._dynamo.disable(self.set_x_absmax)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO This kind of dynamism doesn't seem possible with dynamo under max-autotune
        # But it does seem to work under max-autotune-no-cudagraphs
        if self.calibration_count < self.calibration_limit:
            self.set_x_absmax(X)
        self.calibration_count += 1
        x_vals_int8, x_scales = quantize_activation_per_token_absmax(X, self.x_absmax)
        Y = quant_int8_dynamic_per_token_linear(x_vals_int8, x_scales, self.W_int_repr_t, self.W_scales, self.bias, X.dtype)
        return Y

    def forward_static(self, X: torch.Tensor) -> torch.Tensor:
        x_vals_int8, x_scales = quantize_activation_per_token_absmax(X, self.x_absmax)
        Y = quant_int8_dynamic_per_token_linear(x_vals_int8, x_scales, self.W_int_repr_t, self.W_scales, self.bias, X.dtype)
        return Y

    def set_x_absmax(self, X):
        x_absmax = X.abs().amax(dim=-1, keepdim=True)
        if self.x_absmax is None:
            self.x_absmax_tmp = x_absmax
            n_bits = 8
            # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
            # want float scales to avoid overflows
            self.x_absmax = self.x_absmax_tmp.bfloat16()
            q_max = 2 ** (n_bits - 1) - 1
            eps = torch.finfo(X.dtype).eps
            self.x_absmax.clamp_(min=eps).div_(q_max)
        else:
            self.x_absmax_tmp = torch.maximum(self.x_absmax_tmp, X.abs().amax(dim=-1, keepdim=True))
            n_bits = 8
            # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
            # want float scales to avoid overflows
            self.x_absmax = self.x_absmax_tmp.bfloat16()
            q_max = 2 ** (n_bits - 1) - 1
            eps = torch.finfo(X.dtype).eps
            self.x_absmax.clamp_(min=eps).div_(q_max)

    @classmethod
    def freeze(cls, mod: torch.nn.Linear) -> 'StaticallyPerAxisQuantizedLinear':
        mod.forward = mod.forward_static
        return mod

    @classmethod
    def from_float(cls, mod: torch.nn.Linear) -> 'StaticallyPerAxisQuantizedLinear':
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
            zero_point = torch.zeros(
                min_val_neg.size(), dtype=torch.int64, device=device)

            # quantize based on qmin/qmax/scale/zp
            # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
            x_div = x.transpose(0, 1) / scale
            x_round = torch.round(x_div)
            x_zp = x_round + zero_point
            x_zp = x_zp.transpose(0, 1)
            quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

            return quant, scale, zero_point
        """
        Converts a `mod` of class `torch.nn.Linear` to the dynamically quantized version of it.

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            StaticallyPerAxisQuantizedLinear: The converted quantized linear module.

        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias=mod.bias is not None)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        W_int_repr, W_scales, _W_zps = dynamically_quantize_per_channel(
            mod.weight, -128, 127, torch.int8)
        new_mod.register_buffer('W_int_repr_t', W_int_repr.contiguous().t())
        new_mod.W_scales = nn.Parameter(W_scales)
        new_mod.bias = mod.bias
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

def get_x_absmax(model):
    result = []
    for name, mod in model.named_modules():
        if isinstance(mod, StaticallyPerAxisQuantizedLinear):
            result.append((name, mod.x_absmax))
    return result

def set_x_absmax(model, weights):
    i = 0
    for name, mod in model.named_modules():
        if isinstance(mod, StaticallyPerAxisQuantizedLinear):
            assert weights[i][0][-len(name):] == name, f"{weights[i][0][-len(name):]} / {name}"
            mod.x_absmax = weights[i][1]
            mod.forward = mod.forward_static
            i += 1

def apply_static_quant(model):
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    _replace_with_custom_fn_if_matches_filter(
        model,
        StaticallyPerAxisQuantizedLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))

def freeze_static_quant(model):
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    _replace_with_custom_fn_if_matches_filter(
        model,
        StaticallyPerAxisQuantizedLinear.freeze,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))
