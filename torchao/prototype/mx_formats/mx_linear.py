# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the prototype UX for converting a model to use mx weights
"""

from typing import Any, Optional

import torch

from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    MXFP8Dim1CastKernelChoice,
    MXLinearConfig,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


# convenience wrapper
def _to_mxfp8_then_scaled_mm(
    input_hp: torch.Tensor,
    weight_hp: torch.Tensor,
    kernel_preference: KernelPreference,
    scale_calculation_mode: ScaleCalculationMode,
    wgrad_with_hp: bool = False,
) -> torch.Tensor:
    """
    Performs a matrix multiplication with MXFP8 quantization on both forward and backward passes.

    This function wraps the `mx_mm` autograd function to provide differentiable MXFP8
    matrix multiplication. It dynamically quantizes activations, weights, and gradients
    to MXFP8 format for each matmul operation:

    - Forward: input @ weight_t = output (both quantized to MXFP8)
    - Backward: grad_output @ weight = grad_input (both quantized to MXFP8)
    - Backward: input_t @ grad_output = grad_weight (quantized to MXFP8 unless wgrad_with_hp=True)

    Args:
        input_hp: High precision input tensor of shape [..., in_features]
        weight_hp: High precision weight tensor of shape [out_features, in_features]
        kernel_preference: Whether to use AUTO (best kernel for each operation) or EMULATED mode
        scale_calculation_mode: Scale calculation method (RCEIL or FLOOR) for MXFP8 quantization
        wgrad_with_hp: If True, compute grad_weight in high precision instead of MXFP8. Default: False

    Returns:
        Output tensor of shape [..., out_features] in high precision

    Note:
        Forward and backward grad_input are always computed using MXFP8 with block_size=32
        and element_dtype=float8_e4m3fn. Backward grad_weight uses MXFP8 by default, but can
        optionally use high precision when wgrad_with_hp=True for improved accuracy.
        The Triton kernel is used for dim0 quantization and CUDA kernel for dim1 quantization.
    """
    in_elem_dtype = torch.float8_e4m3fn
    w_elem_dtype = torch.float8_e4m3fn
    grad_elem_dtype = torch.float8_e4m3fn
    block_size = 32
    mxfp8_dim0_cast_kernel_choice = MXFP8Dim0CastKernelChoice.TRITON
    mxfp8_dim1_cast_kernel_choice = MXFP8Dim1CastKernelChoice.CUDA

    return mx_mm.apply(
        input_hp,
        weight_hp,
        in_elem_dtype,
        w_elem_dtype,
        grad_elem_dtype,
        block_size,
        kernel_preference,
        mxfp8_dim0_cast_kernel_choice,
        mxfp8_dim1_cast_kernel_choice,
        scale_calculation_mode,
        wgrad_with_hp,
    )


@torch._dynamo.allow_in_graph
class mx_mm(torch.autograd.Function):
    # There are three gemms in a forward + backward of a Linear layer:
    #
    # 1.       input @ weight_t    = output     (forward pass)
    # 2. grad_output @ weight      = grad_input (backward pass)
    # 3.     input_t @ grad_output = grad_weight (backward pass)
    #
    # input, weight and grad_output can have each their own MX element dtype.

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        in_elem_dtype: Any,
        w_elem_dtype: Any,
        grad_elem_dtype: Any,
        block_size: int,
        kernel_preference: KernelPreference,
        mxfp8_dim0_cast_kernel_choice: MXFP8Dim0CastKernelChoice,
        mxfp8_dim1_cast_kernel_choice: MXFP8Dim1CastKernelChoice,
        scale_calculation_mode: ScaleCalculationMode,
        wgrad_with_hp: bool,
    ):
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.kernel_preference = kernel_preference
        ctx.wgrad_with_hp = wgrad_with_hp
        ctx.mxfp8_dim0_cast_kernel_choice = mxfp8_dim0_cast_kernel_choice
        ctx.mxfp8_dim1_cast_kernel_choice = mxfp8_dim1_cast_kernel_choice
        ctx.scale_calculation_mode = scale_calculation_mode

        # input @ weight_t = output
        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])

        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r,
            in_elem_dtype,
            block_size,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
        )
        weight_mx_dim0 = MXTensor.to_mx(
            weight_hp,
            w_elem_dtype,
            block_size,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
        )
        output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])

        return output

    @staticmethod
    def backward(ctx, grad_output_hp: torch.Tensor):
        input_hp, weight_hp = ctx.saved_tensors
        in_elem_dtype = ctx.in_elem_dtype
        w_elem_dtype = ctx.w_elem_dtype
        grad_elem_dtype = ctx.grad_elem_dtype
        block_size = ctx.block_size
        kernel_preference = ctx.kernel_preference
        mxfp8_dim0_cast_kernel_choice = ctx.mxfp8_dim0_cast_kernel_choice
        mxfp8_dim1_cast_kernel_choice = ctx.mxfp8_dim1_cast_kernel_choice
        scale_calculation_mode = ctx.scale_calculation_mode
        wgrad_with_hp = ctx.wgrad_with_hp

        grad_output_orig_shape = grad_output_hp.shape
        grad_output_hp_r = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

        input_hp_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_hp_orig_shape[-1])

        # grad_output @ weight = grad_input
        grad_output_mx_dim0 = MXTensor.to_mx(
            grad_output_hp_r,
            grad_elem_dtype,
            block_size,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
        )

        if mxfp8_dim1_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                weight_hp,
                block_size,
                w_elem_dtype,
                weight_hp.dtype,
                kernel_preference,
                mxfp8_dim1_cast_kernel_choice,
                scale_calculation_mode,
            )
        else:
            weight_hp_t_c = weight_hp.t().contiguous()
            weight_mx_dim1 = MXTensor.to_mx(
                weight_hp_t_c,
                w_elem_dtype,
                block_size,
                kernel_preference=kernel_preference,
                scaling_mode=scale_calculation_mode,
                mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
            )
        grad_input = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        # input_t @ grad_output = grad_weight
        if wgrad_with_hp:
            # Compute grad_weight in high precision if wgrad_with_hp is True
            grad_weight = torch.mm(grad_output_hp_r.t(), input_hp_r)
        else:
            # Compute grad_weight with MXFP8 quantization
            if mxfp8_dim1_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
                grad_output_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                    grad_output_hp_r,
                    block_size,
                    grad_elem_dtype,
                    grad_output_hp_r.dtype,
                    kernel_preference,
                    mxfp8_dim1_cast_kernel_choice,
                    scale_calculation_mode,
                )
            else:
                grad_output_mx_dim1 = MXTensor.to_mx(
                    grad_output_hp_r.t().contiguous(),
                    grad_elem_dtype,
                    block_size,
                    kernel_preference=kernel_preference,
                    scaling_mode=scale_calculation_mode,
                    mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
                )

            if mxfp8_dim1_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
                input_t_mx_dim0_tmp = _to_mxfp8_dim1_kernel_wrapper(
                    input_hp_r,
                    block_size,
                    in_elem_dtype,
                    input_hp_r.dtype,
                    kernel_preference,
                    mxfp8_dim1_cast_kernel_choice,
                    scale_calculation_mode,
                )
                input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
            else:
                input_t_mx_dim0_tmp = MXTensor.to_mx(
                    input_hp_r.t().contiguous(),
                    in_elem_dtype,
                    block_size,
                    kernel_preference=kernel_preference,
                    scaling_mode=scale_calculation_mode,
                    mxfp8_dim0_cast_kernel_choice=mxfp8_dim0_cast_kernel_choice,
                )
                input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
            grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MXLinear(torch.nn.Linear):
    """
    Linear layer with the compute happening in emulate MX. Currently the MX
    matmul is emulated since there is no hardware support yet. Activations,
    weights and grads are casted to MX and back to high precision for each
    matmul.

    Input, weight and grad_output can have each their own MX element dtype.
    """

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: Optional[MXLinearConfig] = MXLinearConfig(),
    ):
        assert isinstance(mod, torch.nn.Linear), f"unsupported type(mod) {type(mod)}"
        assert isinstance(config, MXLinearConfig)
        mod.__class__ = MXLinear
        mod.config = config
        return mod

    def forward(self, x):
        if torch.is_autocast_enabled():
            # special case autocast
            autocast_dtype = torch.get_autocast_dtype("cuda")
            x = x.to(autocast_dtype)
            w = self.weight.to(autocast_dtype)
        else:
            w = self.weight

        config = self.config
        wgrad_with_hp = False
        y = mx_mm.apply(
            x,
            w,
            config.elem_dtype,
            config.elem_dtype_weight_override or config.elem_dtype,
            config.elem_dtype_grad_output_override or config.elem_dtype,
            config.block_size,
            config.kernel_preference,
            config.mxfp8_dim0_cast_kernel_choice,
            config.mxfp8_dim1_cast_kernel_choice,
            config.scale_calculation_mode,
            wgrad_with_hp,  # temporary, for tests to pass pending deletion of MXLinear
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        s = f"{super().extra_repr()}, {self.config.short_str()}"
        return s


@register_quantize_module_handler(MXLinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXLinearConfig):
    return MXLinear.from_float(module, config=config)
