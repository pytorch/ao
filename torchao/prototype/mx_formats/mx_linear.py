# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines the prototype UX for converting a model to use mx weights
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor

from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    MXGemmKernelChoice,
    MXInferenceLinearConfig,
    MXLinearConfig,
)
from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_to_mxfp8_dim1,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


def _to_mxfp8_dim1_kernel_wrapper(
    a,
    block_size,
    elem_dtype,
    hp_dtype,
    gemm_kernel_choice,
    cast_kernel_choice,
):
    if cast_kernel_choice == MXFP8Dim1CastKernelChoice.TRITON:
        a_data, a_scale = triton_to_mxfp8_dim1(a, block_size)
    elif cast_kernel_choice == MXFP8Dim1CastKernelChoice.CUDA:
        _, a_data, _, a_scale = mxfp8_quantize_cuda(
            a,
            rowwise=False,
            colwise=True,
            scaling_mode="floor",
        )
    else:
        raise ValueError(f"must be one of [CUDA, TRITON], got {cast_kernel_choice}")

    if isinstance(a_data, DTensor):
        assert isinstance(a_scale, DTensor)
        a_data_local = a_data.to_local()
        a_scale_local = a_scale.to_local()
        inner = MXTensor(
            a_scale_local,
            a_data_local.t(),
            elem_dtype,
            block_size,
            hp_dtype,
            False,
            gemm_kernel_choice,
            False,
        )
        mx_tensor = DTensor.from_local(
            inner,
            a_data.device_mesh,
            a_data.placements,
            run_check=False,
            shape=a_data.t().size(),
            stride=a_data.t().stride(),
        )
    else:
        mx_tensor = MXTensor(
            a_scale,
            a_data.t(),
            elem_dtype,
            block_size,
            hp_dtype,
            False,
            gemm_kernel_choice,
            False,
        )
    return mx_tensor


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
        gemm_kernel_choice: MXGemmKernelChoice,
        mxfp8_cast_kernel_choice: MXFP8Dim1CastKernelChoice,
    ):
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.gemm_kernel_choice = gemm_kernel_choice
        ctx.mxfp8_cast_kernel_choice = mxfp8_cast_kernel_choice

        # input @ weight_t = output
        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])

        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r, in_elem_dtype, block_size, gemm_kernel_choice=gemm_kernel_choice
        )
        weight_mx_dim0 = MXTensor.to_mx(
            weight_hp, w_elem_dtype, block_size, gemm_kernel_choice=gemm_kernel_choice
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
        gemm_kernel_choice = ctx.gemm_kernel_choice
        mxfp8_cast_kernel_choice = ctx.mxfp8_cast_kernel_choice

        grad_output_orig_shape = grad_output_hp.shape
        grad_output_hp_r = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

        input_hp_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_hp_orig_shape[-1])

        # grad_output @ weight = grad_input
        grad_output_mx_dim0 = MXTensor.to_mx(
            grad_output_hp_r,
            grad_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
        )

        if mxfp8_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                weight_hp,
                block_size,
                w_elem_dtype,
                weight_hp.dtype,
                gemm_kernel_choice,
                mxfp8_cast_kernel_choice,
            )
        else:
            weight_hp_t_c = weight_hp.t().contiguous()
            weight_mx_dim1 = MXTensor.to_mx(
                weight_hp_t_c,
                w_elem_dtype,
                block_size,
                gemm_kernel_choice=gemm_kernel_choice,
            )
        grad_input = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        # input_t @ grad_output = grad_weight
        if mxfp8_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            grad_output_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                grad_output_hp_r,
                block_size,
                grad_elem_dtype,
                grad_output_hp_r.dtype,
                gemm_kernel_choice,
                mxfp8_cast_kernel_choice,
            )
        else:
            grad_output_mx_dim1 = MXTensor.to_mx(
                grad_output_hp_r.t().contiguous(),
                grad_elem_dtype,
                block_size,
                gemm_kernel_choice=gemm_kernel_choice,
            )

        if mxfp8_cast_kernel_choice != MXFP8Dim1CastKernelChoice.TORCH:
            input_t_mx_dim0_tmp = _to_mxfp8_dim1_kernel_wrapper(
                input_hp_r,
                block_size,
                in_elem_dtype,
                input_hp_r.dtype,
                gemm_kernel_choice,
                mxfp8_cast_kernel_choice,
            )
            input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
        else:
            input_t_mx_dim0_tmp = MXTensor.to_mx(
                input_hp_r.t().contiguous(),
                in_elem_dtype,
                block_size,
                gemm_kernel_choice=gemm_kernel_choice,
            )
            input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
        grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)

        return grad_input, grad_weight, None, None, None, None, None, None


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
        y = mx_mm.apply(
            x,
            w,
            config.elem_dtype,
            config.elem_dtype_weight_override or config.elem_dtype,
            config.elem_dtype_grad_output_override or config.elem_dtype,
            config.block_size,
            config.gemm_kernel_choice,
            config.mxfp8_cast_kernel_choice,
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        s = f"{super().extra_repr()}, {self.config.short_str()}"
        return s


class MXInferenceLinear(torch.nn.Linear):
    """
    Inference version of MXLinear, with the weight pre-quantized to MX.

    Note: this is weight-only quantization, with the gemm being executed
    in high precision.
    """

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: Optional[MXInferenceLinearConfig] = MXInferenceLinearConfig(),
    ):
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(**super_kwargs)
        # TODO(future PR): set to new_mod.weight directly, will need to work
        # through some errors
        new_mod.weight_mx = MXTensor.to_mx(
            mod.weight,
            config.elem_dtype,
            block_size=config.block_size,
            gemm_kernel_choice=config.gemm_kernel_choice,
            pack_fp6=config.pack_fp6,
        )
        new_mod.bias = mod.bias
        new_mod.config = config
        return new_mod

    @torch.no_grad()
    def forward(self, x):
        w_hp = self.weight_mx.to_dtype(x.dtype)
        y = F.linear(x, w_hp, self.bias)
        return y

    def extra_repr(self):
        s = f"{super().extra_repr()}, {self.config.short_str()}"
        return s


@register_quantize_module_handler(MXLinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXLinearConfig):
    return MXLinear.from_float(module, config=config)


@register_quantize_module_handler(MXInferenceLinearConfig)
def _mx_inference_linear_transform(
    module: torch.nn.Module, config: MXInferenceLinearConfig
):
    return MXInferenceLinear.from_float(module, config=config)
