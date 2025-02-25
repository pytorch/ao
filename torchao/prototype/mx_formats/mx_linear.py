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

from torchao.prototype.mx_formats.config import MXGemmKernelChoice, MXLinearConfig
from torchao.prototype.mx_formats.mx_tensor import MXTensor


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
    ):
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.in_elem_dtype = in_elem_dtype
        ctx.w_elem_dtype = w_elem_dtype
        ctx.grad_elem_dtype = grad_elem_dtype
        ctx.block_size = block_size
        ctx.gemm_kernel_choice = gemm_kernel_choice

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
        weight_hp_t_c = weight_hp.t().contiguous()
        in_elem_dtype = ctx.in_elem_dtype
        w_elem_dtype = ctx.w_elem_dtype
        grad_elem_dtype = ctx.grad_elem_dtype
        block_size = ctx.block_size
        gemm_kernel_choice = ctx.gemm_kernel_choice

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
        grad_output_mx_dim1 = MXTensor.to_mx(
            grad_output_hp_r.t().contiguous(),
            grad_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        input_t_mx_dim0_tmp = MXTensor.to_mx(
            input_hp_r.t().contiguous(),
            in_elem_dtype,
            block_size,
            gemm_kernel_choice=gemm_kernel_choice,
        )
        input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
        grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)

        return grad_input, grad_weight, None, None, None, None, None


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
        # TODO(before land): remove this
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
        )
        if self.bias is not None:
            y = y + self.bias
        return y


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
        config: Optional[MXLinearConfig] = MXLinearConfig(),
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
        )
        new_mod.bias = mod.bias
        new_mod.config = config
        return new_mod

    @torch.no_grad()
    def forward(self, x):
        w_hp = self.weight_mx.to_dtype(x.dtype)
        y = F.linear(x, w_hp, self.bias)
        return y


def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=""
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn
            )


def _is_linear(mod, fqn):
    return isinstance(mod, torch.nn.Linear)


def swap_linear_with_mx_linear(
    model,
    *,
    config: Optional[MXLinearConfig] = None,
    filter_fn=None,
):
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXLinear.from_float(mod, config=config),
        combined_filter_fn,
    )


def swap_linear_with_mx_inference_linear(
    model,
    *,
    config: Optional[MXLinearConfig] = None,
    filter_fn=None,
):
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXInferenceLinear.from_float(mod, config=config),
        combined_filter_fn,
    )
