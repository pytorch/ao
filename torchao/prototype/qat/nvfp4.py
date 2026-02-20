# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch

from torchao.prototype.mx_formats.config import QuantizeToNVFP4KernelChoice
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    _handle_use_triton_kernel,
    per_tensor_amax_to_scale,
)
from torchao.quantization.qat import FakeQuantizeConfigBase


@dataclass
class NVFP4FakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights or activations to NVIDIA's NVFP4 format
    according to https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/.

    Fake quantization numerics follow `NVFP4Tensor` closely: https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/nvfp4_tensor.py.

    Args:
        use_per_tensor_scale (bool): Whether to use two-level per-tensor fp32 scaling
            after the initial fp8 (e4m3) block-wise scaling (default True)
        use_swizzled_scales (bool): Whether scales are stored in swizzled (blocked) format
        quantize_to_nvfp4_kernel_choice (QuantizeToNVFP4KernelChoice): Kernel choice for quantize kernel
    """

    use_per_tensor_scale: bool = True
    use_swizzled_scales: bool = False
    quantize_to_nvfp4_kernel_choice: QuantizeToNVFP4KernelChoice = (
        QuantizeToNVFP4KernelChoice.TORCH
    )
    use_triton_kernel: bool = False

    def __post_init__(self):
        self.quantize_to_nvfp4_kernel_choice = _handle_use_triton_kernel(
            self.use_triton_kernel, self.quantize_to_nvfp4_kernel_choice
        )


# TODO: support emulation on non-Blackwell GPUs
class _NVFP4QuantizedForwardFakeQuantizedBackward(torch.autograd.Function):
    """
    Autograd function for NVFP4 quantization + addmm in low precision during forward,
    and fake quantization in high precision during backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation_config: NVFP4FakeQuantizeConfig,
        weight_config: NVFP4FakeQuantizeConfig,
    ) -> torch.Tensor:
        # quantize input activations
        if activation_config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(_input))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None
        _input = NVFP4Tensor.to_nvfp4(
            _input,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=activation_config.use_swizzled_scales,
            quantize_to_nvfp4_kernel_choice=activation_config.quantize_to_nvfp4_kernel_choice,
        )

        # quantize weights
        if weight_config.use_per_tensor_scale:
            tensor_amax = torch.max(torch.abs(weight))
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = None
        weight = NVFP4Tensor.to_nvfp4(
            weight,
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=weight_config.use_swizzled_scales,
            quantize_to_nvfp4_kernel_choice=QuantizeToNVFP4KernelChoice.TORCH,
        )

        # Follow `NVFP4DynamicActivationNVFP4WeightConfig`, always use traditional construction
        # for weights and set `quantize_to_nvfp4_kernel_choice` afterwards
        weight.quantize_to_nvfp4_kernel_choice = (
            weight_config.quantize_to_nvfp4_kernel_choice
        )

        ctx.save_for_backward(_input, weight)

        return _addmm_nvfp4_dispatch(
            _input,
            weight.t(),
            None,  # aten_op, not used
            bias,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        _input, weight = ctx.saved_tensors
        assert isinstance(_input, NVFP4Tensor)
        assert isinstance(weight, NVFP4Tensor)
        _input = _input.dequantize(_input.orig_dtype)
        weight = weight.dequantize(weight.orig_dtype)
        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), _input)
        return grad_input, grad_weight, None, None, None


class NVFP4FakeQuantizedLinear(torch.nn.Linear):
    """
    Linear module for fake quantized NVFP4 weights and/or activations.

    The forward pass follows quantization and addmm numerics in `NVFP4Tensor`
    in lower precision exactly, while the backward pass uses dequantize
    (fake quantized) values in high precision.

    Currently this is only applicable on Blackwell and future generations.
    See https://github.com/pytorch/ao/issues/3102 for more details.

    Example usage::

        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        base_config = NVFP4DynamicActivationNVFP4WeightConfig()
        quantize_(model, QATConfig(base_config, step="prepare"))
        # Model contains `NVFP4FakeQuantizedLinear` now

        train_loop(model)
        quantize_(model, QATConfig(base_config, step="convert"))
        # Model contains `nn.Linear` with `NVFP4Tensor` weights now
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[NVFP4FakeQuantizeConfig] = None,
        weight_config: Optional[NVFP4FakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            *args,
            **kwargs,
        )
        if weight_config is None:
            raise ValueError("Must specify `weight_config`")
        if activation_config is None:
            raise ValueError("Weight only NVFP4 QAT not supported yet")
        self.activation_config = activation_config
        self.weight_config = weight_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.view(-1, x.shape[-1])
        else:
            batch_size = None
        fq = _NVFP4QuantizedForwardFakeQuantizedBackward.apply(
            x, self.weight, self.bias, self.activation_config, self.weight_config
        )
        assert fq.dtype == x.dtype
        if batch_size is not None:
            return fq.view(batch_size, -1, fq.shape[-1])
        else:
            return fq

    def to_linear(self) -> torch.nn.Linear:
        new_linear = torch.nn.Linear(
            self.in_features,
            self.out_features,
            self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if self.weight.device != torch.device("meta"):
            new_linear.weight = self.weight
            new_linear.bias = self.bias
        return new_linear

    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: Optional[NVFP4FakeQuantizeConfig] = None,
        weight_config: Optional[NVFP4FakeQuantizeConfig] = None,
    ):
        new_linear = NVFP4FakeQuantizedLinear(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        # In distributed training, the model may be instantiated
        # on the meta device, in which case there is no need to
        # copy the weights, and doing so will result in an error
        if mod.weight.device != torch.device("meta"):
            new_linear.weight = mod.weight
            new_linear.bias = mod.bias
        return new_linear
