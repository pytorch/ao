# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MX (Microscaling) Quantization-Aware Training (QAT) support.

This module provides QAT support for the OCP Microscaling MX formats (MXFP4, MXFP8).

Key differences between MX and NVFP4:
- Block size: MX uses 32 (default), NVFP4 uses 16 (fixed)
- Scale type: MX uses E8M0 (float8_e8m0fnu), NVFP4 uses float8_e4m3fn
- NVFP4 performs an extra per-tensor scaling, while MX does not
- Scale calculation: MX supports FLOOR, RCEIL, CEIL, EVEN modes
- MX supports multiple element dtypes:
  - MXFP4: torch.float4_e2m1fn_x2 (requires PyTorch 2.8+)
  - MXFP8: torch.float8_e4m3fn, torch.float8_e5m2
"""

from dataclasses import dataclass
from typing import Optional

import torch

from torchao.prototype.mx_formats.config import (
    ScaleCalculationMode,
    _validate_elem_dtype,
    _validate_kernel_preference,
)
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    _addmm_mx_dispatch,
)
from torchao.quantization.qat import FakeQuantizeConfigBase
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

_DEFAULT_MX_DTYPE = torch.float4_e2m1fn_x2


@dataclass
class MXFakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights or activations to the OCP Microscaling MX format
    according to https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Fake quantization numerics follow `MXTensor` closely:
    https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py.

    Supported element dtypes:
    - MXFP4: torch.float4_e2m1fn_x2 (requires PyTorch 2.8+)
    - MXFP8: torch.float8_e4m3fn, torch.float8_e5m2

    Key differences from NVFP4:
    - Block size: 32 (default) vs NVFP4's fixed 16
    - Scale type: E8M0 (float8_e8m0fnu) vs NVFP4's float8_e4m3fn
    - NVFP4 performs an extra per-tensor scaling, while MX does not
    - Supports multiple scale calculation modes (FLOOR, RCEIL, CEIL, EVEN)

    Args:
        dtype (torch.dtype): The element dtype for quantization.
            Supported values: torch.float4_e2m1fn_x2 (requires PyTorch 2.8+),
            torch.float8_e4m3fn, torch.float8_e5m2.
            Default is float4_e2m1fn_x2 on PyTorch 2.8+, float8_e4m3fn otherwise.
        block_size (int): The block size for quantization (default 32, the OCP MX standard)
        scaling_mode (ScaleCalculationMode): How to calculate the block scales (default RCEIL)
        kernel_preference (KernelPreference): Which kernel to use for matmul (default EMULATED)
    """

    dtype: torch.dtype = _DEFAULT_MX_DTYPE
    block_size: int = 32
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.RCEIL
    kernel_preference: KernelPreference = KernelPreference.EMULATED

    def __post_init__(self):
        _validate_elem_dtype(self.dtype)
        _validate_kernel_preference(self.kernel_preference, self.block_size, self.dtype)


class _MXQuantizedForwardFakeQuantizedBackward(torch.autograd.Function):
    """
    Autograd function for MX quantization + addmm in low precision during forward,
    and fake quantization in high precision during backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation_config: MXFakeQuantizeConfig,
        weight_config: MXFakeQuantizeConfig,
    ) -> torch.Tensor:
        # Handle inputs of any rank by reshaping to 2D
        orig_shape = _input.shape
        _input_2d = _input.view(-1, orig_shape[-1])

        # quantize input activations
        _input_2d = MXTensor.to_mx(
            _input_2d,
            elem_dtype=activation_config.dtype,
            block_size=activation_config.block_size,
            scaling_mode=activation_config.scaling_mode,
            kernel_preference=activation_config.kernel_preference,
        )

        weight = MXTensor.to_mx(
            weight,
            elem_dtype=weight_config.dtype,
            block_size=weight_config.block_size,
            scaling_mode=weight_config.scaling_mode,
            kernel_preference=weight_config.kernel_preference,
        )

        ctx.save_for_backward(_input_2d, weight)
        ctx.orig_shape = orig_shape

        # Use addmm when bias is present, mm otherwise
        if bias is not None:
            aten_op = torch.ops.aten.addmm.default
        else:
            aten_op = torch.ops.aten.mm.default

        out = _addmm_mx_dispatch(
            _input_2d,
            weight.t(),
            aten_op,
            bias,
        )

        # Reshape output back to original shape (with last dim changed)
        out_shape = (*orig_shape[:-1], out.shape[-1])
        return out.view(*out_shape)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        _input_2d, weight = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        assert isinstance(_input_2d, MXTensor)
        assert isinstance(weight, MXTensor)
        _input_2d = _input_2d.dequantize(_input_2d.orig_dtype)
        weight = weight.dequantize(weight.orig_dtype)

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        grad_input_2d = torch.mm(grad_output_2d, weight)
        grad_weight = torch.mm(grad_output_2d.t(), _input_2d)

        grad_input = grad_input_2d.view(*orig_shape)
        grad_bias = grad_output_2d.sum(0) if ctx.needs_input_grad[2] else None
        return grad_input, grad_weight, grad_bias, None, None


class MXFakeQuantizedLinear(torch.nn.Linear):
    """
    Linear module for fake quantized MX weights and/or activations.

    The forward pass follows quantization and addmm numerics in `MXTensor`
    in lower precision exactly, while the backward pass uses dequantized
    (fake quantized) values in high precision.


    Example usage::

        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        from torchao.quantization.qat import QATConfig

        base_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float4_e2m1fn_x2,
            weight_dtype=torch.float4_e2m1fn_x2,
        )
        quantize_(model, QATConfig(base_config, step="prepare"))
        # Model contains `MXFakeQuantizedLinear` now

        train_loop(model)
        quantize_(model, QATConfig(base_config, step="convert"))
        # Model contains `nn.Linear` with `MXTensor` weights now
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[MXFakeQuantizeConfig] = None,
        weight_config: Optional[MXFakeQuantizeConfig] = None,
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
            raise ValueError("Weight only MX QAT not supported yet")
        self.activation_config = activation_config
        self.weight_config = weight_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fq = _MXQuantizedForwardFakeQuantizedBackward.apply(
            x, self.weight, self.bias, self.activation_config, self.weight_config
        )
        assert fq.dtype == x.dtype
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
        activation_config: Optional[MXFakeQuantizeConfig] = None,
        weight_config: Optional[MXFakeQuantizeConfig] = None,
    ):
        new_linear = MXFakeQuantizedLinear(
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


class MXFakeQuantizedConv2d(torch.nn.Conv2d):
    """Conv2d module with fake-quantized MX weights and activations.

    Activations and weights are fake quantized in blocks along their input
    channel dimension. The quantized values are dequantized before the
    convolution, so the convolution itself runs in the original high-precision
    dtype. As in :class:`MXFakeQuantizedLinear`, the backward pass uses a
    straight-through estimator.

    The input channels and the input channels per group must be divisible by
    their respective block sizes because ``MXTensor`` does not pad incomplete
    blocks. The high-precision activation and weight dtypes must be
    ``torch.float32`` or ``torch.bfloat16``, matching ``MXTensor`` support.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation_config: Optional[MXFakeQuantizeConfig] = None,
        weight_config: Optional[MXFakeQuantizeConfig] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        if weight_config is None:
            raise ValueError("Must specify `weight_config`")
        if activation_config is None:
            raise ValueError("Weight only MX QAT not supported yet")
        if in_channels % activation_config.block_size != 0:
            raise ValueError(
                "Input channels must be divisible by the activation block size"
            )
        if (in_channels // groups) % weight_config.block_size != 0:
            raise ValueError(
                "Input channels per group must be divisible by the weight block size"
            )
        self.activation_config = activation_config
        self.weight_config = weight_config

    @staticmethod
    def _fake_quantize_input_channel_blocks(
        tensor: torch.Tensor,
        config: MXFakeQuantizeConfig,
    ) -> torch.Tensor:
        # Both Conv2d inputs (..., C, H, W) and weights (O, I, H, W) have
        # their input-channel dimension at -3. MXTensor quantizes its last
        # dimension, so move the channels there temporarily.
        tensor_channel_last = tensor.movedim(-3, -1).contiguous()
        tensor_channel_last = MXTensor.to_mx(
            tensor_channel_last,
            elem_dtype=config.dtype,
            block_size=config.block_size,
            scaling_mode=config.scaling_mode,
            kernel_preference=config.kernel_preference,
        ).dequantize(tensor.dtype)
        return tensor_channel_last.movedim(-1, -3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized_input = self._fake_quantize_input_channel_blocks(
            x, self.activation_config
        )
        quantized_weight = self._fake_quantize_input_channel_blocks(
            self.weight, self.weight_config
        )

        input_ste = x + (quantized_input - x).detach()
        weight_ste = self.weight + (quantized_weight - self.weight).detach()
        return self._conv_forward(input_ste, weight_ste, self.bias)

    def to_conv2d(self) -> torch.nn.Conv2d:
        new_conv = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        if self.weight.device != torch.device("meta"):
            new_conv.weight = self.weight
            new_conv.bias = self.bias
        return new_conv

    @classmethod
    def from_conv2d(
        cls,
        mod: torch.nn.Conv2d,
        activation_config: Optional[MXFakeQuantizeConfig] = None,
        weight_config: Optional[MXFakeQuantizeConfig] = None,
    ) -> "MXFakeQuantizedConv2d":
        new_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
            activation_config=activation_config,
            weight_config=weight_config,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        if mod.weight.device != torch.device("meta"):
            new_conv.weight = mod.weight
            new_conv.bias = mod.bias
        return new_conv
