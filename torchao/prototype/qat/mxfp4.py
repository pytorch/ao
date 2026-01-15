# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MXFP4 Quantization-Aware Training (QAT) support.

This module provides QAT support for the OCP Microscaling MXFP4 format.

Key differences between MXFP4 and NVFP4:
- Block size: MXFP4 uses 32 (default), NVFP4 uses 16 (fixed)
- Scale type: MXFP4 uses E8M0 (float8_e8m0fnu), NVFP4 uses float8_e4m3fn
- Scale calculation: MXFP4 supports FLOOR, RCEIL, CEIL, EVEN modes
"""

from dataclasses import dataclass
from typing import Optional

import torch

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    _addmm_mx_dispatch,
)
from torchao.quantization.qat import FakeQuantizeConfigBase
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


@dataclass
class MXFP4FakeQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for fake quantizing weights or activations to the OCP Microscaling MXFP4 format
    according to https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Fake quantization numerics follow `MXTensor` closely:
    https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py.

    Key differences from NVFP4:
    - Block size: 32 (default) vs NVFP4's fixed 16
    - Scale type: E8M0 (float8_e8m0fnu) vs NVFP4's float8_e4m3fn
    - Supports multiple scale calculation modes (FLOOR, RCEIL, CEIL, EVEN)

    Args:
        block_size (int): The block size for quantization (default 32, the OCP MX standard)
        scaling_mode (ScaleCalculationMode): How to calculate the block scales (default FLOOR)
        kernel_preference (KernelPreference): Which kernel to use for matmul (default EMULATED)
        is_swizzled_scales (bool): Whether scales are stored in swizzled (blocked) format
    """

    block_size: int = 32
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR
    kernel_preference: KernelPreference = KernelPreference.EMULATED
    is_swizzled_scales: bool = False


class _MXFP4QuantizedForwardFakeQuantizedBackward(torch.autograd.Function):
    """
    Autograd function for MXFP4 quantization + addmm in low precision during forward,
    and fake quantization in high precision during backward.

    This is the OCP Microscaling MXFP4 variant which differs from NVFP4 in:
    - Block size: 32 (default) vs 16
    - Scale format: E8M0 vs float8_e4m3fn
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation_config: MXFP4FakeQuantizeConfig,
        weight_config: MXFP4FakeQuantizeConfig,
    ) -> torch.Tensor:
        # quantize input activations
        _input = MXTensor.to_mx(
            _input,
            elem_dtype=torch.float4_e2m1fn_x2,
            block_size=activation_config.block_size,
            scaling_mode=activation_config.scaling_mode,
            kernel_preference=activation_config.kernel_preference,
            is_swizzled_scales=activation_config.is_swizzled_scales,
        )

        # quantize weights
        weight = MXTensor.to_mx(
            weight,
            elem_dtype=torch.float4_e2m1fn_x2,
            block_size=weight_config.block_size,
            scaling_mode=weight_config.scaling_mode,
            kernel_preference=weight_config.kernel_preference,
            is_swizzled_scales=weight_config.is_swizzled_scales,
        )

        ctx.save_for_backward(_input, weight)

        # Use addmm when bias is present, mm otherwise
        if bias is not None:
            aten_op = torch.ops.aten.addmm.default
        else:
            aten_op = torch.ops.aten.mm.default

        return _addmm_mx_dispatch(
            _input,
            weight.t(),
            aten_op,
            bias,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        _input, weight = ctx.saved_tensors
        assert isinstance(_input, MXTensor)
        assert isinstance(weight, MXTensor)
        _input = _input.dequantize(_input._orig_dtype)
        weight = weight.dequantize(weight._orig_dtype)
        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), _input)
        return grad_input, grad_weight, None, None, None


class MXFP4FakeQuantizedLinear(torch.nn.Linear):
    """
    Linear module for fake quantized MXFP4 weights and/or activations.

    The forward pass follows quantization and addmm numerics in `MXTensor`
    in lower precision exactly, while the backward pass uses dequantized
    (fake quantized) values in high precision.

    This uses the OCP Microscaling MXFP4 format which differs from NVFP4:
    - Block size: 32 (default, OCP standard) vs NVFP4's fixed 16
    - Scale format: E8M0 (float8_e8m0fnu) vs NVFP4's float8_e4m3fn
    - Supports multiple scale calculation modes

    Example usage::

        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        from torchao.quantization.qat import QATConfig

        base_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float4_e2m1fn_x2,
            weight_dtype=torch.float4_e2m1fn_x2,
        )
        quantize_(model, QATConfig(base_config, step="prepare"))
        # Model contains `MXFP4FakeQuantizedLinear` now

        train_loop(model)
        quantize_(model, QATConfig(base_config, step="convert"))
        # Model contains `nn.Linear` with `MXTensor` weights now
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[MXFP4FakeQuantizeConfig] = None,
        weight_config: Optional[MXFP4FakeQuantizeConfig] = None,
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
            raise ValueError("Weight only MXFP4 QAT not supported yet")
        self.activation_config = activation_config
        self.weight_config = weight_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.view(-1, x.shape[-1])
        else:
            batch_size = None
        fq = _MXFP4QuantizedForwardFakeQuantizedBackward.apply(
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
        activation_config: Optional[MXFP4FakeQuantizeConfig] = None,
        weight_config: Optional[MXFP4FakeQuantizeConfig] = None,
    ):
        new_linear = MXFP4FakeQuantizedLinear(
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
