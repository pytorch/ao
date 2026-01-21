# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

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
)
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    _addmm_mx_dispatch,
)
from torchao.quantization.qat import FakeQuantizeConfigBase
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


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
            Supported values: torch.float4_e2m1fn_x2 (default), torch.float8_e4m3fn,
            torch.float8_e5m2
        block_size (int): The block size for quantization (default 32, the OCP MX standard)
        scaling_mode (ScaleCalculationMode): How to calculate the block scales (default FLOOR)
        kernel_preference (KernelPreference): Which kernel to use for matmul (default EMULATED)
    """

    dtype: torch.dtype = torch.float4_e2m1fn_x2
    block_size: int = 32
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR
    kernel_preference: KernelPreference = KernelPreference.EMULATED

    def __post_init__(self):
        _validate_elem_dtype(self.dtype)


class _MXQuantizedForwardFakeQuantizedBackward(torch.autograd.Function):
    """
    Autograd function for MX quantization + addmm in low precision during forward,
    and fake quantization in high precision during backward.

    This is the OCP Microscaling MX variant which differs from NVFP4 in:
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
        _input_2d = _input_2d.dequantize(_input_2d._orig_dtype)
        weight = weight.dequantize(weight._orig_dtype)

        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])

        grad_input_2d = torch.mm(grad_output_2d, weight)
        grad_weight = torch.mm(grad_output_2d.t(), _input_2d)

        grad_input = grad_input_2d.view(*orig_shape)
        return grad_input, grad_weight, None, None, None


class MXFakeQuantizedLinear(torch.nn.Linear):
    """
    Linear module for fake quantized MX weights and/or activations.

    The forward pass follows quantization and addmm numerics in `MXTensor`
    in lower precision exactly, while the backward pass uses dequantized
    (fake quantized) values in high precision.

    This uses the OCP Microscaling MX format which differs from NVFP4:
    - Block size: 32 (default, OCP standard) vs NVFP4's fixed 16
    - Scale format: E8M0 (float8_e8m0fnu) vs NVFP4's float8_e4m3fn
    - Supports multiple scale calculation modes
    - Supports multiple element dtypes (MXFP4, MXFP8)

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
