# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .utils import dynamically_quantize_per_channel

__all__ = ["WeightOnlyInt8QuantLinear"]

class WeightOnlyInt8QuantLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`. It implements a
    mixed dtype matrix multiplication using int8 symmetric per-channel weight quantization.

    The primary goal of this class is to leverage int8 quantization for weights to reduce the
    memory footprint and computational requirements while performing linear transformations.
    This can be particularly beneficial for deploying models in low latency environments

    Attributes:
        w_int8 (torch.Tensor): The quantized weights in int8 format.
        scales (torch.Tensor): The scaling factors for each channel to convert the quantized
                               weights back to floating point format during the forward pass.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the WeightOnlyInt8QuantLinear module.

        Args:
            *args: Variable length argument list for `torch.nn.Linear`.
            **kwargs: Arbitrary keyword arguments.
                      Must include 'w_int8' (int8 quantized weights) and 'scales' (scaling factors).
        """
        w_int8 = kwargs.pop("w_int8")
        scales = kwargs.pop("scales")
        super().__init__(*args, **kwargs)

        self.register_buffer("w_int8", w_int8)
        self.register_buffer("scales", scales)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer, which consists of
        mixed dtype matrix multiplication using int8 symmetric per-channel weight quantization.

        Args:
            x (torch.Tensor): The input floating point tensor to the quantized linear layer.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output floating point tensor after the quantized matrix multiplication
                          and rescale.
        """
        x_view = x.view(-1, x.shape[-1])
        y = torch.mm(x_view, self.w_int8.to(x.dtype)) * self.scales
        y = y.reshape(*x.shape[:-1], -1)
        if self.bias is not None:
            y += self.bias
        return y

    @classmethod
    def from_float(cls, mod: torch.nn.Linear):
        """
        Converts a `torch.nn.Linear` module to a `WeightOnlyInt8QuantLinear` module.

        This method performs the conversion by dynamically quantizing the weights of the original
        floating point linear layer to int8 format and creating a new `WeightOnlyInt8QuantLinear`
        instance with these quantized weights and the corresponding scaling factors.

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            WeightOnlyInt8QuantLinear: The converted quantized linear module with int8 weights.
        """
        w_fp32 = mod.weight
        w_int8, scales, _zp = dynamically_quantize_per_channel(
            w_fp32, -128, 127, torch.int8
        )
        # Create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=mod.bias is not None,
            w_int8=w_int8.t().contiguous(),
            scales=scales,
        )
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        del new_mod.weight
        new_mod.bias = mod.bias
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod
