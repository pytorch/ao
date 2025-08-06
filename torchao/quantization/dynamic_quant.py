# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .utils import (
    _quant_int8_dynamic_per_token_linear,
    dynamically_quantize_per_channel,
)

__all__ = ["DynamicallyPerAxisQuantizedLinear"]


class DynamicallyPerAxisQuantizedLinear(torch.nn.Linear):
    """
    This class is a replacement for `torch.nn.Linear`. It implements a
    quantized matmul using int8 dynamic symmetric per-token activation,
    and int8 symmetric per-channel weight quantization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Performs the forward pass of the quantized linear layer which consists
        of int8 dynamic symmetric per-token activation and int8 symmetric per-channel weight
        quantization

        Args:
            X (torch.Tensor): The input floating point tensor to the quantized linear layer.

        Returns:
            torch.Tensor: The output floating point tensor after the quantized matmul and rescale.

        """

        Y = _quant_int8_dynamic_per_token_linear(
            X, self.W_int_repr_t, self.W_scales, self.bias, X.dtype
        )
        return Y

    @classmethod
    def from_float(cls, mod: torch.nn.Linear) -> "DynamicallyPerAxisQuantizedLinear":
        """
        Converts a `mod` of class `torch.nn.Linear` to the
        `DynamicallyPerAxisQuantizedLinear` class

        Args:
            mod (torch.nn.Linear): The original `torch.nn.Linear` module to convert.

        Returns:
            DynamicallyPerAxisQuantizedLinear: The converted quantized linear module.

        """

        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=mod.bias is not None,
        )
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        W_int_repr, W_scales, _W_zps = dynamically_quantize_per_channel(
            mod.weight, -128, 127, torch.int8
        )
        new_mod.register_buffer("W_int_repr_t", W_int_repr.contiguous().t())
        new_mod.W_scales = nn.Parameter(W_scales)
        new_mod.bias = mod.bias
        del new_mod.weight

        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod
