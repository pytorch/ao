# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
from torchao.float8.float8_training_tensor import (
    LinearMMConfig,
    ScaledMMConfig,
)


class _QuantizeableMM(torch.nn.Module):
    """
    A modularized version of `torch.mm` which is easy to module swap to
    a quantized version.

    Note: this is a prototype API which may change in a future release.
    """

    def forward(self, a, b):
        return torch.mm(a, b)


class _Float8MM(torch.nn.Module):
    """
    A float8 quantized version of `_QuantizeableMM`.

    Note: this is a prototype API which may change in a future release.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

    def forward(self, a, b):
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            a = a.to(autocast_dtype)
            b = b.to(autocast_dtype)

        c = matmul_with_hp_or_float8_args.apply(
            a,
            b,
            self.linear_mm_config,
            self.config,
        )
        return c

    @classmethod
    def from_float(cls, mod, config):
        assert isinstance(mod, _QuantizeableMM)
        return _Float8MM(config)
