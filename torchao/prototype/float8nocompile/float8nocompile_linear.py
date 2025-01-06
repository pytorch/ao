# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear` which
does not require `torch.compile` to be performant.
"""
from typing import Optional

import torch

from torchao.float8.config import Float8LinearConfig, ScalingGranularity, ScalingType
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_linear import manual_float8_matmul_with_args_in_float8
from torchao.float8.float8_scaling_utils import NoopFwToFloat8BwDynamic
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, ScaledMMConfig
from torchao.float8.float8_utils import tensor_to_scale

from torchao.prototype.float8nocompile.float8nocompile_scaling_utils import (
    Float8NoCompileConversionFunc,
    NoopFwToFloat8NoCompileBwDynamic,
)
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
)


class Float8LinearNoCompile(torch.nn.Linear):
    """
    Float8LinearNoCompile is a version of Float8Linear that does not require
    the use of torch.compile to be performant.

    Note: this is **prototype** and not suitable for production use.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float8LinearConfig
        """
        config = kwargs.pop("config")
        kernel_algo = kwargs.pop("kernel_algo")
        emulate = config.emulate
        super().__init__(*args, **kwargs)

        self.config = config
        self.kernel_algo = kernel_algo

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO(danielvegamyhre): support for FSDP once dependencies are implemented
        input_fp8 = self.cast_input_to_float8(input)
        weight_fp8_t = self.cast_weight_to_float8_t(self.weight)

        # compute fp8 matmul
        output = manual_float8_matmul_with_args_in_float8.apply(input_fp8, weight_fp8_t)

        # cast grad_output to float8_e5m2 during backward
        return self.cast_output_to_float8_in_bw(output)

    def cast_input_to_float8(self, input: torch.Tensor) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        return Float8NoCompileConversionFunc.apply(
            input,
            self.config.cast_config_input.target_dtype,
            self.linear_mm_config,
            GemmInputRole.INPUT,
            self.kernel_algo,
        )

    def cast_weight_to_float8_t(
        self,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        weight_fp8 = Float8NoCompileConversionFunc.apply(
            weight,
            self.config.cast_config_weight.target_dtype,
            self.linear_mm_config,
            GemmInputRole.WEIGHT,
            self.kernel_algo,
        )
        return weight_fp8.t()

    def cast_output_to_float8_in_bw(self, output: torch.Tensor) -> torch.Tensor:
        # casts grad_output to float8_e5m2 for backward
        return NoopFwToFloat8NoCompileBwDynamic.apply(
            output,
            self.config.cast_config_grad_output.target_dtype,
            self.linear_mm_config,
            self.kernel_algo,
        )

    @classmethod
    def from_float(cls, mod, kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
                kernel_algo=kernel_algo,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias

        # TODO(danielvegamyhre): support for FSDP once dependencies are implemented
        return new_mod
