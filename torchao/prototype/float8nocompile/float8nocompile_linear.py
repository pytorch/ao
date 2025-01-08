# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear` which
does not require `torch.compile` to be performant.
"""

import torch

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, ScaledMMConfig
from torchao.prototype.float8nocompile.float8nocompile_scaling_utils import (
    ToFP8ColumnMajor,
    ToFP8ColumnMajorT,
    ToFP8RowAndColumnMajor,
    ToFP8RowMajor,
    ToFP8RowMajorT,
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
        output = matmul_with_args_in_hp.apply(
            input,
            self.weight,
            self.config,
            self.linear_mm_config,
            self.kernel_algo,
        )
        return output

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


class matmul_with_args_in_hp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_hp, weight_hp, config, linear_mm_config, kernel_algo):
        # output = input @ weight_t
        input_fp8_row_major, input_fp8_col_major = ToFP8RowAndColumnMajor.apply(
            input_hp,
            config.cast_config_input.target_dtype,
            linear_mm_config,
            GemmInputRole.INPUT,
            kernel_algo,
        )
        weight_t_fp8_col_major = ToFP8ColumnMajorT.apply(
            weight_hp,
            config.cast_config_weight.target_dtype,
            linear_mm_config,
            GemmInputRole.WEIGHT,
            kernel_algo,
        )
        output = torch.mm(input_fp8_row_major, weight_t_fp8_col_major)

        # save data for backward before returning
        ctx.save_for_backward(input_fp8_col_major, weight_hp)
        ctx.config = config
        ctx.linear_mm_config = linear_mm_config
        ctx.kernel_algo = kernel_algo

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8_col_major, weight_hp = ctx.saved_tensors

        # cast grad output to float8_e5m2 for backward
        grad_output_fp8_row_major = ToFP8RowMajor.apply(
            grad_output,
            ctx.config.cast_config_grad_output.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
            ctx.kernel_algo,
        )

        # grad_input = grad_output @ weight
        weight_fp8_col_major = ToFP8ColumnMajor.apply(
            weight_hp,
            ctx.config.cast_config_weight.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.WEIGHT,
            ctx.kernel_algo,
        )
        grad_input = torch.mm(grad_output_fp8_row_major, weight_fp8_col_major)

        # grad_weight = grad_output_t @ input
        # apparently this variant is slightly faster than `grad_weight_t = input_t @ grad_output`
        # source: https://github.com/pytorch/ao/blob/fe5f11b2c58b452e01ba9ec7359629928b143619/torchao/float8/float8_linear.py#L84-L85
        grad_output_t_row_major = ToFP8RowMajorT.apply(
            grad_output,
            ctx.config.cast_config_grad_output.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
            ctx.kernel_algo,
        )
        grad_weight = torch.mm(grad_output_t_row_major, input_fp8_col_major)
        return grad_input, grad_weight, None, None, None
