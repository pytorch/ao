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
from torchao.float8.float8_training_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from torchao.prototype.float8nocompile.float8nocompile_scaling_utils import (
    ToFP8ColumnMajor,
    ToFP8ColumnMajorT,
    ToFP8RowAndColumnMajor,
    ToFP8RowMajor,
    ToFP8RowMajorTAndNonT,
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
        self.config = kwargs.pop("config")
        self.kernel_algo = kwargs.pop("kernel_algo")
        self.no_precompute_for_backward = kwargs.pop(
            "no_precompute_for_backward", False
        )
        super().__init__(*args, **kwargs)

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                self.config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                self.config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                self.config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.no_precompute_for_backward:
            output = matmul_with_args_in_hp_no_precompute_for_backward.apply(
                input,
                self.weight,
                self.config,
                self.linear_mm_config,
                self.kernel_algo,
            )
        else:
            output = matmul_with_args_in_hp.apply(
                input,
                self.weight,
                self.config,
                self.linear_mm_config,
                self.kernel_algo,
            )
        return output

    @classmethod
    def from_float(
        cls,
        mod,
        config: Float8LinearConfig,  # only default config is supported, non-defaults silently ignored
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
        no_precompute_for_backward: bool = False,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8 (note: only
                default config is supported, non-defaults silently ignored)
        """
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
                kernel_algo=kernel_algo,
                no_precompute_for_backward=no_precompute_for_backward,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias

        # TODO(danielvegamyhre): support for FSDP once dependencies are implemented
        return new_mod


class matmul_with_args_in_hp(torch.autograd.Function):
    """FP8 matmul with args in high precision to be used in a region without AC.
    FP8 tensors only needed for backward are computed as part of kernels in the forward pass,
    to reduce number of kernel dispatches and increase throughput, at the cost of higher
    peak memory usage."""

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        config: Float8LinearConfig,
        linear_mm_config: LinearMMConfig,
        kernel_algo: KernelAlgorithm,
    ):
        # reshape to be 2D for triton kernels
        orig_input_shape = input_hp.shape
        input_hp = input_hp.reshape(-1, input_hp.shape[-1])

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
        ctx.no_precompute_for_backward = False

        # reshape back to expected dims
        output = output.reshape(*orig_input_shape[:-1], output.shape[-1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output may not be contiguous in cases like:
        # output.sum().backward() where grad is all 1s, so the (M,N) view of the scalar "1"
        # results in a non-contiguous tensor with stride (0,0).
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        input_fp8_col_major, weight_hp = ctx.saved_tensors

        # reshsape to be 2D for triton kernels
        orig_grad_output_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        # cast grad output to float8_e5m2 for backward
        grad_output_fp8_row_major, grad_output_t_row_major = (
            ToFP8RowMajorTAndNonT.apply(
                grad_output,
                ctx.config.cast_config_grad_output.target_dtype,
                ctx.linear_mm_config,
                GemmInputRole.GRAD_OUTPUT,
                ctx.kernel_algo,
            )
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

        # reshape grad input to match original shape
        grad_input = grad_input.reshape(
            *orig_grad_output_shape[:-1], grad_input.shape[-1]
        )

        # grad_weight = grad_output_t @ input
        # apparently this variant is slightly faster than `grad_weight_t = input_t @ grad_output`
        # source: https://github.com/pytorch/ao/blob/fe5f11b2c58b452e01ba9ec7359629928b143619/torchao/float8/float8_linear.py#L84-L85
        grad_weight = torch.mm(grad_output_t_row_major, input_fp8_col_major)

        # grad input shape
        return grad_input, grad_weight, None, None, None, None


class matmul_with_args_in_hp_no_precompute_for_backward(torch.autograd.Function):
    """FP8 matmul with args in high precision to be used in a region with AC.
    FP8 tensors only needed for backward are only computed in the backward pass
    when needed, to reduce peak memory usage."""

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        config: Float8LinearConfig,
        linear_mm_config: LinearMMConfig,
        kernel_algo: KernelAlgorithm,
    ):
        # reshape to be 2D for triton kernels
        orig_input_shape = input_hp.shape
        input_hp = input_hp.reshape(-1, input_hp.shape[-1])

        # output = input @ weight_t
        input_fp8_row_major = ToFP8RowMajor.apply(
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

        # with AC we only will save the original hp input tensor and weight for backward,
        # and do the necessary fp8 conversions during the backward pass.
        ctx.save_for_backward(input_hp, weight_hp)
        ctx.config = config
        ctx.linear_mm_config = linear_mm_config
        ctx.kernel_algo = kernel_algo
        ctx.no_precompute_for_backward = True

        # reshape back to expected dims
        output = output.reshape(*orig_input_shape[:-1], output.shape[-1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output may not be contiguous in cases like:
        # output.sum().backward() where grad is all 1s, so the (M,N) view of the scalar "1"
        # results in a non-contiguous tensor with stride (0,0).
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        input_hp, weight_hp = ctx.saved_tensors

        # reshsape to be 2D for triton kernels
        orig_grad_output_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        # cast grad output to float8_e5m2 for backward
        grad_output_fp8_row_major, grad_output_t_row_major = (
            ToFP8RowMajorTAndNonT.apply(
                grad_output,
                ctx.config.cast_config_grad_output.target_dtype,
                ctx.linear_mm_config,
                GemmInputRole.GRAD_OUTPUT,
                ctx.kernel_algo,
            )
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

        # reshape grad input to match original shape
        grad_input = grad_input.reshape(
            *orig_grad_output_shape[:-1], grad_input.shape[-1]
        )

        # grad_weight = grad_output_t @ input
        # apparently this variant is slightly faster than `grad_weight_t = input_t @ grad_output`
        # source: https://github.com/pytorch/ao/blob/fe5f11b2c58b452e01ba9ec7359629928b143619/torchao/float8/float8_linear.py#L84-L85
        input_fp8_col_major = ToFP8ColumnMajor.apply(
            input_hp,
            ctx.config.cast_config_input.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.INPUT,
            ctx.kernel_algo,
        )
        grad_weight = torch.mm(grad_output_t_row_major, input_fp8_col_major)

        # grad input shape
        return grad_input, grad_weight, None, None, None, None
