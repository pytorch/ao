# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for scaling high precision tensors to float8.
"""

import torch

from torchao.float8.float8_tensor import (
    GemmInputRole,
    LinearMMConfig,
)
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    KernelAlgorithm,
    hp_to_fp8_col_major,
    hp_to_fp8_col_major_t,
    hp_to_fp8_row_and_col_major,
    hp_to_fp8_row_major,
    hp_to_fp8_row_major_t,
)


class ToFP8RowAndColumnMajor(torch.autograd.Function):
    """
    A differentiable conversion to fp8.
    * forward: convert from high precision to float8 and produces both row-major and column-major outputs
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    ):
        fp8_row_major, fp8_col_major = hp_to_fp8_row_and_col_major(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
            algo=kernel_algo,
        )
        return fp8_row_major, fp8_col_major

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None


class ToFP8RowMajor(torch.autograd.Function):
    """
    A differentiable conversion to fp8 in row-major layout.
    * forward: convert from high precision to float8 with row-major memory layout
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    ):
        fp8_row_major = hp_to_fp8_row_major(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
            algo=kernel_algo,
        )
        return fp8_row_major

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None


class ToFP8RowMajorT(torch.autograd.Function):
    """
    A differentiable conversion to fp8 with transposed dimensions in row-major layout.
    * forward: convert from high precision to float8 with transposed dimensions with row-major memory layout
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    ):
        fp8_row_major_t = hp_to_fp8_row_major_t(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
            algo=kernel_algo,
        )
        return fp8_row_major_t

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None


class ToFP8ColumnMajor(torch.autograd.Function):
    """
    A differentiable conversion to fp8 in column-major layout.
    * forward: convert from high precision to float8 with column-major memory layout
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    ):
        fp8_col_major = hp_to_fp8_col_major(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
            algo=kernel_algo,
        )
        return fp8_col_major

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None


class ToFP8ColumnMajorT(torch.autograd.Function):
    """
    A differentiable conversion to fp8 with transposed dimensions in column-major layout.
    * forward: convert from high precision to float8 with transposed dimensions in column-major memory layout.
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
        kernel_algo: KernelAlgorithm = KernelAlgorithm.ATOMIC_MAX,
    ):
        fp8_col_major_t = hp_to_fp8_col_major_t(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
            algo=kernel_algo,
        )
        return fp8_col_major_t

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None
