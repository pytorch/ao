# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    DTYPE_TO_SHORT_STR,
    SUPPORTED_ELEM_DTYPES,
)


class MXGemmKernelChoice(Enum):
    # always available - MX operands are dequantized and a high precision
    # gemm is run
    EMULATED = "emulated"

    # available only when CUDA capability is greater than or equal to 10.0
    CUTLASS = "cutlass"

    # available only when CUDA capability is greater than or equal to 10.0
    # available on recent versions of PyTorch nightly, with https://github.com/pytorch/pytorch/pull/147548
    # note: torch.compile does not work yet, see https://github.com/pytorch/pytorch/issues/147873
    CUBLAS = "cublas"


# Pre-made recipes for common configurations
class MXLinearRecipeName(Enum):
    MXFP8_EMULATED = "mxfp8_emulated"
    MXFP8_CUBLAS = "mxfp8_cublas"
    MXFP8_CUTLASS = "mxfp8_cutlass"
    MXFP4_EMULATED = "mxfp4_emulated"
    MXFP4_CUTLASS = "mxfp4_cutlass"


def _validate_elem_dtype(elem_dtype):
    assert (
        elem_dtype in SUPPORTED_ELEM_DTYPES
    ), f"elem_dtype: expected one of {SUPPORTED_ELEM_DTYPES}, got {elem_dtype}"


def _validate_gemm_kernel_choice(gemm_kernel_choice, block_size, elem_dtype):
    if gemm_kernel_choice == MXGemmKernelChoice.CUTLASS:
        assert (
            block_size == 32
        ), f"block_size must be 32 to use the CUTLASS MX gemm kernels, got {block_size}"
        valid_dtypes = [torch.float8_e4m3fn, DTYPE_FP4]
        assert (
            elem_dtype in valid_dtypes
        ), f"elem_dtype must be one of {valid_dtypes} to use the CUTLASS MX gemm kernels, got {elem_dtype}"
    elif gemm_kernel_choice == MXGemmKernelChoice.CUBLAS:
        assert (
            block_size == 32
        ), f"block_size must be 32 to use the cuBLAS MX gemm kernels, got {block_size}"
        valid_dtypes = [torch.float8_e4m3fn]
        assert (
            elem_dtype in valid_dtypes
        ), f"elem_dtype must be one of {valid_dtypes} to use the CUTLASS MX gemm kernels, got {elem_dtype}"


@dataclass
class MXLinearConfig(AOBaseConfig):
    # block size for scaling, default is 32 to match
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
    # section 5.2
    block_size: int = 32

    # element dtype, used for activations, weights and gradients
    elem_dtype: Any = torch.float8_e4m3fn

    # overrides for element dtype for weights and gradients
    # TODO(future PR): refactor to make this cleaner
    elem_dtype_weight_override: Optional[Any] = None
    elem_dtype_grad_output_override: Optional[Any] = None

    # defines the gemm kernel choice, if the chosen kernel is not supported
    # on the given hardware an exception will be thrown
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED

    # If True, uses a custom triton kernel for cast to mxfp8 across dim1
    # TODO(1945): remove this config option once torch.compile gives us
    # a fast kernel
    use_fp8_dim1_cast_triton_kernel: bool = False

    # If True, uses a custom triton kernel for fp4 dequantize
    use_fp4_custom_triton_dequant_kernel: bool = False

    def __post_init__(self):
        _validate_elem_dtype(self.elem_dtype)
        _validate_gemm_kernel_choice(
            self.gemm_kernel_choice, self.block_size, self.elem_dtype
        )
        if self.elem_dtype_weight_override is not None:
            _validate_elem_dtype(self.elem_dtype_weight_override)
            assert self.gemm_kernel_choice == MXGemmKernelChoice.EMULATED, "unsupported"
        if self.elem_dtype_grad_output_override is not None:
            _validate_elem_dtype(self.elem_dtype_grad_output_override)
            assert self.gemm_kernel_choice == MXGemmKernelChoice.EMULATED, "unsupported"

    @staticmethod
    def from_recipe_name(
        recipe_name: Union[MXLinearRecipeName, str],
    ) -> "MXLinearConfig":
        """
        Input: `MXLinearRecipeName` value, or a string representing a `MXLinearRecipeName` value
        Output: a `MXLinearConfig` configured to implement the specified recipe
        """
        if type(recipe_name) == str:
            valid_names = [n.value for n in MXLinearRecipeName]
            assert (
                recipe_name in valid_names
            ), f"recipe_name {recipe_name} not in valid names {valid_names}"
            recipe_name = MXLinearRecipeName(recipe_name)

        if recipe_name is MXLinearRecipeName.MXFP8_EMULATED:
            return MXLinearConfig()
        elif recipe_name is MXLinearRecipeName.MXFP8_CUBLAS:
            return MXLinearConfig(gemm_kernel_choice=MXGemmKernelChoice.CUBLAS)
        elif recipe_name is MXLinearRecipeName.MXFP8_CUTLASS:
            return MXLinearConfig(gemm_kernel_choice=MXGemmKernelChoice.CUTLASS)
        elif recipe_name is MXLinearRecipeName.MXFP4_EMULATED:
            return MXLinearConfig(elem_dtype=DTYPE_FP4)
        elif recipe_name is MXLinearRecipeName.MXFP4_CUTLASS:
            return MXLinearConfig(
                elem_dtype=DTYPE_FP4, gemm_kernel_choice=MXGemmKernelChoice.CUTLASS
            )
        else:
            raise AssertionError(f"unknown recipe_name {recipe_name}")

    def short_str(self) -> str:
        """
        Returns a concise representation of the current config.
        """
        s = f"bl_sz={self.block_size}, lp_dtype={DTYPE_TO_SHORT_STR[self.elem_dtype]}"
        if self.elem_dtype_weight_override is not None:
            s += (
                f", lp_w_override={DTYPE_TO_SHORT_STR[self.elem_dtype_weight_override]}"
            )
        if self.elem_dtype_grad_output_override is not None:
            s += f", lp_go_override={DTYPE_TO_SHORT_STR[self.elem_dtype_grad_output_override]}"
        s += f", kernel={self.gemm_kernel_choice.value}"
        if self.use_fp8_dim1_cast_triton_kernel:
            s += ", use_fp8_dim1_cast_triton_kernel=True"
        if self.use_fp4_custom_triton_dequant_kernel:
            s += ", use_fp4_custom_triton_dequant_kernel=True"
        return s


@dataclass
class MXInferenceLinearConfig(AOBaseConfig):
    # block size for scaling, default is 32 to match
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
    # section 5.2
    block_size: int = 32

    # element dtype, used for activations, weights and gradients
    elem_dtype: Any = torch.float8_e4m3fn
    # TODO(future PR): support different elem_dtype for activations vs weights

    # defines the gemm kernel choice, if the chosen kernel is not supported
    # on the given hardware an exception will be thrown
    gemm_kernel_choice: MXGemmKernelChoice = MXGemmKernelChoice.EMULATED

    # If True, uses a custom triton kernel for fp4 dequantize
    use_fp4_custom_triton_dequant_kernel: bool = False

    # If True, packs 4xFP6 into 3xuint8 containers for inference, using custom triton
    # kernels (fused unpack/dequantize).
    pack_fp6: bool = True

    def __post_init__(self):
        _validate_elem_dtype(self.elem_dtype)
        _validate_gemm_kernel_choice(
            self.gemm_kernel_choice, self.block_size, self.elem_dtype
        )

    def short_str(self) -> str:
        """
        Returns a concise representation of the current config.
        """
        s = f"bl_sz={self.block_size}, lp_dtype={DTYPE_TO_SHORT_STR[self.elem_dtype]}"
        s += f", kernel={self.gemm_kernel_choice.value}"
        if self.use_fp4_custom_triton_dequant_kernel:
            s += ", use_fp4_custom_triton_dequant_kernel=True"
        if self.elem_dtype in (DTYPE_FP6_E2M3, DTYPE_FP6_E3M2) and self.pack_fp6:
            s += ", pack_fp6=True"
        return s

    # TODO(future PR): add a recipe to config API for inference
