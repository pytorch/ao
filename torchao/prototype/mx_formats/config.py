# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    SUPPORTED_ELEM_DTYPES,
)


class MXGemmKernelChoice(Enum):
    # always available - MX operands are dequantized and a high precision
    # gemm is run
    EMULATED = "emulated"

    # available only when CUDA capability is greater than or equal to 10.0
    CUTLASS = "cutlass"

    # TODO(future PR): add cuBLAS here once we land pytorch/pytorch support


@dataclass
class MXLinearConfig:
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

    # If True, uses a custom triton kernel for fp4 dequantize
    use_fp4_custom_triton_dequant_kernel: bool = False

    def __post_init__(self):
        # validate elem_dtype and its overrides
        assert (
            self.elem_dtype in SUPPORTED_ELEM_DTYPES
        ), f"elem_dtype: expected one of {SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"
        if self.elem_dtype_weight_override is not None:
            assert (
                self.elem_dtype_weight_override in SUPPORTED_ELEM_DTYPES
            ), f"elem_dtype_weight_override: expected one of {SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"
        if self.elem_dtype_grad_output_override is not None:
            assert (
                self.elem_dtype_grad_output_override in SUPPORTED_ELEM_DTYPES
            ), f"elem_dtype_grad_output_override: expected one of {SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"

        # validate that block size and elem_dtype matches kernel choice
        if self.gemm_kernel_choice == MXGemmKernelChoice.CUTLASS:
            assert (
                self.block_size == 32
            ), f"block_size must be 32 to use the CUTLASS MX gemm kernels, got {self.block_size}"
            valid_dtypes = [torch.float8_e4m3fn, DTYPE_FP4]
            assert (
                self.elem_dtype in valid_dtypes
            ), f"elem_dtype must be one of {valid_dtypes} to use the CUTLASS MX gemm kernels, got {self.elem_dtype}"
            assert (
                self.elem_dtype_weight_override is None
            ), "elem_dtype_weight_override not supported for CUTLASS MX gemm kernels"
            assert (
                self.elem_dtype_grad_output_override is None
            ), "elem_dtype_grad_output_override not supported for CUTLASS MX gemm kernels"
