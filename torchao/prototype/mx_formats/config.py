# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Optional

import torch

from torchao.prototype.mx_formats.constants import SUPPORTED_ELEM_DTYPES


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

    # If True, uses a custom triton kernel for fp4 dequantize
    use_fp4_custom_triton_dequant_kernel: bool = False

    def __post_init__(self):
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
