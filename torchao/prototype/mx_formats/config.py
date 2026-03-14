# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch

from torchao.prototype.mx_formats.constants import SUPPORTED_ELEM_DTYPES
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.utils import register_as_pytree_constant


class MXFP8Dim0CastKernelChoice(Enum):
    """
    Defines which kernel to use for mxfp8 casting along dim0.
    """

    TRITON = "triton"
    TORCH = "torch"


class MXFP8Dim1CastKernelChoice(Enum):
    """
    Defines which kernel to use for mxfp8 casting along dim1.
    """

    TRITON = "triton"
    CUDA = "cuda"
    TORCH = "torch"


class QuantizeToNVFP4KernelChoice(str, Enum):
    """Enum for specifying the kernel used for quantizing a high precision
    tensor (float32/bfloat16/float16) to nvfp4 tensor with blockwise quantization
    """

    TORCH = "torch"
    """Use torch native high precision to nvfp4 quantize kernel implemented with torch ops"""

    MSLK = "mslk"
    """Use MSLK triton high precision to nvfp4 quantize kernel"""


torch.serialization.add_safe_globals([QuantizeToNVFP4KernelChoice])

# register as pytree constant so we can use dynamo nonstrict trace in torchao.prototype.moe_training.ep
@register_as_pytree_constant
class ScaleCalculationMode(Enum):
    """
    Enum representing the different methods for calculating MX block scaling.
    There are four methods available:

    FLOOR: This method is recommended by the OCP MX Spec 1.0 and uses X = 2^floor(log2(max_abs(v))-max_exp).
           It result in overflow issues for large values and bad for gradient quantization.

    RCEIL: The method is to apply ceil to the ratio of max_abs(v) and max_pos.
           This method's detail is described in https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
           Section "Computing scaling and conversion factors for FP8 with UE8M0 scales"

    CEIL: This method avoids overflow issues, but small values may shift to 0 due to a large scaling factor.
           It uses X = 2^ceil(log2(max_abs(v))-max_exp).

    EVEN: This method is a trade-off between FLOOR and CEIL. It uses X = 2^(floor(log2(rounding(max_abs(v)))-max_exp)).
           It provides better accuracy for MX4 training compared to FLOOR and CEIL.
           Note: EVEN does not work with torch.compile yet:
           https://gist.github.com/vkuzo/1a04845cd503b1c75291aa1ea3bf79c4

    """

    FLOOR = "floor"
    RCEIL = "rceil"
    CEIL = "ceil"
    EVEN = "even"

    def __eq__(self, other):
        if isinstance(other, ScaleCalculationMode):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


def _validate_elem_dtype(elem_dtype: torch.dtype) -> None:
    """Validate that elem_dtype is a supported MX element dtype."""
    assert elem_dtype in SUPPORTED_ELEM_DTYPES, (
        f"elem_dtype must be one of {SUPPORTED_ELEM_DTYPES}, got {elem_dtype}"
    )


def _validate_kernel_preference(kernel_preference, block_size, elem_dtype):
    if kernel_preference == KernelPreference.AUTO:
        if elem_dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
            assert block_size == 32, f"block_size must be 32, got {block_size}"
        else:
            raise AssertionError(
                f"unsupported {kernel_preference=}, {block_size=}, {elem_dtype=}"
            )
    else:
        assert kernel_preference == KernelPreference.EMULATED, (
            f"unsupported {kernel_preference=}, {block_size=}, {elem_dtype=}"
        )
