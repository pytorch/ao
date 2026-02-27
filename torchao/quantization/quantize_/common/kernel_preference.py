# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch

from torchao.utils import register_as_pytree_constant


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
@register_as_pytree_constant
class KernelPreference(str, Enum):
    """Enum for specifying the groups of kernels that's used for quantization, matrix multiplication
    or other compute ops for quantized tensor

    Examples of how options affects the selected kernels can be found in tensor subclass implementations under torchao/quantization/quantize_/workflows
    """

    AUTO = "auto"
    """Use the most efficient quantize and mm kernels chosen for user based on hardware and library availabilities and versions etc.
    """

    TORCH = "torch"
    """Use torch native quantize and quantized mm kernels
    """

    MSLK = "mslk"
    """Use quantize and quantized mm kernels from mslk library, requires mslk library
    """

    EMULATED = "emulated"
    """Emulates gemm_lowp(A, B) with gemm_fp32(A.dequantize(), B.dequantize()).
    Intended use cases are:
    1. Running CI for product logic on hardware which does not support the
       actual lowp gemm.
    2. Debugging kernel numerics issues.
    """


torch.serialization.add_safe_globals([KernelPreference])
