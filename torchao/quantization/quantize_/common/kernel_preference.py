# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class KernelPreference(str, Enum):
    """
    Enum for specifying the groups of kernels used for quantization, matrix multiplication,
    or other compute ops for quantized tensors.

    Examples of how options affect the selected kernels can be found in tensor subclass
    implementations under ``torchao/quantization/quantize_/workflows``.

    Values:

    * ``AUTO``: Use the most efficient quantize and mm kernels chosen automatically
      based on hardware, library availability, and versions.
    * ``TORCH``: Use torch native quantize and quantized mm kernels.
    * ``MSLK``: Use quantize and quantized mm kernels from the mslk library (requires mslk).
    * ``EMULATED``: Emulates ``gemm_lowp(A, B)`` with ``gemm_fp32(A.dequantize(), B.dequantize())``.
      Intended for running CI on hardware without lowp gemm support, or debugging kernel numerics.
    """

    AUTO = "auto"
    TORCH = "torch"
    MSLK = "mslk"
    EMULATED = "emulated"


torch.serialization.add_safe_globals([KernelPreference])
