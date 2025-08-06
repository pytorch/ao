# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch

from torchao.utils import TORCH_VERSION_AT_LEAST_2_5


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class KernelPreference(str, Enum):
    """Enum for specifying the groups of kernels that's used for quantization, matrix multiplication
    or other compute ops for quantized tensor

    Examples of how options affects the selected kernels can be found in tensor subclass implementations under torchao/quantization/quantize_/workflows
    """

    """Use the most efficient quantize and mm kernels chosen for user based on hardware and library availabilities and versions etc.
    """
    AUTO = "auto"

    """Use torch native quantize and quantized mm kernels
    """
    TORCH = "torch"

    """Use fbgemm quantize and quantized mm kernels, requires fbgemm_gpu_genai library
    """
    FBGEMM = "fbgemm"


if TORCH_VERSION_AT_LEAST_2_5:
    torch.serialization.add_safe_globals([KernelPreference])
