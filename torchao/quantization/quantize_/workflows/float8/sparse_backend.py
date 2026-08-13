# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum

import torch

__all__ = [
    "SparseBackend",
]


# can switch to StrEnum (https://docs.python.org/3/library/enum.html#enum.StrEnum)
# after python 3.10 is end of life (https://devguide.python.org/versions/)
class SparseBackend(str, Enum):
    """
    Kernel backend for the FP8 2:4 sparse conversion and linear operators.
    """

    """
    Hand-written CUTLASS C++ kernels, built as part of the torchao extension.
    """
    LEGACY = "legacy"
    """
    CuTeDSL kernels, JIT-compiled on first use through nvidia-cutlass-dsl.
    """
    CUTEDSL = "cutedsl"


torch.serialization.add_safe_globals([SparseBackend])
