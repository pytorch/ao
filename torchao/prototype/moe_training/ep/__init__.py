# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Expert Parallelism (EP) autograd functions for MoE training with MXFP8.

This module contains custom autograd functions that enable efficient expert parallelism
with selective MXFP8 quantization during forward and backward passes.

The functions are designed to work together in the following pipeline:

Forward: bf16 -> a2a_dispatch (quantize) -> permute -> mxfp8 grouped GEMM -> unpermute -> a2a_combine
Backward: bf16 <- a2a_dispatch <- permute <- mxfp8 grouped GEMMs <- unpermute <- a2a_combine (quantize)
"""

from .a2a_combine import a2a_combine
from .a2a_dispatch import a2a_dispatch
from .permute import permute
from .unpermute import unpermute

__all__ = [
    "a2a_dispatch",
    "a2a_combine",
    "permute",
    "unpermute",
]
