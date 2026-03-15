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

Forward (left to right):
    mxfp8 a2a_dispatch -> mxfp8 permute -> mxfp8 grouped GEMM -> bf16 unpermute -> bf16 a2a_combine
Backward (right to left):
    bf16 a2a_dispatch.bwd <- bf16 permute.bwd <- mxfp8 grouped GEMMs bwd <- mxfp8 unpermute.bwd <- mxfp8 a2a_combine.bwd
"""

from .a2a_combine import a2a_combine_hp_fwd_mxfp8_bwd
from .a2a_dispatch import a2a_dispatch_mxfp8_fwd_hp_bwd
from .permute import permute_mxfp8_fwd_hp_bwd
from .unpermute import unpermute_hp_fwd_mxfp8_bwd

__all__ = [
    "a2a_dispatch_mxfp8_fwd_hp_bwd",
    "permute_mxfp8_fwd_hp_bwd",
    "unpermute_hp_fwd_mxfp8_bwd",
    "a2a_combine_hp_fwd_mxfp8_bwd",
]
