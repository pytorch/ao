# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from importlib import import_module

import torch
from torch import Tensor

try:
    from torch.distributed.tensor import DTensor

    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False


def instantiate_module(module_path, module_suffix):
    return getattr(import_module(module_path), module_suffix)


def is_dtensor(x):
    return HAS_DTENSOR and isinstance(x, DTensor)


def channel_bucketize(input: Tensor, boundaries: Tensor, right: bool = False) -> Tensor:
    """Bucketize each input row against its sorted boundary row.

    Batched binary search uses memory proportional to the inputs and boundaries,
    without constructing an input-by-boundary comparison mask. Ties, duplicate
    boundaries, empty boundary rows, and ``right`` follow torch.bucketize.
    """
    return torch.searchsorted(boundaries.contiguous(), input.contiguous(), right=right)
