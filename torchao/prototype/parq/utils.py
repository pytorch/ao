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
    """Generalizes torch.bucketize to run on 2-D boundaries."""
    inf_pad = torch.full_like(boundaries[:, :1], torch.inf)
    boundaries = (
        torch.cat((-inf_pad, boundaries), dim=1)
        if right
        else torch.cat((boundaries, inf_pad), dim=1)
    )
    boundaries = boundaries.unsqueeze(1)
    input = input.unsqueeze(-1)
    mask = input.ge(boundaries) if right else input.le(boundaries)
    return mask.to(torch.uint8).argmax(dim=-1)
