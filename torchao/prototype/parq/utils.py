# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import Tensor


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
    return mask.int().argmax(dim=-1)
