# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed._tensor import DTensor
from torch.distributed.tensor import Replicate, Shard


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def dtensor_from_local_like(
    local_tensor: torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(like, DTensor):
        return local_tensor
    return DTensor.from_local(
        local_tensor,
        like.device_mesh,
        like.placements,
        run_check=False,
    )


def replicate_like_dtensor(
    tensor: torch.Tensor,
    *candidates: torch.Tensor,
) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor
    dtensor = next(
        (candidate for candidate in candidates if isinstance(candidate, DTensor)), None
    )
    if dtensor is None:
        return tensor
    return DTensor.from_local(
        tensor,
        dtensor.device_mesh,
        [Replicate() for _ in dtensor.device_mesh.shape],
        run_check=False,
    )


def grouped_quant_preserve_shardings():
    # weight_t and outputs keep logical shape (E, K, N).
    return [
        ([Replicate(), Replicate()], [Replicate(), None, None]),
        ([Shard(0), Shard(0)], [Shard(0), None, None]),
        ([Shard(1), Shard(1)], [Shard(1), None, None]),
        ([Shard(2), Shard(2)], [Shard(2), None, None]),
    ]


def grouped_quant_transpose_kn_shardings():
    # weight_t is logical (E, K, N); outputs are logical (E, N, K).
    return [
        ([Replicate(), Replicate()], [Replicate(), None, None]),
        ([Shard(0), Shard(0)], [Shard(0), None, None]),
        ([Shard(2), Shard(2)], [Shard(1), None, None]),
        ([Shard(1), Shard(1)], [Shard(2), None, None]),
    ]
