# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


def is_dtensor(tensor: torch.Tensor) -> bool:
    return isinstance(tensor, DTensor)


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if is_dtensor(tensor) else tensor


def dtensor_from_local_like(
    local_tensor: torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    if not is_dtensor(like):
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
    if is_dtensor(tensor):
        return tensor
    dtensor = next(
        (candidate for candidate in candidates if is_dtensor(candidate)), None
    )
    if dtensor is None:
        return tensor
    return DTensor.from_local(
        tensor,
        dtensor.device_mesh,
        [Replicate() for _ in dtensor.device_mesh.shape],
        run_check=False,
    )


def require_dtensor_replicated(
    tensor: torch.Tensor,
    name: str,
    *,
    reason: str,
) -> None:
    if not is_dtensor(tensor):
        return
    if not all(isinstance(placement, Replicate) for placement in tensor.placements):
        raise NotImplementedError(f"{name} must be replicated {reason}")


def require_dtensor_not_sharded_on_dim(
    tensor: torch.Tensor,
    name: str,
    dim: int,
    *,
    dim_name: str | None = None,
    allow_partial: bool = False,
    reason: str,
) -> None:
    if not is_dtensor(tensor):
        return
    dim_desc = dim_name if dim_name is not None else f"dimension {dim}"
    for placement in tensor.placements:
        if isinstance(placement, Partial) and not allow_partial:
            raise NotImplementedError(f"{name} must not be Partial {reason}")
        if isinstance(placement, Shard) and placement.dim == dim:
            raise NotImplementedError(
                f"{name} must not be sharded on the {dim_desc} {reason}"
            )


def two_output_quant_shardings(
    *,
    shard_dim0_outputs: tuple[Shard, Shard],
    shard_dim1_outputs: tuple[Shard, Shard],
):
    # order is: ([outputs, ...], [inputs, ...])
    return [
        ([Replicate(), Replicate()], [Replicate(), None, None]),
        (list(shard_dim0_outputs), [Shard(0), None, None]),
        (list(shard_dim1_outputs), [Shard(1), None, None]),
    ]


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
