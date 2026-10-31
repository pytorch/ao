# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_tensor,
    init_device_mesh,
)

from torchao.optim import AdamW8bit
from torchao.optim.subclass_8bit import OptimState8bit


def _check_state_allocation(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        optimizer = AdamW8bit([torch.zeros(1)], block_size=256)
        # A globally eligible parameter must not switch to float32 merely
        # because FSDP2 puts fewer than 4096 elements on each rank.
        for placement in (Replicate(), Shard(0)):
            parameter = distribute_tensor(torch.zeros(6, 1024), mesh, [placement])
            for signed in (False, True):
                state = optimizer._new_buffer(parameter, signed)
                assert isinstance(state.to_local(), OptimState8bit)
                assert state.shape == parameter.shape
                assert state.placements == parameter.placements
                assert state.to_local().shape == parameter.to_local().shape

        # Keep the existing behavior for already eligible local shards.
        parameter = distribute_tensor(torch.zeros(16, 1024), mesh, [Shard(1)])
        assert isinstance(
            optimizer._new_buffer(parameter, True).to_local(), OptimState8bit
        )

        small = distribute_tensor(torch.zeros(2, 1024), mesh, [Shard(0)])
        assert type(optimizer._new_buffer(small, True).to_local()) is torch.Tensor

        for shape, placement in [((5, 1024), Shard(1)), ((2, 2176), Shard(0))]:
            parameter = distribute_tensor(torch.zeros(shape), mesh, [placement])
            with pytest.raises(ValueError, match="Low-bit optimizer state requires"):
                optimizer._new_buffer(parameter, True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_dtensor_state_precision_uses_global_parameter_size(tmp_path):
    mp.spawn(
        _check_state_allocation,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
