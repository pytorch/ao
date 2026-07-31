# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
4-GPU FSDP2 + TP smoke coverage for blockwise FP8 MoE grouped GEMM.

This is intended as a manual torchrun test for a 2D ``(dp, tp) = (2, 2)`` mesh:

    NCCL_SOCKET_IFNAME=lo torchrun --standalone --nproc_per_node=4 \
        test/prototype/moe_training/test_fp8_blockwise_fsdp2_tp.py

Note: for now, this does not run in CI.
"""

import os

import pytest


def _skip_or_exit(reason: str) -> None:
    if __name__ == "__main__":
        print(f"SKIPPED: {reason}")
        raise SystemExit(0)
    pytest.skip(reason, allow_module_level=True)


if os.environ.get("WORLD_SIZE") != "4":
    _skip_or_exit("Manual 4-GPU torchrun test; not run in regular pytest jobs")

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn import functional as F

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from _fp8_blockwise_distributed_test_utils import (
    allreduce_reference_grads,
    assert_close,
    assert_dtensor_parameter_grads_match,
    assert_dtensor_parameter_values_match,
    assert_parameters_are_dtensors,
    full_tensor,
    get_blockwise_moe_skip_reason,
    get_replicated_local_batch,
    make_blockwise_grouped_experts_pair,
    parallelize_blockwise_grouped_experts_tensor_parallel,
)

if skip_reason := get_blockwise_moe_skip_reason(
    triton_module=triton,
    min_cuda_devices=4,
):
    _skip_or_exit(skip_reason)


def setup_distributed() -> DeviceMesh:
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    assert world_size == 4, f"This test requires WORLD_SIZE=4, got {world_size}"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1)
    return init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))


def _test_blockwise_grouped_mm_fsdp2_tp(mesh: DeviceMesh) -> None:
    dp_rank, _tp_rank = mesh.get_coordinate()
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    ref_model, dist_model = make_blockwise_grouped_experts_pair(
        broadcast_weights=True,
        pad_token_groups_for_grouped_mm=True,
    )
    dist_model = parallelize_blockwise_grouped_experts_tensor_parallel(
        dist_model,
        tp_mesh,
    )
    dist_model = fully_shard(dist_model, mesh=dp_mesh)

    ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
    dist_optim = torch.optim.SGD(dist_model.parameters(), lr=1e-2)
    ref_named_params = dict(ref_model.named_parameters())
    dist_named_params = dict(dist_model.named_parameters())
    assert set(ref_named_params) == set(dist_named_params)
    assert_parameters_are_dtensors(dist_named_params.values())

    offs = torch.tensor([129, 256], dtype=torch.int32, device="cuda")

    for iter_idx in range(2):
        local_input, local_target = get_replicated_local_batch(
            replica_count=dp_mesh.size(),
            replica_index=dp_rank,
            iter_idx=iter_idx,
        )

        ref_optim.zero_grad(set_to_none=True)
        dist_optim.zero_grad(set_to_none=True)

        ref_out = ref_model(local_input, offs)
        dist_out = dist_model(local_input, offs)
        assert_close(dist_out, ref_out, min_sqnr=23.0)

        ref_loss = F.mse_loss(ref_out, local_target)
        dist_loss = F.mse_loss(full_tensor(dist_out), local_target)
        assert_close(dist_loss, ref_loss, atol=1e-2, rtol=1e-2)

        ref_loss.backward()
        dist_loss.backward()
        allreduce_reference_grads(
            ref_model,
            world_size=dp_mesh.size(),
            group=dp_mesh.get_group(),
        )
        assert_dtensor_parameter_grads_match(
            ref_named_params.values(),
            dist_named_params.values(),
            min_sqnr=20.0,
        )

        ref_optim.step()
        dist_optim.step()
        assert_dtensor_parameter_values_match(
            ref_named_params.values(),
            dist_named_params.values(),
            min_sqnr=20.0,
        )


if __name__ == "__main__":
    device_mesh = setup_distributed()
    try:
        _test_blockwise_grouped_mm_fsdp2_tp(device_mesh)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
