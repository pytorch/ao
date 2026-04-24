# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
4-GPU FSDP2 + TP parity for Float8BlockwiseLinear.

This is intended as a manual torchrun test for a 2D ``(dp, tp) = (2, 2)`` mesh:

    NCCL_SOCKET_IFNAME=lo torchrun --standalone --nproc_per_node=4 \
        test/prototype/blockwise_fp8_training/test_fsdp2_tp.py

Note: for now, this does not run in CI.
"""

import os

import pytest

# This file is a manual 4-GPU torchrun test. Skip module import entirely when
# pytest collects it in regular single-process jobs.
if os.environ.get("WORLD_SIZE") != "4":
    pytest.skip(
        "Manual 4-GPU torchrun test; not run in regular pytest jobs",
        allow_module_level=True,
    )

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn import functional as F

try:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
except ImportError:
    pytest.skip(
        "Tensor parallel APIs require a newer torch build",
        allow_module_level=True,
    )

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from _distributed_test_utils import (
    allreduce_reference_grads,
    assert_close,
    assert_dtensor_parameter_grads_match,
    assert_dtensor_parameter_values_match,
    assert_parameters_are_dtensors,
    get_blockwise_linear_skip_reason,
    get_replicated_local_batch,
    make_quantized_toy_model_pair,
)

if skip_reason := get_blockwise_linear_skip_reason(
    triton_module=triton,
    min_cuda_devices=4,
):
    pytest.skip(skip_reason, allow_module_level=True)

torch.set_float32_matmul_precision("high")


def setup_distributed() -> DeviceMesh:
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    assert world_size == 4, (
        f"This test requires WORLD_SIZE=4, got {world_size}. "
        "Run with: torchrun --standalone --nproc_per_node=4 "
        "test/prototype/blockwise_fp8_training/test_fsdp2_tp.py"
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    torch.manual_seed(1)
    return device_mesh


def _test_blockwise_mlp_fsdp2_tp_parity(mesh: DeviceMesh, size: int = 128) -> None:
    tp_plan = {
        "ffn.w1": ColwiseParallel(),
        "ffn.w2": ColwiseParallel(),
        "ffn.out_proj": RowwiseParallel(),
    }
    dp_rank, _tp_rank = mesh.get_coordinate()
    dp_mesh = mesh["dp"]

    for use_triton in (False, True):
        ref_model, dist_model = make_quantized_toy_model_pair(
            size=size,
            use_triton=use_triton,
            broadcast_weights=True,
        )

        dist_model = parallelize_module(dist_model, mesh["tp"], tp_plan)
        dist_model = fully_shard(dist_model, mesh=dp_mesh)

        ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
        dist_optim = torch.optim.SGD(dist_model.parameters(), lr=1e-2)

        ref_named_params = dict(ref_model.named_parameters())
        dist_named_params = dict(dist_model.named_parameters())

        assert set(ref_named_params) == set(dist_named_params)
        assert_parameters_are_dtensors(dist_named_params.values())

        for iter_idx in range(2):
            local_input, local_target = get_replicated_local_batch(
                replica_count=dp_mesh.size(),
                replica_index=dp_rank,
                iter_idx=iter_idx,
                size=size,
            )

            ref_optim.zero_grad(set_to_none=True)
            dist_optim.zero_grad(set_to_none=True)

            ref_out = ref_model(local_input)
            dist_out = dist_model(local_input)
            assert_close(dist_out, ref_out)

            ref_loss = F.mse_loss(ref_out, local_target)
            dist_loss = F.mse_loss(dist_out, local_target)
            assert_close(dist_loss, ref_loss, atol=1e-3, rtol=1e-3)

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
            )

            ref_optim.step()
            dist_optim.step()

            assert_dtensor_parameter_values_match(
                ref_named_params.values(),
                dist_named_params.values(),
            )


if __name__ == "__main__":
    device_mesh = setup_distributed()
    try:
        _test_blockwise_mlp_fsdp2_tp_parity(device_mesh)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
