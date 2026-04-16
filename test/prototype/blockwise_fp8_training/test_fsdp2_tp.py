# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
4-GPU FSDP2 + TP parity for Float8BlockwiseLinear.

This is intended as a manual torchrun test for a 2D ``(dp, tp) = (2, 2)`` mesh:

    NCCL_SOCKET_IFNAME=lo torchrun --nproc_per_node=4 \
        test/prototype/blockwise_fp8_training/test_fsdp2_tp.py

Note: for now, this does not run in CI.
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import DTensor
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

from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseLinear,
    Float8BlockwiseLinearConfig,
)
from torchao.quantization import quantize_
from torchao.testing.training.dtensor_utils import ToyModel
from torchao.utils import is_sm_at_least_90

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

if torch.cuda.device_count() < 4:
    pytest.skip("Need at least 4 CUDA devices", allow_module_level=True)

if not is_sm_at_least_90():
    pytest.skip(
        "Float8BlockwiseLinear currently requires CUDA SM90+",
        allow_module_level=True,
    )

if version.parse(triton.__version__) < version.parse("3.3.0"):
    pytest.skip("Triton version < 3.3.0", allow_module_level=True)

torch.set_float32_matmul_precision("high")


def setup_distributed() -> DeviceMesh:
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    assert world_size == 4, (
        f"This test requires WORLD_SIZE=4, got {world_size}. "
        "Run with: torchrun --nproc_per_node=4 "
        "test/prototype/blockwise_fp8_training/test_fsdp2_tp.py"
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    torch.manual_seed(1)
    return device_mesh


def _broadcast_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param, src=0)


def _init_model(size: int = 128) -> torch.nn.Module:
    torch.manual_seed(42)
    model = ToyModel(size).cuda().to(torch.bfloat16)
    _broadcast_module(model)
    return model


def _set_use_triton(model: torch.nn.Module, use_triton: bool) -> None:
    converted = 0
    for module in model.modules():
        if isinstance(module, Float8BlockwiseLinear):
            module.use_triton = use_triton
            converted += 1
    assert converted > 0


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )


def _get_local_batch(
    mesh: DeviceMesh,
    iter_idx: int,
    size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    dp_rank, _tp_rank = mesh.get_coordinate()
    torch.manual_seed(100 + iter_idx)
    global_input = torch.randn(
        mesh["dp"].size(),
        1,
        size,
        size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    global_target = torch.randn_like(global_input)
    dist.broadcast(global_input, src=0)
    dist.broadcast(global_target, src=0)
    return global_input[dp_rank].contiguous(), global_target[dp_rank].contiguous()


def _allreduce_reference_grads(model: torch.nn.Module, dp_mesh: DeviceMesh) -> None:
    dp_group = dp_mesh.get_group()
    for param in model.parameters():
        assert param.grad is not None
        dist.all_reduce(param.grad, group=dp_group)
        param.grad.div_(dp_mesh.size())


def _test_blockwise_mlp_fsdp2_tp_parity(mesh: DeviceMesh, size: int = 128) -> None:
    tp_plan = {
        "ffn.w1": ColwiseParallel(),
        "ffn.w2": ColwiseParallel(),
        "ffn.out_proj": RowwiseParallel(),
    }

    for use_triton in (False, True):
        ref_model = _init_model(size)
        dist_model = copy.deepcopy(ref_model)

        quantize_(ref_model, Float8BlockwiseLinearConfig())
        quantize_(dist_model, Float8BlockwiseLinearConfig())
        _set_use_triton(ref_model, use_triton)
        _set_use_triton(dist_model, use_triton)

        dist_model = parallelize_module(dist_model, mesh["tp"], tp_plan)
        dist_model = fully_shard(dist_model, mesh=mesh["dp"])

        ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
        dist_optim = torch.optim.SGD(dist_model.parameters(), lr=1e-2)

        ref_named_params = dict(ref_model.named_parameters())
        dist_named_params = dict(dist_model.named_parameters())

        assert set(ref_named_params) == set(dist_named_params)
        for param in dist_named_params.values():
            assert isinstance(param, DTensor)

        for iter_idx in range(2):
            local_input, local_target = _get_local_batch(mesh, iter_idx, size)

            ref_optim.zero_grad(set_to_none=True)
            dist_optim.zero_grad(set_to_none=True)

            ref_out = ref_model(local_input)
            dist_out = dist_model(local_input)
            _assert_close(dist_out, ref_out)

            ref_loss = F.mse_loss(ref_out, local_target)
            dist_loss = F.mse_loss(dist_out, local_target)
            _assert_close(dist_loss, ref_loss, atol=1e-3, rtol=1e-3)

            ref_loss.backward()
            dist_loss.backward()
            _allreduce_reference_grads(ref_model, mesh["dp"])

            for name, ref_param in ref_named_params.items():
                dist_param = dist_named_params[name]
                assert ref_param.grad is not None
                assert dist_param.grad is not None
                assert isinstance(dist_param.grad, DTensor)
                _assert_close(_full_tensor(dist_param.grad), ref_param.grad)

            ref_optim.step()
            dist_optim.step()

            for name, ref_param in ref_named_params.items():
                _assert_close(_full_tensor(dist_named_params[name]), ref_param)


if __name__ == "__main__":
    device_mesh = setup_distributed()
    try:
        _test_blockwise_mlp_fsdp2_tp_parity(device_mesh)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
