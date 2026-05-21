# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
4-GPU FSDP2 + TP smoke coverage for NVFP4Linear.

Run with:
    torchrun --standalone --nproc_per_node=4 -m pytest \
        test/prototype/moe_training/nvfp4_training/test_nvfp4_fsdp2_tp.py -q

Requires SM100 (Blackwell) hardware and 4 GPUs.
"""

import os

import pytest

if os.environ.get("WORLD_SIZE") != "4":
    pytest.skip(
        "Manual 4-GPU torchrun test; not run in regular pytest jobs",
        allow_module_level=True,
    )

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.utils._triton import has_triton

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
    NVFP4ColwiseParallel,
    NVFP4RowwiseParallel,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import NVFP4Linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if not torch.cuda.is_available():
    pytest.skip("Requires CUDA", allow_module_level=True)

if torch.cuda.device_count() < 4:
    pytest.skip("Requires at least 4 CUDA devices", allow_module_level=True)

if not is_sm_at_least_100():
    pytest.skip("Requires SM100+ hardware", allow_module_level=True)


class NVFP4MLP(nn.Module):
    """Small gated MLP using NVFP4Linear layers."""

    def __init__(
        self,
        size: int,
        hidden_size: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.w1 = NVFP4Linear(
            size,
            hidden_size,
            bias=True,
            kernel_preference=KernelPreference.TRITON,
            device=device,
            dtype=dtype,
        )
        self.w2 = NVFP4Linear(
            size,
            hidden_size,
            bias=True,
            kernel_preference=KernelPreference.TRITON,
            device=device,
            dtype=dtype,
        )
        self.out_proj = NVFP4Linear(
            hidden_size,
            size,
            bias=True,
            kernel_preference=KernelPreference.TRITON,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w1(x)) * self.w2(x)
        return self.out_proj(hidden)


def setup_distributed() -> DeviceMesh:
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    assert world_size == 4, (
        f"This test requires WORLD_SIZE=4, got {world_size}. "
        "Run with: torchrun --standalone --nproc_per_node=4 -m pytest "
        "test/prototype/moe_training/nvfp4_training/test_nvfp4_fsdp2_tp.py -q"
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1)
    return init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))


@pytest.fixture(scope="module")
def distributed_env() -> DeviceMesh:
    device_mesh = setup_distributed()
    yield device_mesh
    torch.cuda.synchronize()
    torch._dynamo.reset()
    if dist.is_initialized():
        dist.destroy_process_group()


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _local_batch(
    *,
    dp_rank: int,
    tp_rank: int,
    iter_idx: int,
    dp_size: int,
    tp_size: int,
    m: int,
    k: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(101 + iter_idx)
    x_full = torch.randn(dp_size, m, k, device=device, dtype=torch.bfloat16)
    target_full = torch.randn_like(x_full)
    dist.broadcast(x_full, src=0)
    dist.broadcast(target_full, src=0)

    m_per_tp = m // tp_size
    row_slice = slice(tp_rank * m_per_tp, (tp_rank + 1) * m_per_tp)
    return (
        x_full[dp_rank, row_slice, :].contiguous(),
        target_full[dp_rank, row_slice, :].contiguous(),
    )


def _parallelize_nvfp4_mlp(model: NVFP4MLP, tp_mesh: DeviceMesh) -> NVFP4MLP:
    return parallelize_module(
        model,
        tp_mesh,
        {
            "w1": NVFP4ColwiseParallel(use_local_output=False),
            "w2": NVFP4ColwiseParallel(use_local_output=False),
            "out_proj": NVFP4RowwiseParallel(),
        },
    )


def _test_nvfp4_mlp_fsdp2_tp_smoke(
    distributed_env: DeviceMesh,
    *,
    compile_model: bool,
) -> None:
    mesh = distributed_env
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    dp_rank, tp_rank = mesh.get_coordinate()
    device = mesh.device_type
    M, K, H = 512, 256, 512

    model = NVFP4MLP(K, H, device=device, dtype=torch.bfloat16)
    model = _parallelize_nvfp4_mlp(model, tp_mesh)
    if compile_model:
        prepare_for_cuda_graph(
            torch.device(device),
            sign_vectors=(
                model.w1.rht_sign_vector,
                model.w2.rht_sign_vector,
                model.out_proj.rht_sign_vector,
            ),
        )
        model = torch.compile(model, mode="reduce-overhead")
    model = fully_shard(model, mesh=dp_mesh)

    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    expected_shape = (M // tp_mesh.size(), K)

    for iter_idx in range(2):
        x, target = _local_batch(
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            iter_idx=iter_idx,
            dp_size=dp_mesh.size(),
            tp_size=tp_mesh.size(),
            m=M,
            k=K,
            device=device,
        )

        optim.zero_grad(set_to_none=True)
        out = model(x)
        assert out.shape == expected_shape
        assert out.dtype == torch.bfloat16
        assert not out.isnan().any(), "FSDP2+TP output contains NaN"

        loss = F.mse_loss(out.float(), target.float())
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name}.grad is None"
            grad = _local_tensor(param.grad)
            assert not grad.isnan().any(), f"{name}.grad contains NaN"

        optim.step()
        torch.cuda.synchronize()


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_nvfp4_mlp_fsdp2_tp_smoke(distributed_env: DeviceMesh):
    _test_nvfp4_mlp_fsdp2_tp_smoke(distributed_env, compile_model=False)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="torch.compile requires PyTorch 2.10+"
)
def test_nvfp4_mlp_fsdp2_tp_cuda_graph_compile_smoke(distributed_env: DeviceMesh):
    _test_nvfp4_mlp_fsdp2_tp_smoke(distributed_env, compile_model=True)
