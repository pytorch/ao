"""Shared fixtures and utilities for MoE QAT tests."""

import os

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from .reference_moe import MoE, MoEArgs

target_devices = ["cpu"]
if torch.cuda.is_available():
    target_devices.append("cuda")


@pytest.fixture(scope="module", params=target_devices)
def distributed_env(request):
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == 4, (
        f"This test requires world_size=4, but got world_size={world_size}. "
        "Run with: torchrun --nproc_per_node=4 -m pytest"
    )

    torch.manual_seed(42)
    device_type = request.param

    tp_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("tp",))
    ep_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("ep",))
    dp_mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
    ep_tp_mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("ep", "tp"))
    dp_tp_mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("dp", "tp"))

    yield {
        "tp_mesh": tp_mesh,
        "ep_mesh": ep_mesh,
        "dp_mesh": dp_mesh,
        "ep_tp_mesh": ep_tp_mesh,
        "dp_tp_mesh": dp_tp_mesh,
        "device_type": device_type,
    }

    torch.distributed.destroy_process_group()


def create_moe_model(device, use_grouped_mm, dtype=torch.float32):
    dim, hidden_dim = 1024, 2048
    args = MoEArgs(
        num_experts=8,
        num_shared_experts=2,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device=device, dtype=dtype)


def _expert_weight_filter(param, fqn):
    """A `params_filter_fn` for MoEQATConfig: wraps only 3D expert weight parameters,
    skipping 2D parameters (e.g., gate, router, bias)."""
    return param.ndim == 3


def _moe_input(model, batch=2, seq=8):
    """Create input tensor whose last dim matches the model's dim."""
    dim = model.experts.gate_proj.shape[-1]
    device=model.experts.gate_proj.device
    dtype=model.experts.gate_proj.dtype
    return torch.randn(batch, seq, dim, device=device, dtype=dtype)
