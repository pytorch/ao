"""Shared fixtures and utilities for MoE QAT tests."""

import pytest
import torch
from torch import nn

from .reference_moe import MoE, MoEArgs

target_devices = ["cpu"]
if torch.cuda.is_available():
    target_devices.append("cuda")


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(42)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def use_grouped_mm():
    return torch.cuda.is_available()


@pytest.fixture
def moe_model(device, use_grouped_mm):
    dim, hidden_dim = 512, 1024
    args = MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device)


def create_moe_model(device):
    dim, hidden_dim = 1024, 2048
    args = MoEArgs(
        num_experts=8,
        num_shared_experts=2,
        use_grouped_mm=(device == "cuda"),
        load_balance_coeff=None,
    )
    model = MoE(args, dim=dim, hidden_dim=hidden_dim)
    with torch.no_grad():
        for param in model.parameters():
            nn.init.trunc_normal_(param, std=0.5)
    return model.to(device)


def _expert_weight_filter(param, fqn):
    """A `params_filter_fn` for MoEQATConfig: wraps only 3D expert weight parameters,
    skipping 2D parameters (e.g., gate, router, bias)."""
    return param.ndim == 3


def _moe_input(model, batch=2, seq=8):
    """Create input tensor whose last dim matches the model's dim."""
    dim = model.experts.gate_proj.shape[-1]
    return torch.randn(batch, seq, dim, device=model.experts.gate_proj.device)
