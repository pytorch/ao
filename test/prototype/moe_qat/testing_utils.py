"""Shared fixtures and utilities for MoE QAT tests."""

import pytest
import torch
from torch import nn

from .reference_moe import MoE, MoEArgs

target_devices = ["cpu"]
if torch.cuda.is_available():
    target_devices.append("cuda")


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
