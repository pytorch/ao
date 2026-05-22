"""Shared fixtures and utilities for MoE QAT tests."""

import pytest
import torch
from torch import nn

from .reference_moe import MoE, MoEArgs


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


@pytest.fixture
def weight_config():
    from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
    from torchao.quantization.granularity import PerRow
    return Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())


def _expert_weight_filter(param, fqn):
    """A `params_filter_fn` for MoEQATConfig: wraps only 3D expert weight parameters,
    skipping 2D parameters (e.g., gate, router, bias)."""
    return param.ndim == 3


def _moe_input(model, batch=2, seq=8):
    """Create input tensor whose last dim matches the model's dim."""
    dim = model.experts.w1.shape[-1]
    return torch.randn(batch, seq, dim, device=model.experts.w1.device)
