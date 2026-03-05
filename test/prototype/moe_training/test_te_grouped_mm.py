# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for TransformerEngine grouped GEMM integration (KernelPreference.TE).

These tests validate:
  1. Custom op registration and fake (meta) implementations
  2. Forward + backward numerical correctness vs BF16 grouped MM reference
  3. Integration through _to_mxfp8_then_scaled_grouped_mm with KernelPreference.TE
  4. Model conversion via MXFP8TrainingOpConfig with MXFP8_TE recipe
  5. torch.compile compatibility (custom ops have fake impls)
"""

import pytest
import torch
from torch.nn import functional as F

from torchao.utils import torch_version_at_least

if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
):
    pytest.skip("Unsupported PyTorch version or hardware", allow_module_level=True)

try:
    from torchao.prototype.moe_training.te_grouped_mm import (
        _offs_to_m_splits,
        te_gemm_dgrad,
        te_gemm_fwd,
        te_gemm_wgrad,
    )
except ImportError:
    pytest.skip(
        "TransformerEngine not available, skipping TE grouped MM tests",
        allow_module_level=True,
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.config import (
    MXFP8TrainingOpConfig,
    MXFP8TrainingRecipe,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.quantization.quantize_.common import KernelPreference

torch._dynamo.config.cache_size_limit = 1000


# ──────────────────────────────────────────────────────────────────────────────
# Test custom ops directly
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (2048, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (4, 8))
@pytest.mark.parametrize("use_fp8", (True, False))
def test_te_gemm_fwd(M, K, N, num_experts, use_fp8):
    """Forward pass: compare TE grouped GEMM output vs BF16 reference."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)

    ref_out = torch._grouped_mm(
        x, w_t, offs=offs, out_dtype=torch.bfloat16
    )

    te_out = te_gemm_fwd(x, w_t, offs, torch.bfloat16, use_fp8)

    if use_fp8:
        sqnr = compute_error(ref_out, te_out)
        assert sqnr >= 25.0, f"Forward sqnr {sqnr} too low for MXFP8 mode"
    else:
        torch.testing.assert_close(ref_out, te_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024)])
@pytest.mark.parametrize("num_experts", (4,))
@pytest.mark.parametrize("use_fp8", (True, False))
def test_te_gemm_dgrad(M, K, N, num_experts, use_fp8):
    """DGRAD: compare TE dgrad output vs BF16 reference."""
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)

    ref_grad_input = torch._grouped_mm(
        grad_output,
        w_t.transpose(-2, -1),
        offs=offs,
        out_dtype=torch.bfloat16,
    )

    te_grad_input = te_gemm_dgrad(grad_output, w_t, offs, torch.bfloat16, use_fp8)

    if use_fp8:
        sqnr = compute_error(ref_grad_input, te_grad_input)
        assert sqnr >= 25.0, f"DGRAD sqnr {sqnr} too low for MXFP8 mode"
    else:
        torch.testing.assert_close(
            ref_grad_input, te_grad_input, atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024)])
@pytest.mark.parametrize("num_experts", (4,))
@pytest.mark.parametrize("use_fp8", (True, False))
def test_te_gemm_wgrad(M, K, N, num_experts, use_fp8):
    """WGRAD: compare TE wgrad output vs BF16 reference."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    grad_output = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)

    ref_wgrad = torch._grouped_mm(
        grad_output.transpose(-2, -1),
        x,
        offs=offs,
        out_dtype=torch.bfloat16,
    ).transpose(-2, -1)

    te_wgrad = te_gemm_wgrad(x, grad_output, offs, torch.bfloat16, use_fp8)

    if use_fp8:
        sqnr = compute_error(ref_wgrad, te_wgrad)
        assert sqnr >= 20.0, f"WGRAD sqnr {sqnr} too low for MXFP8 mode"
    else:
        torch.testing.assert_close(ref_wgrad, te_wgrad, atol=1e-2, rtol=1e-2)


# ──────────────────────────────────────────────────────────────────────────────
# Test integration through _to_mxfp8_then_scaled_grouped_mm
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (2048, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (4, 8))
def test_te_integrated_fwd_bwd(M, K, N, num_experts):
    """Full forward + backward through the torchao MXFP8 autograd function
    with KernelPreference.TE, compared to BF16 reference."""
    block_size = 32
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    w_t = w.transpose(-2, -1).requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)

    x_ref = x.clone().detach().requires_grad_(True)
    w_t_ref = w_t.clone().detach().requires_grad_(True)
    offs_ref = offs.clone()

    # Forward
    out = _to_mxfp8_then_scaled_grouped_mm(
        x,
        w_t,
        offs=offs,
        block_size=block_size,
        kernel_preference=KernelPreference.TE,
    )
    ref_out = torch._grouped_mm(
        x_ref, w_t_ref, offs=offs_ref, out_dtype=torch.bfloat16
    )

    sqnr = compute_error(ref_out, out)
    assert sqnr >= 25.0, f"Output sqnr {sqnr} too low"

    # Backward
    labels = torch.ones_like(ref_out)
    F.mse_loss(ref_out, labels).backward()
    F.mse_loss(out, labels).backward()

    sqnr_input_grad = compute_error(x_ref.grad, x.grad)
    assert sqnr_input_grad >= 22.0, (
        f"Input grad sqnr {sqnr_input_grad} too low"
    )

    sqnr_weight_grad = compute_error(w_t_ref.grad, w_t.grad)
    assert sqnr_weight_grad >= 20.0, (
        f"Weight grad sqnr {sqnr_weight_grad} too low"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test config / recipe
# ──────────────────────────────────────────────────────────────────────────────


def test_mxfp8_te_recipe():
    """Verify MXFP8_TE recipe creates correct config."""
    config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_TE)
    assert config.kernel_preference == KernelPreference.TE
    assert config.out_dtype == torch.bfloat16
    assert config.wgrad_with_hp is False


# ──────────────────────────────────────────────────────────────────────────────
# Test model conversion
# ──────────────────────────────────────────────────────────────────────────────


def test_te_model_conversion():
    """Test that quantize_() with MXFP8_TE recipe correctly swaps parameters
    and that forward + backward produce valid gradients."""
    from torchao.prototype.moe_training.tensor import (
        MXFP8TrainingWeightWrapperTensor,
    )
    from torchao.quantization.quant_api import quantize_

    num_experts = 4
    K, N = 256, 512

    class SimpleMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.Linear(K, N, bias=False)
            self.gate_weight = torch.nn.Parameter(
                torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
            )

        def forward(self, x, offs):
            return torch._grouped_mm(
                x, self.gate_weight, offs=offs, out_dtype=torch.bfloat16
            )

    model = SimpleMoE().to("cuda", torch.bfloat16)

    config = MXFP8TrainingOpConfig.from_recipe(MXFP8TrainingRecipe.MXFP8_TE)

    def filter_fn(mod, fqn):
        return "gate" in fqn or isinstance(mod, SimpleMoE)

    quantize_(model, config=config, filter_fn=filter_fn)

    assert isinstance(model.gate_weight.data, MXFP8TrainingWeightWrapperTensor), (
        "gate_weight should be wrapped in MXFP8TrainingWeightWrapperTensor"
    )
    assert model.gate_weight.data.config.kernel_preference == KernelPreference.TE

    M = 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=32)

    out = model(x, offs)
    assert out.shape == (M, N)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert model.gate_weight.grad is not None


# ──────────────────────────────────────────────────────────────────────────────
# Test torch.compile compatibility (fake impls)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024)])
@pytest.mark.parametrize("num_experts", (4,))
def test_te_compile(M, K, N, num_experts):
    """Verify that torch.compile traces through TE custom ops without graph breaks."""
    block_size = 32
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda")
    w_t = w.transpose(-2, -1).requires_grad_(True)
    offs = generate_jagged_offs(num_experts, M, multiple_of=block_size)

    compiled_fn = torch.compile(
        _to_mxfp8_then_scaled_grouped_mm, fullgraph=True
    )
    out = compiled_fn(
        x,
        w_t,
        offs=offs,
        block_size=block_size,
        kernel_preference=KernelPreference.TE,
    )
    assert out.shape == (M, N)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert w_t.grad is not None


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_te_fwd_padding_path():
    """Forward with non-32-aligned per-expert token counts exercises the
    internal pad/unpad logic.  Output should still match the expected shape
    and contain no NaN/Inf."""
    num_experts, K, N = 4, 256, 512
    m_splits = [17, 45, 3, 63]
    M = sum(m_splits)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")

    cumulative = torch.tensor(m_splits, device="cuda", dtype=torch.int32).cumsum(0)

    te_out = te_gemm_fwd(x, w_t, cumulative, torch.bfloat16, True)
    assert te_out.shape == (M, N), f"Expected ({M}, {N}), got {te_out.shape}"
    assert not torch.isnan(te_out).any(), "NaN in padded forward output"
    assert not torch.isinf(te_out).any(), "Inf in padded forward output"


def test_te_fwd_zero_token_expert():
    """An expert receiving zero tokens should not cause errors."""
    num_experts, K, N = 4, 256, 512
    m_splits = [32, 0, 64, 32]
    M = sum(m_splits)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")

    cumulative = torch.tensor(m_splits, device="cuda", dtype=torch.int32).cumsum(0)

    te_out = te_gemm_fwd(x, w_t, cumulative, torch.bfloat16, True)
    assert te_out.shape == (M, N)
    assert not torch.isnan(te_out).any(), "NaN with zero-token expert"


def test_m_splits_caching():
    """_offs_to_m_splits should cache the result on the tensor object so
    that repeated calls (fwd, dgrad, wgrad) don't re-sync GPU→CPU."""
    offs = torch.tensor([32, 96, 128], device="cuda", dtype=torch.int32)

    splits1 = _offs_to_m_splits(offs)
    assert splits1 == [32, 64, 32]

    assert hasattr(offs, "_cached_m_splits"), "Cache attribute not set"

    splits2 = _offs_to_m_splits(offs)
    assert splits1 is splits2, "Second call should return the cached object"
