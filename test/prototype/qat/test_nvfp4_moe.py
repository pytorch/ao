"""
Tests for NVFP4 QAT tensor subclass approach for MoE models.

The tensor subclass intercepts ``torch._grouped_mm`` and injects FP4 fake
quantization on both activation and weight operands — model-agnostic,
no module swaps required.

See ``test_nvfp4_moe_module_swap.py`` for tests of the older module-swap approach.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.prototype.qat.nvfp4_moe import (
    NVFP4FakeQuantizedScaledGroupedMMTensor,
    _calculate_fp4_global_scale_factor,
    apply_nvfp4_moe_qat,
    remove_nvfp4_moe_qat,
)
from torchao.quantization.utils import compute_error

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from flashinfer import fp4_quantize, nvfp4_block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.utils import get_compute_capability

    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    _has_qwen3_experts = True
except ImportError:
    _has_qwen3_experts = False


def _has_flashinfer_sm100():
    """Check if flashinfer is available and GPU supports SM100+."""
    if not _has_flashinfer:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] >= 10
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ===========================================================================
# Tensor subclass tests
# ===========================================================================


class TestSubclassWrapsAndUnwraps:
    """Test that the tensor subclass wrapping/unwrapping works correctly."""

    def test_wrap_unwrap_roundtrip(self):
        """Wrapping produces the correct subclass type/shape/dtype, and unwrapping recovers the original data."""
        w = torch.randn(4, 64, 32, dtype=torch.bfloat16, device="cuda")
        w_orig = w.clone()
        wrapped = NVFP4FakeQuantizedScaledGroupedMMTensor(w)

        # Wrapping produces the right type, shape, dtype
        assert isinstance(wrapped, NVFP4FakeQuantizedScaledGroupedMMTensor)
        assert wrapped.shape == w.shape
        assert wrapped.dtype == w.dtype

        # Unwrapping recovers the original data
        assert torch.equal(wrapped._data, w_orig)


class TestGroupedMMIntercept:
    """Test that torch._grouped_mm is intercepted with fake quantization."""

    def test_output_shape(self):
        """grouped_mm with a wrapped weight produces the correct output shape."""
        E, K, N = 4, 64, 32
        T = 16
        A = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")
        B_wrapped = NVFP4FakeQuantizedScaledGroupedMMTensor(B)
        # Create offsets: evenly distribute tokens across experts
        offs = torch.tensor(
            [T // E * (i + 1) for i in range(E)],
            dtype=torch.int32,
            device="cuda",
        )
        out = torch._grouped_mm(A, B_wrapped, offs=offs)
        assert out.shape == (T, N)

    def test_fake_quantization_applied(self):
        """Output should differ from unquantized grouped_mm due to fake quant."""
        E, K, N = 4, 64, 32
        T = 16
        torch.manual_seed(42)
        A = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")
        offs = torch.tensor(
            [T // E * (i + 1) for i in range(E)],
            dtype=torch.int32,
            device="cuda",
        )

        # Unquantized reference
        out_ref = torch._grouped_mm(A, B, offs=offs)

        # Quantized via subclass
        B_wrapped = NVFP4FakeQuantizedScaledGroupedMMTensor(B)
        out_fq = torch._grouped_mm(A, B_wrapped, offs=offs)

        # They should have the same shape but different values
        assert out_ref.shape == out_fq.shape
        assert not torch.equal(out_ref, out_fq), (
            "Fake-quantized output should differ from unquantized"
        )


class TestSubclassGradientFlows:
    """Test that gradients flow through the tensor subclass."""

    def test_gradient_on_wrapped_param(self):
        """grouped_mm with a wrapped nn.Parameter produces non-zero gradients for both activation and weight."""
        E, K, N = 4, 64, 32
        T = 16
        torch.manual_seed(42)
        A = torch.randn(T, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        B_data = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")
        B_wrapped = NVFP4FakeQuantizedScaledGroupedMMTensor(B_data)
        B_param = nn.Parameter(B_wrapped)
        offs = torch.tensor(
            [T // E * (i + 1) for i in range(E)],
            dtype=torch.int32,
            device="cuda",
        )
        out = torch._grouped_mm(A, B_param, offs=offs)
        loss = out.sum()
        loss.backward()
        assert A.grad is not None, "Activation gradient is None"
        assert B_param.grad is not None, "Weight gradient is None"
        assert torch.any(B_param.grad != 0), "Weight gradient is all zeros"


class TestApplyRemoveRoundtrip:
    """Test apply/remove QAT on a dummy module with num_experts attribute."""

    def _make_dummy_experts_module(self, E, K, N):
        """Create a simple module that mimics HF *Experts with 3D params."""
        module = nn.Module()
        module.num_experts = E
        module.hidden_dim = K
        module.intermediate_dim = N
        module.is_transposed = False
        module.gate_up_proj = nn.Parameter(
            torch.randn(E, 2 * N, K, dtype=torch.bfloat16, device="cuda")
        )
        module.down_proj = nn.Parameter(
            torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")
        )
        return module

    def test_apply_wraps_3d_params(self):
        """apply_nvfp4_moe_qat wraps all 3D params on modules with num_experts."""
        model = nn.Module()
        model.experts = self._make_dummy_experts_module(4, 64, 32)
        apply_nvfp4_moe_qat(model)
        for param_name in ["gate_up_proj", "down_proj"]:
            param = getattr(model.experts, param_name)
            assert isinstance(
                param.data, NVFP4FakeQuantizedScaledGroupedMMTensor
            ), f"{param_name} not wrapped"

    def test_remove_unwraps_params(self):
        """apply then remove round-trips params back to plain tensors with identical values."""
        model = nn.Module()
        model.experts = self._make_dummy_experts_module(4, 64, 32)
        orig_gate_up = model.experts.gate_up_proj.data.clone()
        orig_down = model.experts.down_proj.data.clone()

        apply_nvfp4_moe_qat(model)
        remove_nvfp4_moe_qat(model)

        for param_name in ["gate_up_proj", "down_proj"]:
            param = getattr(model.experts, param_name)
            assert not isinstance(
                param.data, NVFP4FakeQuantizedScaledGroupedMMTensor
            ), f"{param_name} still wrapped after remove"

        assert torch.equal(model.experts.gate_up_proj.data, orig_gate_up)
        assert torch.equal(model.experts.down_proj.data, orig_down)

    def test_skips_non_expert_params(self):
        """Params on modules without num_experts should not be wrapped."""
        model = nn.Module()
        model.experts = self._make_dummy_experts_module(4, 64, 32)
        model.gate = nn.Linear(64, 4, device="cuda", dtype=torch.bfloat16)

        apply_nvfp4_moe_qat(model)
        assert not isinstance(
            model.gate.weight.data, NVFP4FakeQuantizedScaledGroupedMMTensor
        ), "gate.weight should not be wrapped"


class _DummySwiGLUExperts(nn.Module):
    """Minimal MoE experts module that mirrors the HF ``grouped_mm`` backend.

    Implements: sort tokens by expert -> ``_grouped_mm`` (gate_up_proj) ->
    SwiGLU -> ``_grouped_mm`` (down_proj) -> weighted scatter-back.

    This is the same data flow as ``transformers.integrations.moe.grouped_mm_experts_forward``
    but without the HF dependency, so we can test the tensor subclass interception
    in isolation.
    """

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.is_transposed = False
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_dim, intermediate_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        num_top_k = top_k_index.size(-1)
        device = hidden_states.device

        # Expand tokens for each top-k selection
        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )
        expert_ids = top_k_index.reshape(-1)
        sample_weights = top_k_weights.reshape(-1)

        selected = hidden_states[token_idx]

        # Sort by expert
        perm = torch.argsort(expert_ids)
        inv_perm = torch.argsort(perm)
        selected_g = selected[perm]
        expert_ids_g = expert_ids[perm]
        sample_weights_g = sample_weights[perm]

        # Compute offsets (cumulative token counts per expert)
        counts = torch.histc(
            expert_ids_g.int(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts - 1,
        )
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)

        # Up + gate projection: [S, H] @ [E, 2*I, H]^T -> [S, 2*I]
        up_gate = torch._grouped_mm(selected_g, self.gate_up_proj.transpose(-1, -2), offs=offs)

        # SwiGLU: silu(gate) * up
        gate, up = up_gate.chunk(2, dim=-1)
        act = F.silu(gate) * up

        # Down projection: [S, I] @ [E, H, I]^T -> [S, H]
        down = torch._grouped_mm(act, self.down_proj.transpose(-1, -2), offs=offs)

        # Weighted scatter-back
        weighted = down * sample_weights_g.unsqueeze(-1)
        weighted = weighted[inv_perm]
        out = weighted.view(num_tokens, num_top_k, -1).sum(dim=1)
        return out.to(hidden_states.dtype)


class TestDummyMoEModel:
    """Test apply/remove QAT on a dummy MoE model that calls torch._grouped_mm."""

    def _make_model(self, E=4, H=64, I=32):
        """Build a simple model with a router + experts."""
        model = nn.Module()
        model.gate = nn.Linear(H, E, bias=False)
        model.experts = _DummySwiGLUExperts(E, H, I)
        return model.to(device="cuda", dtype=torch.bfloat16)

    def _make_inputs(self, T=8, H=64, E=4, top_k=2):
        """Generate hidden states and routing indices/weights."""
        hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.stack(
            [torch.randperm(E, device="cuda")[:top_k] for _ in range(T)]
        )
        top_k_weights = torch.randn(
            T, top_k, dtype=torch.bfloat16, device="cuda"
        ).softmax(dim=-1)
        return hidden, top_k_index, top_k_weights

    def test_output_differs_but_close(self):
        """QAT output differs from unquantized (fake quant is active) but stays close (SQNR >= 5 dB)."""
        torch.manual_seed(0)
        model = self._make_model()
        with torch.no_grad():
            model.experts.gate_up_proj.normal_(0, 0.02)
            model.experts.down_proj.normal_(0, 0.02)
        hidden, top_k_index, top_k_weights = self._make_inputs()

        out_ref = model.experts(hidden, top_k_index, top_k_weights)

        apply_nvfp4_moe_qat(model)
        out_qat = model.experts(hidden, top_k_index, top_k_weights)

        assert out_ref.shape == out_qat.shape
        assert not torch.equal(out_ref, out_qat), (
            "QAT output should differ from unquantized due to fake quantization"
        )
        sqnr = compute_error(out_ref, out_qat)
        assert sqnr >= 5.0, f"SQNR {sqnr:.2f} dB too low — QAT output too far from reference"

    def test_gradients_flow_to_all_expert_params(self):
        """All expert parameters should receive non-zero gradients after a QAT forward+backward."""
        torch.manual_seed(0)
        model = self._make_model()
        with torch.no_grad():
            model.experts.gate_up_proj.normal_(0, 0.02)
            model.experts.down_proj.normal_(0, 0.02)
        apply_nvfp4_moe_qat(model)

        hidden, top_k_index, top_k_weights = self._make_inputs()
        out = model.experts(hidden, top_k_index, top_k_weights)
        out.sum().backward()

        for name in ["gate_up_proj", "down_proj"]:
            param = getattr(model.experts, name)
            assert param.grad is not None, f"{name}.grad is None"
            assert torch.any(param.grad != 0), f"{name}.grad is all zeros"

    def test_remove_restores_unquantized_output(self):
        """After remove, the model should produce the same output as the original unquantized model."""
        torch.manual_seed(0)
        model = self._make_model()
        hidden, top_k_index, top_k_weights = self._make_inputs()

        out_before = model.experts(hidden, top_k_index, top_k_weights)

        apply_nvfp4_moe_qat(model)
        remove_nvfp4_moe_qat(model)

        out_after = model.experts(hidden, top_k_index, top_k_weights)
        assert torch.equal(out_before, out_after), (
            "Output should be identical after apply+remove roundtrip"
        )

    def test_router_not_affected(self):
        """The router (gate) linear should NOT be wrapped by apply_nvfp4_moe_qat."""
        model = self._make_model()
        apply_nvfp4_moe_qat(model)
        assert not isinstance(
            model.gate.weight.data, NVFP4FakeQuantizedScaledGroupedMMTensor
        ), "Router weight should not be wrapped"


@pytest.mark.skipif(not _has_qwen3_experts, reason="Requires transformers with Qwen3MoeExperts")
class TestE2EWithQwen3Experts:
    """End-to-end test with real Qwen3MoeExperts module."""

    def test_forward_and_gradients(self):
        """Apply QAT to real Qwen3MoeExperts, verify forward output shape, gradients on all 3D params, and clean unwrap."""
        from transformers import Qwen3MoeConfig

        E, H, I, T, top_k = 4, 64, 32, 8, 2

        config = Qwen3MoeConfig(
            hidden_size=H,
            intermediate_size=I,
            num_experts=E,
            num_experts_per_tok=top_k,
            experts_implementation="grouped_mm",
        )
        experts = Qwen3MoeExperts(config).to(device="cuda", dtype=torch.bfloat16)

        # Wrap with QAT
        wrapper = nn.Module()
        wrapper.experts = experts
        apply_nvfp4_moe_qat(wrapper)

        # Verify wrapping
        for name, param in wrapper.experts.named_parameters(recurse=False):
            if param.ndim == 3:
                assert isinstance(
                    param.data, NVFP4FakeQuantizedScaledGroupedMMTensor
                ), f"{name} not wrapped"

        # Forward pass — Qwen3MoeExperts.forward(hidden_states, top_k_index, top_k_weights)
        hidden = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        top_k_index = torch.stack(
            [torch.randperm(E, device="cuda")[:top_k] for _ in range(T)]
        )
        top_k_weights = torch.randn(
            T, top_k, dtype=torch.bfloat16, device="cuda"
        ).softmax(dim=-1)

        out = wrapper.experts(hidden, top_k_index, top_k_weights)
        assert out.shape == (T, H)

        loss = out.sum()
        loss.backward()

        for name, param in wrapper.experts.named_parameters(recurse=False):
            if param.ndim == 3:
                assert param.grad is not None, f"{name}.grad is None"

        # Unwrap
        remove_nvfp4_moe_qat(wrapper)
        for name, param in wrapper.experts.named_parameters(recurse=False):
            assert not isinstance(
                param.data, NVFP4FakeQuantizedScaledGroupedMMTensor
            ), f"{name} still wrapped"


# ===========================================================================
# Flashinfer kernel comparison: tensor subclass vs fused kernel
# ===========================================================================


# ---- FP4 quantization helpers for kernel comparison ----------------------


def _quant_fp4(a, a_global_sf, is_sf_swizzled_layout=True):
    """Quantize *a* to packed FP4 with a pre-computed global scale."""
    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), 16, False, is_sf_swizzled_layout
    )
    return a_fp4, a_sf, a_global_sf


def _quant_fp4_batches(a, num_experts, is_sf_swizzled_layout=True):
    """Per-expert FP4 quantization (independent global scale per expert)."""
    quant_a, sfs, global_sfs = [], [], []
    for i in range(num_experts):
        g = _calculate_fp4_global_scale_factor(a[i])
        a_fp4, a_sf, _ = _quant_fp4(a[i], g, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(g)
    return torch.stack(quant_a), torch.stack(sfs), torch.stack(global_sfs)


def _prepare_static_weights_for_trtllm_fp4_moe(
    gemm1_weights, gemm2_weights, gemm1_scales, gemm2_scales,
    hidden_size, intermediate_size, num_experts, is_gated_activation,
):
    """Shuffle packed-FP4 weights and interleave block scales for the kernel."""
    epilogue_tile_m = 128
    gemm1_intermediate_size = (
        2 * intermediate_size if is_gated_activation else intermediate_size
    )

    g1_w = gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate_size, hidden_size // 2
    )
    g1_sf = gemm1_scales.view(torch.float8_e4m3fn).reshape(
        num_experts, gemm1_intermediate_size, hidden_size // 16
    )
    g2_w = gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 2
    )
    g2_sf = gemm2_scales.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 16
    )

    cache = {}
    g1ws, g1ss, g2ws, g2ss = [], [], [], []
    for i in range(num_experts):
        pi = _maybe_get_cached_w3_w1_permute_indices(
            cache, g1_w[i].view(torch.uint8), epilogue_tile_m,
            is_gated_act_gemm=is_gated_activation,
        )
        g1ws.append(g1_w[i].view(torch.uint8)[pi.to(g1_w.device)].contiguous())

        pi_sf = _maybe_get_cached_w3_w1_permute_indices(
            cache, g1_sf[i].view(torch.uint8), epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=is_gated_activation,
        )
        g1ss.append(nvfp4_block_scale_interleave(
            g1_sf[i].view(torch.uint8)[pi_sf.to(g1_sf.device)].contiguous()
        ))

        pi = get_w2_permute_indices_with_cache(
            cache, g2_w[i].view(torch.uint8), epilogue_tile_m,
        )
        g2ws.append(g2_w[i].view(torch.uint8)[pi.to(g2_w.device)].contiguous())

        pi_sf = get_w2_permute_indices_with_cache(
            cache, g2_sf[i].view(torch.uint8), epilogue_tile_m,
            num_elts_per_sf=16,
        )
        g2ss.append(nvfp4_block_scale_interleave(
            g2_sf[i].view(torch.uint8)[pi_sf.to(g2_sf.device)].contiguous()
        ))

    g1ws = torch.stack(g1ws)
    g1ss = (
        torch.stack(g1ss).view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_intermediate_size, hidden_size // 16)
    )
    g2ws = torch.stack(g2ws)
    g2ss = (
        torch.stack(g2ss).view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return g1ws, g1ss, g2ws, g2ss


# ---- Routing reference ---------------------------------------------------


def _routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK -> Softmax routing reference (``RoutingMethodType.Renormalize``).

    Returns:
        ``(permuted_buffer_size, expanded_token_idx_to_permuted_idx,
        num_tokens_per_expert, top_k_logits, top_k_indices)``
    """
    device = expert_logits.device
    expert_logits_cpu = expert_logits.cpu()
    num_tokens = expert_logits_cpu.shape[0]

    topk_values, topk_idx = torch.topk(expert_logits_cpu, k=top_k, dim=-1)
    topk_values = F.softmax(topk_values.float(), dim=-1)

    scores = torch.zeros_like(expert_logits_cpu)
    for i in range(num_tokens):
        for j in range(top_k):
            scores[i, topk_idx[i, j]] = topk_values[i, j]

    top_k_logits, top_k_indices = torch.topk(scores, top_k, dim=1)

    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64)
    expanded_token_idx_to_expert = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )
    expanded_token_idx_to_idx_in_expert = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )

    for token_idx in range(num_tokens):
        for k in range(top_k):
            expanded_idx = token_idx * top_k + k
            expert_index = top_k_indices[token_idx, k]
            expanded_token_idx_to_expert[expanded_idx] = expert_index
            expanded_token_idx_to_idx_in_expert[expanded_idx] = (
                num_tokens_per_expert[expert_index]
            )
            num_tokens_per_expert[expert_index] += 1

    padded_prefix_sum = torch.zeros(num_experts + 1, dtype=torch.int64)
    for ii in range(num_experts):
        padded_prefix_sum[ii + 1] = padded_prefix_sum[ii] + (
            (num_tokens_per_expert[ii] + padding - 1) // padding * padding
        )
    permuted_buffer_size = padded_prefix_sum[num_experts].item()

    expanded_token_idx_to_permuted_idx = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expanded_idx = token_idx * top_k + k
            expert = expanded_token_idx_to_expert[expanded_idx]
            offset_in_expert = expanded_token_idx_to_idx_in_expert[expanded_idx]
            permuted_idx = padded_prefix_sum[expert] + offset_in_expert
            expanded_token_idx_to_permuted_idx[expanded_idx] = permuted_idx

    return (
        permuted_buffer_size,
        expanded_token_idx_to_permuted_idx.to(device),
        num_tokens_per_expert.to(device),
        top_k_logits.to(device),
        top_k_indices.to(device),
    )


def _check_accuracy(a, b, atol, rtol, percent):
    """Assert that at least *percent* of elements are close."""
    assert torch.isfinite(a).all(), "Non-finite values in reference output"
    assert torch.isfinite(b).all(), "Non-finite values in kernel output"
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean().item()
    assert match_ratio >= percent, (
        f"Only {match_ratio:.4f} of elements match "
        f"(need {percent:.4f}, atol={atol}, rtol={rtol})"
    )


@pytest.mark.skipif(
    not _has_flashinfer_sm100(),
    reason="Requires flashinfer and SM100+ GPU (Blackwell)",
)
class TestSubclassVsKernel:
    """Compare the tensor subclass MoE forward against the flashinfer
    ``trtllm_fp4_block_scale_moe`` fused kernel.

    Known sources of numerical difference (tensor subclass vs kernel):

    1. **Weight quantization layout**: The subclass sees weights after
       ``.transpose(-1, -2)`` (called by the HF grouped_mm backend), so
       FP4 block-scale boundaries are along a different dimension than the
       kernel's storage-layout quantization.
    2. **Intermediate re-quantization**: The subclass intercepts each
       ``_grouped_mm`` independently, so the SwiGLU output is re-quantized
       at GEMM2. The kernel quantizes it only once.

    Because of these differences, we use SQNR (signal-to-quantization-noise
    ratio) rather than tight element-wise tolerances.
    """

    def test_subclass_vs_kernel_swiglu(self):
        """Standard SwiGLU activation, no bias."""
        E, H, I, T, top_k = 8, 1024, 512, 64, 2
        self._run_comparison(E, H, I, T, top_k)

    # ------------------------------------------------------------------

    def _run_comparison(self, E, H, I, T, top_k):
        from torchao.prototype.qat.nvfp4_moe_module_swap import _run_moe_reference

        padding = 128
        torch.manual_seed(0)

        # ---- 1. Generate random inputs --------------------------------
        hidden_states = 2 * torch.randn(
            T, H, device="cuda", dtype=torch.bfloat16
        )
        gemm1_weights = torch.randn(
            E, 2 * I, H, device="cuda", dtype=torch.bfloat16
        )
        gemm2_weights = torch.randn(
            E, H, I, device="cuda", dtype=torch.bfloat16
        )
        expert_logits = torch.randn(
            T, E, device="cuda", dtype=torch.bfloat16
        )

        # ---- 2. Compute routing (Renormalize: TopK -> Softmax) ---------
        routing_result = _routing_reference_renormalize(
            expert_logits, top_k, E, padding
        )
        (
            permuted_buffer_size,
            expanded_token_idx_to_permuted_idx,
            num_tokens_per_expert,
            top_k_logits,
            top_k_indices,
        ) = routing_result
        permute_info = routing_result[:4]

        # ---- 3. Run tensor subclass forward ----------------------------
        # Wrap weights with the tensor subclass.
        # is_transposed=False matches Qwen3's default: HF transposes the
        # weight before calling _grouped_mm, so the subclass needs to
        # transpose back to storage layout for correct FP4 block boundaries.
        gemm1_w_wrapped = NVFP4FakeQuantizedScaledGroupedMMTensor(
            gemm1_weights, is_transposed=False,
        )
        gemm2_w_wrapped = NVFP4FakeQuantizedScaledGroupedMMTensor(
            gemm2_weights, is_transposed=False,
        )

        # Permute hidden states into expert-sorted buffer (matching kernel routing)
        permuted_hidden = torch.zeros(
            permuted_buffer_size, H, device="cuda", dtype=torch.bfloat16
        )
        token_ids = (
            torch.arange(T, device="cuda")
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        valid = expanded_token_idx_to_permuted_idx >= 0
        permuted_hidden[expanded_token_idx_to_permuted_idx[valid]] = (
            hidden_states[token_ids[valid]]
        )

        # Compute cumulative offsets for _grouped_mm (padded)
        padded_counts = (
            (num_tokens_per_expert + padding - 1) // padding * padding
        )
        offs = torch.cumsum(padded_counts, dim=0).to(torch.int32)

        # GEMM1: [S, H] @ [E, 2*I, H]^T -> [S, 2*I]
        # The tensor subclass intercepts _grouped_mm and applies fake quant
        up_gate = torch._grouped_mm(
            permuted_hidden,
            gemm1_w_wrapped.transpose(-1, -2),
            offs=offs,
        )

        # SwiGLU — kernel layout: first I cols = value, next I cols = gate
        value = up_gate[:, :I]
        gate = up_gate[:, I:]
        act = F.silu(gate) * value

        # GEMM2: [S, I] @ [E, H, I]^T -> [S, H]
        down = torch._grouped_mm(
            act,
            gemm2_w_wrapped.transpose(-1, -2),
            offs=offs,
        )

        # Scatter-back: unpermute and weighted-sum (vectorized)
        k_ids = (
            torch.arange(top_k, device="cuda")
            .unsqueeze(0)
            .expand(T, -1)
            .reshape(-1)
        )
        weights = top_k_logits[token_ids[valid], k_ids[valid]].unsqueeze(1)
        subclass_output = torch.zeros(
            T, H, device="cuda", dtype=torch.float32
        )
        subclass_output.index_add_(
            0,
            token_ids[valid],
            down[expanded_token_idx_to_permuted_idx[valid]].float() * weights.float(),
        )

        # ---- 4. Run module-swap reference to get c_gsf for kernel ------
        _, c_gsf = _run_moe_reference(
            hidden_states.float(),
            permute_info,
            gemm1_weights.float(),
            gemm2_weights.float(),
            E, T, top_k, H, I, padding,
        )

        # ---- 5. Quantize weights to FP4 for kernel ---------------------
        gemm1_fp4, gemm1_sf_lin, gemm1_gsf = _quant_fp4_batches(
            gemm1_weights, E, is_sf_swizzled_layout=False
        )
        gemm2_fp4, gemm2_sf_lin, gemm2_gsf = _quant_fp4_batches(
            gemm2_weights, E, is_sf_swizzled_layout=False
        )

        # ---- 6. Quantize hidden states to FP4 for kernel ---------------
        hs_gsf = _calculate_fp4_global_scale_factor(hidden_states)
        hs_fp4_kern, hs_sf_kern, _ = _quant_fp4(
            hidden_states, hs_gsf, is_sf_swizzled_layout=False
        )
        hs_sf_kern = hs_sf_kern.view(torch.float8_e4m3fn).reshape(T, -1)

        # ---- 7. Shuffle weights and interleave scales for kernel -------
        g1ws, g1ss, g2ws, g2ss = _prepare_static_weights_for_trtllm_fp4_moe(
            gemm1_fp4, gemm2_fp4, gemm1_sf_lin, gemm2_sf_lin,
            H, I, E, is_gated_activation=True,
        )

        # Output scales (combine global scales for the kernel).
        scale_c_fc1 = c_gsf * (1.0 / gemm1_gsf) * (1.0 / hs_gsf)
        scale_gate_fc1 = (1.0 / gemm1_gsf) * (1.0 / hs_gsf)
        scale_c_fc2 = (1.0 / c_gsf) * (1.0 / gemm2_gsf)

        # ---- 8. Call the fused kernel ----------------------------------
        kernel_output = trtllm_fp4_block_scale_moe(
            routing_logits=expert_logits,
            routing_bias=None,
            hidden_states=hs_fp4_kern,
            hidden_states_scale=hs_sf_kern,
            gemm1_weights=g1ws,
            gemm1_weights_scale=g1ss,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=g2ws,
            gemm2_weights_scale=g2ss,
            gemm2_bias=None,
            output1_scale_scalar=scale_c_fc1,
            output1_scale_gate_scalar=scale_gate_fc1,
            output2_scale_scalar=scale_c_fc2,
            num_experts=E,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=1,  # Renormalize (TopK -> Softmax)
            do_finalize=True,
        )
        kernel_output = kernel_output[0].float()

        # ---- 9. Compare -----------------------------------------------
        # With the is_transposed layout fix (quantize in storage layout),
        # we get ~24 dB SQNR and ~95% element-wise match, comparable to
        # the module-swap approach.
        sqnr = compute_error(kernel_output, subclass_output)
        assert sqnr >= 18.0, (
            f"SQNR {sqnr:.2f} dB too low — tensor subclass output too far "
            f"from kernel output"
        )

        _check_accuracy(
            subclass_output, kernel_output,
            atol=0.1, rtol=0.85, percent=0.925,
        )
