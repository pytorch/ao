"""
Tests for NVFP4 QAT module-swap approach for MoE models.

Includes:
- Unit tests for the module-swap QAT fake-quantization path (no flashinfer dependency).
- Reference-vs-kernel comparison tests that verify a decomposed PyTorch
  reference matches the flashinfer ``trtllm_fp4_block_scale_moe`` fused
  kernel (requires flashinfer + SM100+ GPU).
"""

import pytest
import torch
import torch.nn.functional as F

from torchao.prototype.qat.nvfp4_moe_module_swap import (
    NVFP4FakeQuantizedMoE,
    _calculate_fp4_global_scale_factor,
    _fp4_fake_quantize,
    _run_moe_reference,
)
from torchao.quantization.utils import compute_error

# ---------------------------------------------------------------------------
# Optional flashinfer imports (for reference-vs-kernel tests)
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


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_routing(num_tokens, num_experts, top_k, device):
    """Create synthetic routing inputs."""
    router_indices = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    )  # [T, top_k]
    routing_weights = torch.randn(num_tokens, num_experts, device=device).softmax(
        dim=-1
    )
    return router_indices, routing_weights


# ===========================================================================
# Module-swap tests
# ===========================================================================


@pytest.mark.skipif(not _has_flashinfer, reason="Requires flashinfer")
class TestFakeQuantizeRoundtrip:
    def test_shape_dtype_sqnr_and_ste(self):
        """FP4 fake quantize preserves shape/dtype, has bounded noise (SQNR >= 8 dB), and STE passes gradients through as identity."""
        torch.manual_seed(42)
        x = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_fq = _fp4_fake_quantize(x)

        # Shape and dtype preserved
        assert x_fq.shape == x.shape
        assert x_fq.dtype == x.dtype

        # Quantization noise is bounded
        sqnr = compute_error(x, x_fq)
        assert sqnr >= 8.0, f"SQNR {sqnr:.2f} dB below threshold 8.0 dB"

        # STE: gradient should be all ones (from sum)
        x_fq.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


@pytest.mark.skipif(not _has_flashinfer, reason="Requires flashinfer")
class TestGradientFlow:
    def test_all_params_get_gradients(self):
        """Module-swap NVFP4FakeQuantizedMoE produces non-zero gradients for both gemm1 and gemm2 weights."""
        E, H, I, T, top_k = 4, 64, 64, 16, 2

        torch.manual_seed(42)
        module = NVFP4FakeQuantizedMoE(E, H, I).to(device="cuda", dtype=torch.bfloat16)
        # Initialize with small random weights
        with torch.no_grad():
            module.gemm1_weight.normal_(0, 0.02)
            module.gemm2_weight.normal_(0, 0.02)

        hidden = torch.randn(1, T, H, dtype=torch.bfloat16, device="cuda")
        router_indices, routing_weights = _make_routing(T, E, top_k, "cuda")

        output = module(
            hidden, router_indices=router_indices, routing_weights=routing_weights
        )
        loss = output.sum()
        loss.backward()

        for name in ["gemm1_weight", "gemm2_weight"]:
            param = getattr(module, name)
            assert param.grad is not None, f"{name}.grad is None"
            assert torch.any(param.grad != 0), f"{name}.grad is all zeros"


# ===========================================================================
# Flashinfer-dependent test utilities for TestReferenceVsKernel
# ===========================================================================


# ---- FP4 quantization helpers -------------------------------------------


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


def prepare_static_weights_for_trtllm_fp4_moe(
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_scales: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    is_gated_activation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            cache,
            g1_w[i].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=is_gated_activation,
        )
        g1ws.append(g1_w[i].view(torch.uint8)[pi.to(g1_w.device)].contiguous())

        pi_sf = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            g1_sf[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=is_gated_activation,
        )
        g1ss.append(
            nvfp4_block_scale_interleave(
                g1_sf[i].view(torch.uint8)[pi_sf.to(g1_sf.device)].contiguous()
            )
        )

        pi = get_w2_permute_indices_with_cache(
            cache,
            g2_w[i].view(torch.uint8),
            epilogue_tile_m,
        )
        g2ws.append(g2_w[i].view(torch.uint8)[pi.to(g2_w.device)].contiguous())

        pi_sf = get_w2_permute_indices_with_cache(
            cache,
            g2_sf[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        g2ss.append(
            nvfp4_block_scale_interleave(
                g2_sf[i].view(torch.uint8)[pi_sf.to(g2_sf.device)].contiguous()
            )
        )

    g1ws = torch.stack(g1ws)
    g1ss = (
        torch.stack(g1ss)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, gemm1_intermediate_size, hidden_size // 16)
    )
    g2ws = torch.stack(g2ws)
    g2ss = (
        torch.stack(g2ss)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, hidden_size, intermediate_size // 16)
    )
    return g1ws, g1ss, g2ws, g2ss


# ---- Routing reference ---------------------------------------------------


def _routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK -> Softmax routing reference (``RoutingMethodType.Renormalize``).

    1. Select the top-*k* experts per token.
    2. Softmax-normalise the selected logits (only across the *k* chosen).
    3. Build the permutation tables that map (token, k) -> position in the
       padded, expert-sorted buffer.

    Returns:
        ``(permuted_buffer_size, expanded_token_idx_to_permuted_idx,
        num_tokens_per_expert, top_k_logits, top_k_indices)``
    """
    device = expert_logits.device
    expert_logits_cpu = expert_logits.cpu()
    num_tokens = expert_logits_cpu.shape[0]

    # Step 1: TopK -> Softmax normalisation
    topk_values, topk_idx = torch.topk(expert_logits_cpu, k=top_k, dim=-1)
    topk_values = F.softmax(topk_values.float(), dim=-1)

    # Build full-expert scores with only the selected entries non-zero.
    scores = torch.zeros_like(expert_logits_cpu)
    for i in range(num_tokens):
        for j in range(top_k):
            scores[i, topk_idx[i, j]] = topk_values[i, j]

    # Step 2: Compute permutation from the (sparse) score tensor.
    top_k_logits, top_k_indices = torch.topk(scores, top_k, dim=1)

    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64)
    expanded_token_idx_to_expert = -torch.ones(num_tokens * top_k, dtype=torch.int64)
    expanded_token_idx_to_idx_in_expert = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )

    for token_idx in range(num_tokens):
        for k in range(top_k):
            expanded_idx = token_idx * top_k + k
            expert_index = top_k_indices[token_idx, k]
            expanded_token_idx_to_expert[expanded_idx] = expert_index
            expanded_token_idx_to_idx_in_expert[expanded_idx] = num_tokens_per_expert[
                expert_index
            ]
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


# ---- Accuracy check -------------------------------------------------------


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


# ---- Kernel comparison tests ----------------------------------------------


@pytest.mark.skipif(
    not _has_flashinfer_sm100(),
    reason="Requires flashinfer and SM100+ GPU (Blackwell)",
)
class TestReferenceVsKernel:
    """Compare the decomposed-PyTorch module-swap reference against the flashinfer
    ``trtllm_fp4_block_scale_moe`` fused kernel."""

    def test_swiglu(self):
        """Standard SwiGLU activation, no bias."""
        E, H, I, T, top_k = 8, 1024, 512, 64, 2
        self._run_comparison(E, H, I, T, top_k)

    # ------------------------------------------------------------------

    def _run_comparison(self, E, H, I, T, top_k):
        padding = 128
        torch.manual_seed(0)

        # ---- 1. Generate random inputs --------------------------------
        hidden_states = 2 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        gemm1_weights = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16)
        gemm2_weights = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16)
        expert_logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

        # ---- 2. Compute routing (Renormalize: TopK -> Softmax) ---------
        routing_result = _routing_reference_renormalize(
            expert_logits, top_k, E, padding
        )
        # permute_info is the first 4 elements; top_k_indices is test-only
        permute_info = routing_result[:4]

        # ---- 3. Quantize weights to FP4 for kernel ---------------------
        gemm1_fp4, gemm1_sf_lin, gemm1_gsf = _quant_fp4_batches(
            gemm1_weights, E, is_sf_swizzled_layout=False
        )
        gemm2_fp4, gemm2_sf_lin, gemm2_gsf = _quant_fp4_batches(
            gemm2_weights, E, is_sf_swizzled_layout=False
        )

        # ---- 4. Quantize hidden states to FP4 for kernel ---------------
        hs_gsf = _calculate_fp4_global_scale_factor(hidden_states)
        hs_fp4_kern, hs_sf_kern, _ = _quant_fp4(
            hidden_states, hs_gsf, is_sf_swizzled_layout=False
        )
        hs_sf_kern = hs_sf_kern.view(torch.float8_e4m3fn).reshape(T, -1)

        # ---- 5. Run reference (handles fake quantization internally) ---
        ref_output, c_gsf = _run_moe_reference(
            hidden_states.float(),
            permute_info,
            gemm1_weights.float(),
            gemm2_weights.float(),
            E,
            T,
            top_k,
            H,
            I,
            padding,
        )

        # ---- 6. Shuffle weights and interleave scales for kernel -------
        g1ws, g1ss, g2ws, g2ss = prepare_static_weights_for_trtllm_fp4_moe(
            gemm1_fp4,
            gemm2_fp4,
            gemm1_sf_lin,
            gemm2_sf_lin,
            H,
            I,
            E,
            is_gated_activation=True,
        )

        # Output scales (combine global scales for the kernel).
        scale_c_fc1 = c_gsf * (1.0 / gemm1_gsf) * (1.0 / hs_gsf)
        scale_gate_fc1 = (1.0 / gemm1_gsf) * (1.0 / hs_gsf)
        scale_c_fc2 = (1.0 / c_gsf) * (1.0 / gemm2_gsf)

        # ---- 7. Call the fused kernel ----------------------------------
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

        # ---- 8. Compare -----------------------------------------------
        _check_accuracy(
            ref_output,
            kernel_output,
            atol=0.1,
            rtol=0.85,
            percent=0.925,
        )
