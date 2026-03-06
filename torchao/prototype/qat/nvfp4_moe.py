"""
FP4 QAT for MoE (Mixture of Experts) models.

This module provides fake quantization for MoE expert layers using decomposed
PyTorch ops with flashinfer's block-scale FP4 quantize-dequantize roundtrip
inserted before each GEMM. Gradients flow through the STE (Straight-Through
Estimator) automatically.

The per-expert computation matches the flashinfer ``trtllm_fp4_block_scale_moe``
kernel numerics: standard SwiGLU activation with contiguous gate/value halves
and kernel weight layout (``gemm1_weight [E, 2*I, H]``, ``gemm2_weight [E, H, I]``).

Usage::

    from torchao.prototype.qat.nvfp4_moe import apply_nvfp4_moe_qat
    model = apply_nvfp4_moe_qat(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MoE reference forward
# ---------------------------------------------------------------------------


def _calculate_fp4_global_scale_factor(tensor):
    """Compute FP4 global scale: ``(448 * 6) / amax``.

    448 is the max representable FP8-E4M3 value and 6 is the max FP4-E2M1
    value; their product bounds the NvFP4 dynamic range.
    """
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def _quant_dequant_fp4(a):
    """FP4 quantize-then-dequantize roundtrip (flashinfer, block-scale).

    Matches the intermediate-activation quantisation that the
    ``trtllm_fp4_block_scale_moe`` kernel performs between GEMM1 and GEMM2.
    Returns ``(dequantized, global_scale_factor)``.
    """
    from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize

    a_global_sf = _calculate_fp4_global_scale_factor(a)
    a_fp4, a_sf = fp4_quantize(a.cuda(), a_global_sf.cuda(), 16, False, True)
    a_pt = e2m1_and_ufp8sf_scale_to_float(
        a_fp4.cpu(),
        a_sf.cpu().reshape(-1),
        (1 / a_global_sf).cpu(),
        16,  # sf_vec_size
        1,   # ufp8_type (E4M3)
        True,  # is_sf_swizzled_layout
    )
    return a_pt.cuda(), a_global_sf


def _fp4_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """FP4 fake quantize with STE, using flashinfer's block-scale FP4.

    Matches the quantisation the ``trtllm_fp4_block_scale_moe`` kernel
    applies to activations and weights.  The forward pass uses the
    dequantized value; the backward treats quantisation as identity (STE).
    """
    dq_val, _ = _quant_dequant_fp4(x.to(torch.bfloat16).detach())
    return x + (dq_val.to(x.dtype) - x).detach()


def _build_permute_info(router_indices, routing_weights, num_experts, padding):
    """Build permutation tables from the module's routing tensors.

    Converts ``(router_indices, routing_weights)`` — the format used by
    :class:`NVFP4FakeQuantizedMoE` — into the padded-permutation tables
    that :func:`_run_moe_reference` expects.

    Args:
        router_indices: ``[T, top_k]`` pre-selected expert indices.
            Values in ``[0, num_experts]``; ``num_experts`` is a no-op sentinel.
        routing_weights: ``[T, E]`` full routing weights (softmax over experts).
        num_experts: number of experts.
        padding: alignment padding for expert token buffers.

    Returns:
        ``(permuted_buffer_size, expanded_token_idx_to_permuted_idx,
        num_tokens_per_expert, top_k_logits)``
    """
    device = router_indices.device
    num_tokens, top_k = router_indices.shape
    ri_cpu = router_indices.cpu()

    # Per-(token, k) weights — clamp sentinel indices and zero them out.
    top_k_logits = routing_weights.gather(
        1, router_indices.clamp(max=num_experts - 1)
    )
    top_k_logits = top_k_logits.masked_fill(router_indices >= num_experts, 0.0)

    # Count tokens per expert.
    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64)
    expert_token_count = torch.zeros(num_experts, dtype=torch.int64)
    for t in range(num_tokens):
        for k in range(top_k):
            eidx = ri_cpu[t, k].item()
            if eidx < num_experts:
                num_tokens_per_expert[eidx] += 1

    # Padded prefix sum.
    padded_prefix_sum = torch.zeros(num_experts + 1, dtype=torch.int64)
    for i in range(num_experts):
        padded_prefix_sum[i + 1] = padded_prefix_sum[i] + (
            (num_tokens_per_expert[i] + padding - 1) // padding * padding
        )
    permuted_buffer_size = padded_prefix_sum[num_experts].item()

    # Token-to-permuted-index mapping.
    expanded_token_idx_to_permuted_idx = -torch.ones(
        num_tokens * top_k, dtype=torch.int64
    )
    for t in range(num_tokens):
        for k in range(top_k):
            eidx = ri_cpu[t, k].item()
            if eidx < num_experts:
                expanded_idx = t * top_k + k
                permuted_idx = padded_prefix_sum[eidx] + expert_token_count[eidx]
                expanded_token_idx_to_permuted_idx[expanded_idx] = permuted_idx
                expert_token_count[eidx] += 1

    return (
        permuted_buffer_size,
        expanded_token_idx_to_permuted_idx.to(device),
        num_tokens_per_expert.to(device),
        top_k_logits,
    )


def _run_moe_reference(
    hidden_states_float,
    permute_info,
    gemm1_weights_float,
    gemm2_weights_float,
    num_experts,
    num_tokens,
    top_k,
    hidden_size,
    intermediate_size,
    padding,
    gemm1_bias=None,
    gemm2_bias=None,
):
    """MoE forward with NVFP4 fake quantization, in pure PyTorch.

    Matches the flashinfer ``trtllm_fp4_block_scale_moe`` kernel numerics:

    - Hidden states are fake-quantized once globally (before permutation),
      matching the kernel's single-amax input quantization.
    - Weights are fake-quantized per-expert before each GEMM.
    - The intermediate activation (GEMM1 → SwiGLU → GEMM2) goes through a
      single FP4 quant-dequant roundtrip with STE; the kernel takes this FP4
      data directly into GEMM2, so no re-quantization is needed.

    Returns ``(output, c_global_sf)`` where *c_global_sf* is the FP4 global
    scale of the intermediate activation (diagnostic — needed by the kernel
    comparison tests for output-scale computation).
    """
    total_padded, expanded_idx, num_tok_per_expert, expert_weight = permute_info
    expanded_idx = expanded_idx.cpu()
    num_tok_per_expert = num_tok_per_expert.cpu()
    expert_weight = expert_weight.float()

    # 0. Fake-quantize hidden states once globally (matches kernel input quantization).
    hidden_states_fq = _fp4_fake_quantize(hidden_states_float)

    # 1. Permute tokens into expert-sorted order.
    permute_out = torch.full(
        (total_padded, hidden_size), float("nan"), device="cuda"
    ).float()
    for i in range(num_tokens):
        for j in range(top_k):
            pid = expanded_idx[i * top_k + j]
            permute_out[pid] = hidden_states_fq[i]

    # 2. GEMM1 — per-expert matmul: [T_e, H] @ [2*I, H]^T → [T_e, 2*I]
    gemm1_out = torch.full(
        (total_padded, 2 * intermediate_size), float("nan"), device="cuda"
    ).float()
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert[eidx].item()
        if n == 0:
            continue
        act = permute_out[pos : pos + n]
        w1 = _fp4_fake_quantize(gemm1_weights_float[eidx]).float()
        gemm1_out[pos : pos + n] = act @ w1.t()
        if gemm1_bias is not None:
            gemm1_out[pos : pos + n] += gemm1_bias[eidx]
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 3. SwiGLU activation: silu(gate) * value
    #    Weight layout: first I cols = value ("up"), next I cols = gate.
    act_out = torch.full(
        (total_padded, intermediate_size), float("nan"), device="cuda"
    ).float()
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert[eidx].item()
        if n == 0:
            continue
        a = gemm1_out[pos : pos + n]
        value = a[:, :intermediate_size]
        gate = a[:, intermediate_size:]
        act_out[pos : pos + n] = F.silu(gate) * value
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 4. Intermediate FP4 quant-dequant with STE (matches kernel numerics).
    #    The kernel quantizes the SwiGLU output to FP4 and feeds it directly
    #    into GEMM2, so no additional activation quantization is needed in step 5.
    act_bf16 = act_out.to(torch.bfloat16)
    dq_val, c_global_sf = _quant_dequant_fp4(act_bf16.detach())
    act_out = (act_bf16 + (dq_val.to(act_bf16.dtype) - act_bf16).detach()).float()

    # 5. GEMM2 — per-expert matmul: [T_e, I] @ [H, I]^T → [T_e, H]
    #    Activation is already at FP4 precision from step 4; only weights
    #    need fake quantization.
    gemm2_out = torch.full(
        (total_padded, hidden_size), float("nan"), device="cuda"
    ).float()
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert[eidx].item()
        if n == 0:
            continue
        act = act_out[pos : pos + n]
        w2 = _fp4_fake_quantize(gemm2_weights_float[eidx]).float()
        gemm2_out[pos : pos + n] = act @ w2.t()
        if gemm2_bias is not None:
            gemm2_out[pos : pos + n] += gemm2_bias[eidx]
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 6. Finalise: weighted sum over each token's top-k experts.
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device="cuda")
    for i in range(num_tokens):
        for k in range(top_k):
            pid = expanded_idx[i * top_k + k]
            output[i] += gemm2_out[pid] * expert_weight[i, k]

    return output, c_global_sf


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class NVFP4FakeQuantizedMoE(nn.Module):
    """Drop-in replacement for ``GptOssExperts`` that inserts NVFP4 fake
    quantization before each GEMM in the expert forward pass.

    The per-expert computation matches the flashinfer
    ``trtllm_fp4_block_scale_moe`` kernel: standard SwiGLU with kernel weight
    layout (``gemm1_weight [E, 2*I, H]``, ``gemm2_weight [E, H, I]``).

    All operations are standard PyTorch ops, so autograd traces through
    them natively. The NVFP4 fake quantization (quantize->dequantize roundtrip)
    injects quantization noise in the forward pass, and the STE lets
    gradients flow through as if the quantization were identity.
    """

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Kernel weight layout
        self.gemm1_weight = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.gemm2_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        # Optional biases (None by default, populated by from_experts if source has them)
        self.gemm1_bias = None
        self.gemm2_bias = None

    @classmethod
    def from_experts(cls, experts: nn.Module) -> "NVFP4FakeQuantizedMoE":
        """Create from an existing ``GptOssExperts`` module, converting weight layout.

        Converts GptOss weight layout (interleaved columns, ``[E, H, 2*I]``) to
        kernel layout (contiguous halves, ``[E, 2*I, H]``).
        """
        new = cls.__new__(cls)
        nn.Module.__init__(new)

        new.num_experts = experts.num_experts
        new.hidden_size = experts.hidden_size
        new.intermediate_size = experts.intermediate_size

        # Convert GptOss layout to kernel layout:
        # GptOss gate_up_proj: [E, H, 2*I] with interleaved columns (even=gate, odd=value)
        # Kernel gemm1_weight: [E, 2*I, H] with contiguous halves [value; gate]
        gate_up = experts.gate_up_proj.data  # [E, H, 2*I]
        gate_cols = gate_up[:, :, ::2]       # [E, H, I] — gate (even cols)
        value_cols = gate_up[:, :, 1::2]     # [E, H, I] — value (odd cols)
        # Stack as [value; gate] then transpose to [E, 2*I, H]
        gemm1_kernel = torch.cat([value_cols, gate_cols], dim=2).transpose(1, 2).contiguous()
        new.gemm1_weight = nn.Parameter(gemm1_kernel)

        # GptOss down_proj: [E, I, H] → kernel gemm2_weight: [E, H, I]
        new.gemm2_weight = nn.Parameter(experts.down_proj.data.transpose(1, 2).contiguous())

        # Convert biases: de-interleave gate_up_proj_bias
        bias = experts.gate_up_proj_bias.data  # [E, 2*I]
        gate_bias = bias[:, ::2]    # [E, I]
        value_bias = bias[:, 1::2]  # [E, I]
        new.gemm1_bias = nn.Parameter(torch.cat([value_bias, gate_bias], dim=1))
        new.gemm2_bias = nn.Parameter(experts.down_proj_bias.data.clone())

        return new

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices=None,
        routing_weights=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states_2d = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states_2d.shape[0]
        top_k = router_indices.shape[1]

        with torch.no_grad():
            permute_info = _build_permute_info(
                router_indices, routing_weights, self.num_experts, padding=1,
            )

        output, _ = _run_moe_reference(
            hidden_states_2d,
            permute_info,
            self.gemm1_weight,
            self.gemm2_weight,
            self.num_experts,
            num_tokens,
            top_k,
            self.hidden_size,
            self.intermediate_size,
            padding=1,
            gemm1_bias=self.gemm1_bias,
            gemm2_bias=self.gemm2_bias,
        )

        return output.to(hidden_states.dtype).view(batch_size, -1, self.hidden_size)


# Backward compatibility alias
NVFP4FakeQuantizedGptOssExperts = NVFP4FakeQuantizedMoE


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def apply_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Replace all ``GptOssExperts`` modules with ``NVFP4FakeQuantizedMoE``.

    This applies NVFP4 QAT to the MoE expert layers so that the forward pass
    uses decomposed PyTorch ops with NVFP4 fake quantization while backward
    uses the STE for gradient computation.

    Args:
        model: A HuggingFace model containing ``GptOssExperts`` modules.

    Returns:
        The same model with expert modules replaced in-place.
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, GptOssExperts):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        new_module = NVFP4FakeQuantizedMoE.from_experts(module)
        setattr(parent, attr_name, new_module)

    return model
