"""
FP4 QAT for MoE (Mixture of Experts) models via module swap.

This module provides fake quantization for MoE expert layers using block-scale
FP4 (E2M1) quantize-dequantize roundtrips inserted before each GEMM, implemented
entirely in PyTorch (no CPU↔GPU transfers).  Gradients flow through the STE
(Straight-Through Estimator) automatically.

The per-expert computation matches the flashinfer ``trtllm_fp4_block_scale_moe``
kernel numerics: standard SwiGLU activation with contiguous gate/value halves
and kernel weight layout (``gemm1_weight [E, 2*I, H]``, ``gemm2_weight [E, H, I]``).

**Currently only supports Qwen3 MoE** (``Qwen3MoeSparseMoeBlock``).  Adding
support for other architectures requires implementing model-specific weight
packing in ``from_*_experts`` and ``remove_nvfp4_moe_qat``.  For a
model-agnostic approach, use ``torchao.prototype.qat.nvfp4_moe`` (tensor
subclass) instead.

Requires transformers >= 5.0 for the ``Qwen3MoeExperts`` fused expert layout
and ``Qwen3MoeTopKRouter``.

Usage::

    from torchao.prototype.qat.nvfp4_moe_module_swap import apply_nvfp4_moe_qat
    model = apply_nvfp4_moe_qat(model)
    # ... train ...
    from torchao.prototype.qat.nvfp4_moe_module_swap import remove_nvfp4_moe_qat
    model = remove_nvfp4_moe_qat(model)
"""

from importlib.metadata import version

import torch

_transformers_version = tuple(int(x) for x in version("transformers").split(".")[:2])
if _transformers_version < (5, 0):
    raise ImportError(
        f"torchao.prototype.qat.nvfp4_moe_module_swap requires transformers >= 5.0 "
        f"(found {version('transformers')}). The Qwen3MoeExperts fused expert "
        f"layout is not available in older versions."
    )
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FP4 E2M1 quantization (pure PyTorch, all on GPU)
# ---------------------------------------------------------------------------

# Representable positive E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoint boundaries between consecutive values (for round-to-nearest)
_E2M1_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_SF_BLOCK_SIZE = 16


def _calculate_fp4_global_scale_factor(tensor: torch.Tensor) -> torch.Tensor:
    """Compute FP4 global scale: ``(448 * 6) / amax``.

    448 is the max representable FP8-E4M3 value and 6 is the max FP4-E2M1
    value; their product bounds the NvFP4 dynamic range.
    """
    return (448 * 6) / tensor.float().abs().nan_to_num().max()


def _quant_dequant_fp4(
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP4 E2M1 block-scale quantize-dequantize roundtrip, all on GPU.

    Implements the same numerics as flashinfer's ``fp4_quantize`` followed by
    ``e2m1_and_ufp8sf_scale_to_float``, but without any CPU↔GPU transfers.

    Returns ``(dequantized, global_scale_factor)``.
    """
    device = a.device
    gsf = _calculate_fp4_global_scale_factor(a)

    # Scale by global scale factor.
    x = a.float() * gsf
    orig_shape = x.shape
    x_flat = x.reshape(-1, _SF_BLOCK_SIZE)

    # Per-block FP8-E4M3 scale factor: block_amax / fp4_max, quantized to FP8.
    block_amax = x_flat.abs().amax(dim=-1, keepdim=True)
    block_sf = (block_amax / 6.0).to(torch.float8_e4m3fn).float()

    # Normalize by block scale and round to nearest E2M1 value.
    x_norm = x_flat / block_sf.clamp(min=torch.finfo(torch.float8_e4m3fn).tiny)
    sign = x_norm.sign()
    x_abs = x_norm.abs()
    boundaries = torch.tensor(_E2M1_BOUNDARIES, device=device)
    values = torch.tensor(_E2M1_VALUES, device=device)
    indices = torch.bucketize(x_abs, boundaries)
    x_quant = sign * values[indices]

    # Dequantize: reverse the block scale and global scale.
    x_dq = (x_quant * block_sf).reshape(orig_shape) / gsf

    return x_dq, gsf


def _fp4_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """FP4 fake quantize with STE, using block-scale FP4 (E2M1).

    Matches the quantisation the ``trtllm_fp4_block_scale_moe`` kernel
    applies to activations and weights.  The forward pass uses the
    dequantized value; the backward treats quantisation as identity (STE).
    """
    dq_val, _ = _quant_dequant_fp4(x.to(torch.bfloat16).detach())
    return x + (dq_val.to(x.dtype) - x).detach()


# ---------------------------------------------------------------------------
# MoE reference forward
# ---------------------------------------------------------------------------


def _build_permute_info(
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    padding: int,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Per-(token, k) weights — clamp sentinel indices and zero them out.
    top_k_logits = routing_weights.gather(1, router_indices.clamp(max=num_experts - 1))
    top_k_logits = top_k_logits.masked_fill(router_indices >= num_experts, 0.0)

    # Flat view of expert assignments: [num_tokens * top_k]
    flat_indices = router_indices.reshape(-1)  # [T * top_k]
    valid = flat_indices < num_experts

    # Count tokens per expert (on GPU).
    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64, device=device)
    num_tokens_per_expert.scatter_add_(
        0, flat_indices[valid], torch.ones_like(flat_indices[valid])
    )

    # Padded counts and prefix sum (on GPU).
    padded_counts = (num_tokens_per_expert + padding - 1) // padding * padding
    padded_prefix_sum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    padded_prefix_sum[1:] = padded_counts.cumsum(0)
    permuted_buffer_size = padded_prefix_sum[num_experts].item()

    # Token-to-permuted-index mapping (on GPU).
    # For each valid (token, k) pair, compute its position within its expert's
    # block by finding the rank among all valid entries for the same expert.
    expanded_token_idx_to_permuted_idx = -torch.ones(
        num_tokens * top_k, dtype=torch.int64, device=device
    )

    # Compute within-expert rank for each valid entry using argsort.
    # Sort valid entries by expert index; ties broken by flat position (stable).
    valid_positions = valid.nonzero(as_tuple=True)[0]  # indices into flat_indices
    valid_experts = flat_indices[valid_positions]

    # Stable sort by expert gives us contiguous expert groups in order.
    sort_order = valid_experts.argsort(stable=True)
    sorted_positions = valid_positions[sort_order]
    sorted_experts = valid_experts[sort_order]

    # Within-expert rank: position minus the start of that expert's group.
    expert_start_in_sorted = torch.zeros(num_experts, dtype=torch.int64, device=device)
    expert_start_in_sorted[1:] = num_tokens_per_expert[:-1].cumsum(0)
    # Each sorted entry's rank = its index in sorted array - expert_start_in_sorted[expert]
    sorted_idx = torch.arange(sorted_positions.shape[0], device=device)
    within_expert_rank = sorted_idx - expert_start_in_sorted[sorted_experts]

    permuted_idx = padded_prefix_sum[sorted_experts] + within_expert_rank
    expanded_token_idx_to_permuted_idx[sorted_positions] = permuted_idx

    return (
        permuted_buffer_size,
        expanded_token_idx_to_permuted_idx,
        num_tokens_per_expert,
        top_k_logits,
    )


def _run_moe_reference(
    hidden_states_float: torch.Tensor,
    permute_info: tuple[int, torch.Tensor, torch.Tensor, torch.Tensor],
    gemm1_weights_float: torch.Tensor,
    gemm2_weights_float: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    padding: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    # Moved to CPU because the per-expert loops below need Python ints for
    # tensor slicing.  This is a single small transfer (num_experts int64s)
    # and avoids a GPU sync on every .item() call inside the loop.
    num_tok_per_expert_cpu = num_tok_per_expert.cpu()
    expert_weight = expert_weight.float()
    device = hidden_states_float.device

    # 0. Fake-quantize hidden states once globally (matches kernel input quantization).
    hidden_states_fq = _fp4_fake_quantize(hidden_states_float)

    # 1. Permute tokens into expert-sorted order (vectorized scatter).
    permute_out = torch.zeros(
        total_padded, hidden_size, device=device, dtype=torch.float32
    )
    token_ids = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    valid = expanded_idx >= 0
    permute_out[expanded_idx[valid]] = hidden_states_fq[token_ids[valid]].float()

    # 2. GEMM1 — per-expert matmul: [T_e, H] @ [2*I, H]^T → [T_e, 2*I]
    gemm1_out = torch.zeros(
        total_padded, 2 * intermediate_size, device=device, dtype=torch.float32
    )
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert_cpu[eidx].item()
        if n == 0:
            continue
        act = permute_out[pos : pos + n]
        w1 = _fp4_fake_quantize(gemm1_weights_float[eidx]).float()
        gemm1_out[pos : pos + n] = act @ w1.t()
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 3. SwiGLU activation: silu(gate) * value
    #    Weight layout: first I cols = value ("up"), next I cols = gate.
    act_out = torch.zeros(
        total_padded, intermediate_size, device=device, dtype=torch.float32
    )
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert_cpu[eidx].item()
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
    gemm2_out = torch.zeros(
        total_padded, hidden_size, device=device, dtype=torch.float32
    )
    pos = 0
    for eidx in range(num_experts):
        n = num_tok_per_expert_cpu[eidx].item()
        if n == 0:
            continue
        act = act_out[pos : pos + n]
        w2 = _fp4_fake_quantize(gemm2_weights_float[eidx]).float()
        gemm2_out[pos : pos + n] = act @ w2.t()
        pos += n
        pos = (pos + padding - 1) // padding * padding

    # 6. Finalise: weighted sum over each token's top-k experts (vectorized gather).
    k_ids = (
        torch.arange(top_k, device=device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .reshape(-1)
    )
    weights = expert_weight[token_ids[valid], k_ids[valid]].unsqueeze(1)
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)
    output.index_add_(0, token_ids[valid], gemm2_out[expanded_idx[valid]] * weights)

    return output, c_global_sf


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class NVFP4FakeQuantizedMoE(nn.Module):
    """Batched MoE experts with NVFP4 fake quantization before each GEMM.

    The per-expert computation matches the flashinfer
    ``trtllm_fp4_block_scale_moe`` kernel: standard SwiGLU with kernel weight
    layout (``gemm1_weight [E, 2*I, H]``, ``gemm2_weight [E, H, I]``).

    All operations are standard PyTorch ops, so autograd traces through
    them natively. The NVFP4 fake quantization (quantize->dequantize roundtrip)
    injects quantization noise in the forward pass, and the STE lets
    gradients flow through as if the quantization were identity.
    """

    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int
    ) -> None:
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

    @classmethod
    def from_qwen3_experts(cls, experts: nn.Module) -> "NVFP4FakeQuantizedMoE":
        """Create from a ``Qwen3MoeExperts`` module (transformers 5.x).

        ``Qwen3MoeExperts`` stores fused 3D parameters ``gate_up_proj [E, 2*I, H]``
        and ``down_proj [E, H, I]``.

        Weights are stacked into kernel layout:
        ``gemm1_weight [E, 2*I, H]`` (up then gate), ``gemm2_weight [E, H, I]``.
        """
        num_experts = experts.num_experts
        hidden_size = experts.hidden_dim
        intermediate_size = experts.intermediate_dim

        new = cls(num_experts, hidden_size, intermediate_size)

        # gate_up_proj is [E, 2*I, H] but in order [gate; up].
        # Kernel layout expects [up; gate], so swap the halves.
        I = intermediate_size
        gate_up = experts.gate_up_proj.data  # [E, 2*I, H]
        gate = gate_up[:, :I, :]  # [E, I, H]
        up = gate_up[:, I:, :]  # [E, I, H]
        new.gemm1_weight = nn.Parameter(
            torch.cat([up, gate], dim=1)  # [E, 2*I, H] in [up; gate] order
        )
        new.gemm2_weight = nn.Parameter(experts.down_proj.data.clone())

        return new

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states_2d = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states_2d.shape[0]
        top_k = router_indices.shape[1]

        with torch.no_grad():
            permute_info = _build_permute_info(
                router_indices,
                routing_weights,
                self.num_experts,
                padding=1,
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
        )

        return output.to(hidden_states.dtype).view(batch_size, -1, self.hidden_size)


class NVFP4FakeQuantizedQwen3MoeBlock(nn.Module):
    """Drop-in replacement for ``Qwen3MoeSparseMoeBlock`` that inserts NVFP4
    fake quantization in the expert forward pass.

    Keeps the original gate (router) and routing logic; replaces the
    ``Qwen3MoeExperts`` with a single batched :class:`NVFP4FakeQuantizedMoE`.
    """

    def __init__(
        self,
        gate: nn.Module,
        qat_experts: NVFP4FakeQuantizedMoE,
        top_k: int,
        norm_topk_prob: bool,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = qat_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    @classmethod
    def from_qwen3_moe_block(
        cls, block: nn.Module
    ) -> "NVFP4FakeQuantizedQwen3MoeBlock":
        """Create from an existing ``Qwen3MoeSparseMoeBlock``."""
        qat_experts = NVFP4FakeQuantizedMoE.from_qwen3_experts(block.experts)
        top_k = block.gate.top_k
        norm_topk_prob = block.gate.norm_topk_prob
        return cls(
            gate=block.gate,
            qat_experts=qat_experts,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_2d = hidden_states.view(-1, hidden_dim)

        # Qwen3MoeTopKRouter returns (logits, weights, indices)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_2d)

        # Build full routing_weights tensor [T, E] for _build_permute_info
        num_tokens = hidden_states_2d.shape[0]
        num_experts = self.experts.num_experts
        full_routing_weights = torch.zeros(
            num_tokens,
            num_experts,
            dtype=routing_weights.dtype,
            device=routing_weights.device,
        )
        full_routing_weights.scatter_(1, selected_experts, routing_weights)

        # QAT expert forward
        output = self.experts(
            hidden_states,
            router_indices=selected_experts,
            routing_weights=full_routing_weights,
        )

        return output


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def remove_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Convert QAT MoE blocks back to standard ``Qwen3MoeSparseMoeBlock``.

    Unbatches ``gemm1_weight [E, 2*I, H]`` and ``gemm2_weight [E, H, I]``
    back into fused ``Qwen3MoeExperts`` with 3D parameters
    ``gate_up_proj [E, 2*I, H]`` and ``down_proj [E, H, I]``.

    Args:
        model: A model with :class:`NVFP4FakeQuantizedQwen3MoeBlock` modules.

    Returns:
        The same model with QAT blocks replaced by standard blocks in-place.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts,
        Qwen3MoeSparseMoeBlock,
    )

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, NVFP4FakeQuantizedQwen3MoeBlock):
            replacements.append((name, module))

    for name, qat_block in replacements:
        qat_experts = qat_block.experts
        E = qat_experts.num_experts
        H = qat_experts.hidden_size
        I = qat_experts.intermediate_size

        # Restore Qwen3MoeExperts with fused 3D params.
        experts_mod = Qwen3MoeExperts.__new__(Qwen3MoeExperts)
        nn.Module.__init__(experts_mod)
        experts_mod.num_experts = E
        experts_mod.hidden_dim = H
        experts_mod.intermediate_dim = I
        experts_mod.act_fn = F.silu

        # gemm1_weight is [E, 2*I, H] in [up; gate] order.
        # Qwen3MoeExperts stores gate_up_proj as [E, 2*I, H] in [gate; up] order.
        up = qat_experts.gemm1_weight.data[:, :I, :]  # [E, I, H]
        gate = qat_experts.gemm1_weight.data[:, I:, :]  # [E, I, H]
        experts_mod.gate_up_proj = nn.Parameter(
            torch.cat([gate, up], dim=1)  # [E, 2*I, H] in [gate; up] order
        )
        experts_mod.down_proj = nn.Parameter(qat_experts.gemm2_weight.data.clone())

        # Create a standard Qwen3MoeSparseMoeBlock shell.
        # We skip __init__ (requires config) and set attributes directly.
        new_block = Qwen3MoeSparseMoeBlock.__new__(Qwen3MoeSparseMoeBlock)
        nn.Module.__init__(new_block)
        new_block.gate = qat_block.gate
        new_block.experts = experts_mod

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name
        setattr(parent, attr_name, new_block)

    return model


def apply_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Replace MoE expert modules with NVFP4 fake-quantized versions.

    Supports ``Qwen3MoeSparseMoeBlock`` (from HuggingFace transformers).

    This applies NVFP4 QAT to the MoE expert layers so that the forward pass
    uses decomposed PyTorch ops with NVFP4 fake quantization while backward
    uses the STE for gradient computation.

    Args:
        model: A HuggingFace MoE model.

    Returns:
        The same model with expert modules replaced in-place.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        new_module = NVFP4FakeQuantizedQwen3MoeBlock.from_qwen3_moe_block(module)
        setattr(parent, attr_name, new_module)

    return model
