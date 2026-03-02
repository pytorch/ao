"""
NVFP4 QAT for MoE (Mixture of Experts) models.

This module provides fake quantization for MoE expert layers using the
flashinfer ``trtllm_fp4_block_scale_routed_moe`` NVFP4 kernel. The forward
pass uses the actual NVFP4 kernel (simulating inference-time quantization
noise), while the backward pass replays the forward in high-precision bf16
and lets autograd compute gradients (Straight-Through Estimator).

Usage::

    from torchao.prototype.qat.nvfp4_moe import apply_nvfp4_moe_qat
    model = apply_nvfp4_moe_qat(model)
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from flashinfer.fp4_quantization import block_scale_interleave, fp4_quantize
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)

# Max representable values: FP8 E4M3 = 448, FP4 E2M1 = 6
_FP8_MAX = 448.0
_FP4_MAX = 6.0

# Kernel internals constant for MMA epilogue tile
_EPILOGUE_TILE_M = 128

# Minimum alignment for the shuffle functions (M % 128 == 0).
_SHUFFLE_ALIGNMENT = 128

# The TRTLLM batched-GEMM runner needs the hidden dimension to be a
# multiple of 512 for valid kernel configs (matching vLLM's check).
_KERNEL_ALIGNMENT = 512

# ActivationType.Swiglu from flashinfer
_ACTIVATION_TYPE_SWIGLU = 3


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _round_up(x: int, alignment: int) -> int:
    """Round *x* up to the next multiple of *alignment*."""
    return (x + alignment - 1) // alignment * alignment


def _compute_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the global scale factor for FP4 quantization.

    global_scale = (FP8_MAX * FP4_MAX) / amax(tensor)
    """
    amax = tensor.abs().max()
    amax = torch.clamp(amax, min=1e-12)
    return torch.tensor(
        (_FP8_MAX * _FP4_MAX) / amax.item(),
        dtype=torch.float32,
        device=tensor.device,
    )


def _deinterleave_gate_up(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """De-interleave gate_up_proj and reorder to [w3(up), w1(gate)] layout.

    The original GptOssExperts stores gate/up weights interleaved:
        even columns = gate (w1), odd columns = up (w3)

    The flashinfer kernel expects [w3, w1] ordering with shape [E, 2*I, H].

    Args:
        gate_up_proj: [E, H, 2*I] with even=gate, odd=up interleaving
        gate_up_proj_bias: [E, 2*I] with same interleaving

    Returns:
        w13: [E, 2*I, H] contiguous (w3 first half, w1 second half)
        w13_bias: [E, 2*I] float32 (up_bias first, gate_bias second)
    """
    # De-interleave: even=gate(w1), odd=up(w3)
    gate = gate_up_proj[:, :, 0::2]  # [E, H, I]
    up = gate_up_proj[:, :, 1::2]  # [E, H, I]

    # Reorder to [w3(up), w1(gate)] and transpose for kernel layout
    w13 = torch.cat([up, gate], dim=-1)  # [E, H, 2*I]
    w13 = w13.transpose(1, 2).contiguous()  # [E, 2*I, H]

    # De-interleave bias similarly
    gate_bias = gate_up_proj_bias[:, 0::2]  # [E, I]
    up_bias = gate_up_proj_bias[:, 1::2]  # [E, I]
    w13_bias = torch.cat([up_bias, gate_bias], dim=-1).to(torch.float32).contiguous()

    return w13, w13_bias


def _quantize_and_shuffle_weights(
    w13: torch.Tensor,  # [E, 2*I, H]
    w2: torch.Tensor,  # [E, H, I]
    cache: Dict,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize weights to FP4 and apply TRT-LLM row shuffling per expert.

    Returns:
        w13_fp4: shuffled packed FP4 weights for GEMM1
        w13_scales: shuffled block scales for GEMM1 (float8)
        w2_fp4: shuffled packed FP4 weights for GEMM2
        w2_scales: shuffled block scales for GEMM2 (float8)
        w13_global_sf: [E] per-expert global scale factors for GEMM1 weights
        w2_global_sf: [E] per-expert global scale factors for GEMM2 weights
    """
    E = w13.shape[0]
    two_I = w13.shape[1]
    H = w13.shape[2]
    I = w2.shape[2]

    w13_fp4_list = []
    w13_scales_list = []
    w2_fp4_list = []
    w2_scales_list = []
    w13_global_sf_list = []
    w2_global_sf_list = []

    for e in range(E):
        # --- GEMM1 (w13) weights ---
        w13_e = w13[e]  # [2*I, H]
        w13_gsf = _compute_global_scale(w13_e)
        w13_global_sf_list.append(w13_gsf)

        w13_fp4_e, w13_sc_e = fp4_quantize(
            w13_e, w13_gsf, sf_vec_size=16, is_sf_swizzled_layout=False
        )

        # Shuffle weights for MMA tile layout
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_fp4_e.view(torch.uint8), _EPILOGUE_TILE_M
        )
        w13_fp4_list.append(
            w13_fp4_e.view(torch.uint8)[perm.to(w13_fp4_e.device)].contiguous()
        )

        # Shuffle scales
        perm_sf = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w13_sc_e.view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=16,
        )
        w13_scales_list.append(
            block_scale_interleave(
                w13_sc_e.view(torch.uint8)[perm_sf.to(w13_sc_e.device)].contiguous()
            )
        )

        # --- GEMM2 (w2) weights ---
        w2_e = w2[e]  # [H, I]
        w2_gsf = _compute_global_scale(w2_e)
        w2_global_sf_list.append(w2_gsf)

        w2_fp4_e, w2_sc_e = fp4_quantize(
            w2_e, w2_gsf, sf_vec_size=16, is_sf_swizzled_layout=False
        )

        # Shuffle weights
        perm = get_w2_permute_indices_with_cache(
            cache, w2_fp4_e.view(torch.uint8), _EPILOGUE_TILE_M
        )
        w2_fp4_list.append(
            w2_fp4_e.view(torch.uint8)[perm.to(w2_fp4_e.device)].contiguous()
        )

        # Shuffle scales
        perm_sf = get_w2_permute_indices_with_cache(
            cache,
            w2_sc_e.view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=16,
        )
        w2_scales_list.append(
            block_scale_interleave(
                w2_sc_e.view(torch.uint8)[perm_sf.to(w2_sc_e.device)].contiguous()
            )
        )

    # Stack per-expert results
    w13_fp4 = torch.stack(w13_fp4_list)
    w13_scales = (
        torch.stack(w13_scales_list)
        .view(torch.float8_e4m3fn)
        .reshape(E, two_I, H // 16)
    )

    w2_fp4 = torch.stack(w2_fp4_list)
    w2_scales = (
        torch.stack(w2_scales_list)
        .view(torch.float8_e4m3fn)
        .reshape(E, H, I // 16)
    )

    w13_global_sf = torch.stack(w13_global_sf_list)
    w2_global_sf = torch.stack(w2_global_sf_list)

    return w13_fp4, w13_scales, w2_fp4, w2_scales, w13_global_sf, w2_global_sf


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class _NVFP4MoEForwardBF16Backward(torch.autograd.Function):
    """Autograd function: NVFP4 flashinfer kernel forward, bf16 replay backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        hidden_states: torch.Tensor,  # [num_tokens, H]
        gate_up_proj: torch.Tensor,  # [E, H, 2*I]
        gate_up_proj_bias: torch.Tensor,  # [E, 2*I]
        down_proj: torch.Tensor,  # [E, I, H]
        down_proj_bias: torch.Tensor,  # [E, H]
        router_indices: torch.Tensor,  # [num_tokens, top_k]
        routing_weights: torch.Tensor,  # [num_tokens, num_experts]
        num_experts: int,
        intermediate_size: int,
        alpha: float,
        limit: float,
        permute_cache: Dict,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        top_k = router_indices.shape[1]

        # ----- Weight preparation -----
        w13, w13_bias = _deinterleave_gate_up(gate_up_proj, gate_up_proj_bias)
        w2 = down_proj.transpose(1, 2).contiguous()  # [E, H, I]
        w2_bias = down_proj_bias.to(torch.float32)  # [E, H]

        # ----- Dimension alignment -----
        # The flashinfer shuffle functions require M % 128 == 0, and the
        # TRTLLM batched-GEMM runner needs the hidden dim to be a multiple
        # of 512 for valid kernel configs.  Pad only the dimensions that
        # actually need it.
        two_I = w13.shape[1]
        I = w2.shape[2]

        # Hidden dim: used as M for w2 shuffle AND K for GEMM1.
        # Must satisfy both shuffle (128) and kernel (512) alignment.
        padded_H = _round_up(hidden_size, _KERNEL_ALIGNMENT)
        # Intermediate dim: used as M for w13 shuffle. Only needs 128.
        padded_2I = _round_up(two_I, _SHUFFLE_ALIGNMENT)
        need_pad_h = padded_H != hidden_size
        need_pad_i = padded_2I != two_I

        # For w2, the K dimension is I.  The kernel's GEMM2 uses the
        # intermediate_size parameter to know the K dimension.  If we
        # padded 2*I we must also pad I for w2's K to stay consistent.
        # If 2*I didn't need padding, keep I as-is.
        padded_I = padded_2I // 2 if need_pad_i else I

        if need_pad_h or need_pad_i:
            pad_h = padded_H - hidden_size
            pad_2i = padded_2I - two_I
            pad_i = padded_I - I

            # Pad w13: [E, 2*I, H] → [E, padded_2I, padded_H]
            w13 = torch.nn.functional.pad(w13, (0, pad_h, 0, pad_2i))
            if need_pad_i:
                w13_bias = torch.nn.functional.pad(w13_bias, (0, pad_2i))

            # Pad w2: [E, H, I] → [E, padded_H, padded_I]
            w2 = torch.nn.functional.pad(w2, (0, pad_i, 0, pad_h))
            if need_pad_h:
                w2_bias = torch.nn.functional.pad(w2_bias, (0, pad_h))

            # Pad hidden_states: [T, H] → [T, padded_H]
            if need_pad_h:
                hidden_states_for_kernel = torch.nn.functional.pad(
                    hidden_states, (0, pad_h)
                )
            else:
                hidden_states_for_kernel = hidden_states
        else:
            hidden_states_for_kernel = hidden_states

        w13_fp4, w13_scales, w2_fp4, w2_scales, w13_gsf, w2_gsf = (
            _quantize_and_shuffle_weights(w13, w2, permute_cache)
        )

        # ----- Activation quantization -----
        a_global_sf = _compute_global_scale(hidden_states_for_kernel)
        hidden_fp4, hidden_scales = fp4_quantize(
            hidden_states_for_kernel, a_global_sf, sf_vec_size=16,
            is_sf_swizzled_layout=False,
        )

        # ----- Pack topk_ids (expert index << 16 | bf16 weight as int16) -----
        valid_mask = router_indices < num_experts
        safe_indices = router_indices.clamp(max=num_experts - 1)
        topk_weights = routing_weights.gather(1, safe_indices)
        topk_weights = topk_weights * valid_mask.to(topk_weights.dtype)

        packed_topk = (safe_indices.to(torch.int32) << 16) | (
            topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        )

        # ----- Compute output scales -----
        # a_scale = 1/a_global_sf (per-tensor activation dequant scale)
        # w13_scale = 1/w13_gsf  (per-expert weight dequant scale)
        # w2_scale = 1/w2_gsf    (per-expert weight dequant scale)
        a_scale = 1.0 / a_global_sf  # scalar
        w13_scale = 1.0 / w13_gsf  # [E]
        w2_scale = 1.0 / w2_gsf  # [E]

        g1_scale_c = (a_scale * w13_scale).to(torch.float32)  # [E]
        g1_alphas = (a_scale * w13_scale).to(torch.float32)  # [E]
        g2_alphas = w2_scale.to(torch.float32)  # [E]

        # ----- Activation parameters -----
        dev = hidden_states.device
        alpha_t = torch.full((num_experts,), alpha, dtype=torch.float32, device=dev)
        beta_t = torch.full((num_experts,), 1.0, dtype=torch.float32, device=dev)
        clamp_t = torch.full((num_experts,), limit, dtype=torch.float32, device=dev)

        # ----- Call flashinfer kernel -----
        output = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=hidden_fp4,
            hidden_states_scale=hidden_scales.view(torch.float8_e4m3fn).flatten(),
            gemm1_weights=w13_fp4,
            gemm1_weights_scale=w13_scales,
            gemm1_bias=w13_bias,
            gemm1_alpha=alpha_t,
            gemm1_beta=beta_t,
            gemm1_clamp_limit=clamp_t,
            gemm2_weights=w2_fp4,
            gemm2_weights_scale=w2_scales,
            gemm2_bias=w2_bias,
            output1_scale_scalar=g1_scale_c,
            output1_scale_gate_scalar=g1_alphas,
            output2_scale_scalar=g2_alphas,
            num_experts=num_experts,
            top_k=top_k,
            n_group=0,
            topk_group=0,
            intermediate_size=padded_I,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            activation_type=_ACTIVATION_TYPE_SWIGLU,
        )[0]

        # Slice back to original hidden size if we padded
        if need_pad_h:
            output = output[:, :hidden_size]

        # ----- Save for backward -----
        ctx.save_for_backward(
            hidden_states,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            router_indices,
            routing_weights,
        )
        ctx.num_experts = num_experts
        ctx.alpha = alpha
        ctx.limit = limit

        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        (
            hidden_states,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            router_indices,
            routing_weights,
        ) = ctx.saved_tensors

        num_experts = ctx.num_experts
        alpha = ctx.alpha
        limit = ctx.limit
        hidden_size = hidden_states.shape[-1]

        # Replay the forward in bf16 with autograd tracking
        with torch.enable_grad():
            hs = hidden_states.detach().requires_grad_(True)
            gup = gate_up_proj.detach().requires_grad_(True)
            gup_bias = gate_up_proj_bias.detach().requires_grad_(True)
            dp = down_proj.detach().requires_grad_(True)
            dp_bias = down_proj_bias.detach().requires_grad_(True)
            rw = routing_weights.detach().requires_grad_(True)

            # Compute expert routing mask (no grad needed)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    router_indices, num_classes=num_experts + 1
                )
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0
                ).nonzero()

            # Replay the original GptOssExperts training-path forward
            num_tokens = hs.shape[0]
            next_states = torch.zeros(
                num_tokens, hidden_size, dtype=hs.dtype, device=hs.device
            )
            for expert_idx_item in expert_hit:
                expert_idx = expert_idx_item[0]
                if expert_idx == num_experts:
                    continue
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])

                current_state = hs[token_idx]
                gate_up = (
                    current_state @ gup[expert_idx] + gup_bias[expert_idx]
                )
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=limit)
                up = up.clamp(min=-limit, max=limit)
                glu = gate * torch.sigmoid(gate * alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ dp[expert_idx] + dp_bias[expert_idx]
                weighted_output = out * rw[token_idx, expert_idx, None]

                # Accumulate using non-in-place advanced indexing
                padded = torch.zeros_like(next_states)
                padded[token_idx] = weighted_output.to(hs.dtype)
                next_states = next_states + padded

            next_states.backward(grad_output)

        # Return grads for each forward input (None for non-tensor args)
        return (
            hs.grad,
            gup.grad,
            gup_bias.grad,
            dp.grad,
            dp_bias.grad,
            None,  # router_indices
            rw.grad,
            None,  # num_experts
            None,  # intermediate_size
            None,  # alpha
            None,  # limit
            None,  # permute_cache
        )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class NVFP4FakeQuantizedGptOssExperts(nn.Module):
    """Drop-in replacement for ``GptOssExperts`` that uses the flashinfer
    NVFP4 fused-MoE kernel in the forward pass and bf16 replay in backward.

    The module keeps the same ``nn.Parameter`` attributes as the original so
    that optimizer state and checkpointing work unchanged.
    """

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )

        # Permute index cache (computed once, reused every forward)
        self._permute_cache: Dict = {}

    @classmethod
    def from_experts(cls, experts: nn.Module, config) -> "NVFP4FakeQuantizedGptOssExperts":
        """Create from an existing ``GptOssExperts`` module, sharing parameter storage."""
        new = cls.__new__(cls)
        nn.Module.__init__(new)

        new.intermediate_size = experts.intermediate_size
        new.num_experts = experts.num_experts
        new.hidden_size = experts.hidden_size
        new.expert_dim = experts.expert_dim
        new.alpha = experts.alpha
        new.limit = experts.limit

        # Share parameters (no copy)
        new.gate_up_proj = experts.gate_up_proj
        new.gate_up_proj_bias = experts.gate_up_proj_bias
        new.down_proj = experts.down_proj
        new.down_proj_bias = experts.down_proj_bias

        new._permute_cache = {}
        return new

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices=None,
        routing_weights=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states_2d = hidden_states.reshape(-1, self.hidden_size)

        output = _NVFP4MoEForwardBF16Backward.apply(
            hidden_states_2d,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            router_indices,
            routing_weights,
            self.num_experts,
            self.intermediate_size,
            self.alpha,
            self.limit,
            self._permute_cache,
        )

        return output.view(batch_size, -1, self.hidden_size)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def apply_nvfp4_moe_qat(model: nn.Module) -> nn.Module:
    """Replace all ``GptOssExperts`` modules with ``NVFP4FakeQuantizedGptOssExperts``.

    This applies NVFP4 QAT to the MoE expert layers so that the forward pass
    uses the flashinfer NVFP4 kernel while backward uses bf16 GEMMs.

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

        new_module = NVFP4FakeQuantizedGptOssExperts.from_experts(
            module, model.config
        )
        setattr(parent, attr_name, new_module)

    return model
