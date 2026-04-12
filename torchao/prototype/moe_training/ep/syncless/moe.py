# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
MXFP8 GroupedExperts module for use with the syncless expert-parallel
dispatch/combine pipeline.

Accepts the 4-arg dispatch output format from
``SynclessExpertParallel._token_dispatch``:
  (output_e4m3, output_scales_e8m0, num_tokens_per_expert, expert_padded_offsets)

Fuses w1 and w3 into a single w13 parameter of shape
(num_experts, 2 * hidden_dim, dim) so the two x @ w1 and x @ w3
projections are computed with a single grouped GEMM.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor


class MXFP8GroupedExpertsFunc(torch.autograd.Function):
    """MXFP8 grouped experts autograd function that saves activations to a unified GPU+CPU buffer.

    Forward:
      h13 = x @ w13.T
      h = SwiGLU(h13)
      out = h @ w2.T
    Saves x (FP8 data + scales) and h13 (BF16) to the
    ``SavedActivationsBuffer``.

    Backward: reads saved activations, computes dgrad + wgrad
    manually, and frees the buffer allocation.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        output_e4m3,
        output_scales_e8m0,
        num_tokens_per_expert,
        w13,
        w2,
        saved_activations_buffer,
    ):
        from torchao.prototype.moe_training.ep.syncless.copy_kernel import (
            copy_into_buffer_2d,
        )
        from torchao.prototype.moe_training.ep.syncless.cutedsl_mxfp8_gmm import (
            grouped_gemm as cutedsl_grouped_gemm,
        )
        from torchao.prototype.moe_training.kernels.mxfp8 import (
            mx_block_rearrange_2d_M_groups_cuda,
            triton_mx_block_rearrange_per_group_3d,
        )
        from torchao.prototype.moe_training.mxfp8_grouped_mm import _compute_fwd
        from torchao.prototype.mx_formats.config import ScaleCalculationMode
        from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
        from torchao.quantization.quantize_.common import KernelPreference

        block_size = 32
        hidden_dim = w13.shape[1] // 2

        group_end_offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        # Get actual token count as a GPU tensor (no D2H sync).
        num_tokens = group_end_offs[-1].to(torch.int64)

        # Allocate exact rows via CUDAAllocator (GPU-side, no sync).
        buf = saved_activations_buffer
        offset = buf.alloc(num_tokens)

        # Save dispatch output (FP8 data + scales) via Triton copy kernel.
        copy_into_buffer_2d(
            output_e4m3, buf.dispatch_out_data_raw, offset, num_tokens, buf.dim
        )
        copy_into_buffer_2d(
            output_scales_e8m0,
            buf.dispatch_out_scales_raw,
            offset,
            num_tokens,
            buf._scale_dim,
        )

        # Quantize w13 weights to FP8 along K (contraction dim) and
        # rearrange scales to blocked layout.
        # triton_to_mxfp8_dim0 quantizes along the last dim (K), matching
        # the forward GEMM's contraction axis.
        w13_e4m3, w13_scales = triton_to_mxfp8_dim0(
            w13, inner_block_size=block_size, scaling_mode="rceil"
        )
        w13_scales_blocked = triton_mx_block_rearrange_per_group_3d(w13_scales)

        # note: these had to be in row major layout for efficient transport with their
        # corresponding data over the a2a dispatch.
        # blocked layout would require a bunch of slow discontiguous reads and is harder to reason about.
        scale_a = output_scales_e8m0.view(torch.float8_e8m0fnu)
        scale_a_blocked = mx_block_rearrange_2d_M_groups_cuda(scale_a, group_end_offs)

        # First GEMM: h13 = x @ w13.T
        # We need this special CuTe DSL gmm kernel here, rather than torch._scaled_grouped_mm,
        # because it accepts out_offset=... arg, to write directly into the GPU+CPU buffer
        # for saving input activations, with zero copies.
        cutedsl_grouped_gemm(
            output_e4m3,
            w13_e4m3.transpose(-2, -1),
            scale_a_blocked,
            w13_scales_blocked,
            group_end_offs,
            self=buf.swiglu_input,
            out_offset=offset,
            out_dtype=torch.bfloat16,
        )

        # Fused SwiGLU forward: reads from saved-activations buffer at
        # GPU-determined offset, computes silu(h1)*h3, zero-pads to M rows.
        # No D2H sync (.item()) needed.
        from torchao.prototype.moe_training.ep.syncless.silu_mul_kernel import (
            silu_mul_fw,
        )

        M = output_e4m3.shape[0]
        h = silu_mul_fw(
            buf.swiglu_input,
            saved_activation_buffer_offset=offset,
            num_tokens=num_tokens,
            sym_mem_buffer_rows=M,
        )

        # Second GEMM: out = h @ w2.T (no out= — output goes to combine)
        out = _compute_fwd(
            h,
            w2.transpose(-2, -1),
            group_end_offs,
            block_size,
            torch.bfloat16,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
        )

        ctx.offset = offset
        ctx.num_tokens = num_tokens
        ctx.M = M
        ctx.hidden_dim = hidden_dim
        ctx.saved_activations_buffer = saved_activations_buffer
        ctx.save_for_backward(group_end_offs, w13, w2)

        return out

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_out):
        from torchao.prototype.moe_training.mxfp8_grouped_mm import (
            _compute_dgrad,
        )
        from torchao.prototype.mx_formats.config import ScaleCalculationMode
        from torchao.prototype.mx_formats.kernels import (
            triton_mxfp8_dequant_dim0,
        )
        from torchao.quantization.quantize_.common import KernelPreference

        group_end_offs, w13, w2 = ctx.saved_tensors
        offset = ctx.offset
        num_tokens = ctx.num_tokens
        M = ctx.M
        hidden_dim = ctx.hidden_dim
        buf = ctx.saved_activations_buffer
        block_size = 32

        # --- restore saved FP8 activations directly from buffer ---------------
        # TODO: remove when dequant kernel can accept (buffer, offset, len) args
        offset_cpu = offset.item()
        num_tokens_cpu = num_tokens.item()
        x_data = buf.dispatch_out_data[offset_cpu : offset_cpu + num_tokens_cpu]
        x_scales = buf.dispatch_out_scales[offset_cpu : offset_cpu + num_tokens_cpu]

        # TODO: fused dequant 1x32 -> requant 32x1 kernel
        x_bf16 = triton_mxfp8_dequant_dim0(
            x_data,
            x_scales.view(torch.uint8),
            out_dtype=torch.bfloat16,
            scale_block_size=block_size,
        )

        # w2 backward — compute dgrad first (needed by SwiGLU bwd kernel)
        # dgrad: grad_h = grad_out @ w2
        grad_h = _compute_dgrad(
            grad_out,
            w2.transpose(-2, -1),
            group_end_offs,
            block_size,
            torch.bfloat16,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
        )

        # Fused SwiGLU backward: recompute h from saved h13, compute grad_h13.
        # Reads from saved-activations buffer at GPU-determined offset.
        from torchao.prototype.moe_training.ep.syncless.silu_mul_kernel import (
            silu_mul_bw,
        )

        h = torch.empty(M, hidden_dim, device=grad_out.device, dtype=torch.bfloat16)
        grad_h13 = torch.empty(
            M, 2 * hidden_dim, device=grad_out.device, dtype=torch.bfloat16
        )
        silu_mul_bw(
            buf.swiglu_input,
            grad_h,
            saved_activation_buffer_offset=offset,
            num_tokens=num_tokens,
            h_out=h,
            grad_h13_out=grad_h13,
        )

        # wgrad: grad_w2 = grad_out.T @ h  (per group)
        grad_w2 = torch._grouped_mm(
            grad_out.transpose(-2, -1),
            h,
            offs=group_end_offs,
            out_dtype=torch.bfloat16,
        )

        # Pad x_bf16 to M rows for w13 wgrad.
        if num_tokens_cpu < M:
            pad_rows = M - num_tokens_cpu
            x_bf16 = torch.cat(
                [x_bf16, x_bf16.new_zeros(pad_rows, x_bf16.shape[1])], dim=0
            )

        # w13 backward
        # dgrad: grad_x = grad_h13 @ w13
        grad_x = _compute_dgrad(
            grad_h13,
            w13.transpose(-2, -1),
            group_end_offs,
            block_size,
            torch.bfloat16,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
        )

        # wgrad: grad_w13 = grad_h13.T @ x_bf16  (per group)
        grad_w13 = torch._grouped_mm(
            grad_h13.transpose(-2, -1),
            x_bf16,
            offs=group_end_offs,
            out_dtype=torch.bfloat16,
        )

        # free buffer allocation (GPU-side, no sync)
        buf.free(offset)

        return grad_x, None, None, grad_w13, grad_w2, None


class MXFP8GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        saved_activations_buffer,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.saved_activations_buffer = saved_activations_buffer

    def forward(
        self,
        output_e4m3: torch.Tensor,
        output_scales_e8m0: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        expert_padded_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w13, DTensor):
            w13 = self.w13.to_local()
            w2 = self.w2.to_local()
        else:
            w13 = self.w13
            w2 = self.w2

        return MXFP8GroupedExpertsFunc.apply(
            output_e4m3,
            output_scales_e8m0,
            num_tokens_per_expert,
            w13,
            w2,
            self.saved_activations_buffer,
        )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w13, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


# =============================================================================
# MoE model from torchtitan/models/moe/moe.py
# =============================================================================


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice with optional node limited routing
    top_k: int = 1
    num_expert_groups: int | None = None
    num_limited_groups: int | None = None
    use_grouped_mm: bool = True
    load_balance_coeff: float | None = 1e-3

    _debug_force_load_balance: bool = False


class FeedForward(nn.Module):
    """
    Dense FFN layer or shared experts in MoE layers.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice routing with optional node-limited routing."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_expert_groups: int | None,
        num_limited_groups: int | None,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_tokens = scores.size(0)
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.gate(x)

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class TokenReorderer(nn.Module):
    """Reorders token indices to match the order of experts."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class SynclessMXFP8MoE(nn.Module):
    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        saved_activations_buffer,
    ):
        super().__init__()

        num_experts = moe_args.num_experts
        self.experts = MXFP8GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            saved_activations_buffer=saved_activations_buffer,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            num_expert_groups=moe_args.num_expert_groups,
            num_limited_groups=moe_args.num_limited_groups,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts

        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        routed_input = x[token_indices_experts_sorted // self.router.top_k]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        out = self.shared_experts(x) if self.shared_experts is not None else None

        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )
        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
