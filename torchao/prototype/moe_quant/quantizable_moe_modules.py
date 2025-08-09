from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "MoEFeedForwardAOQuantizable",
    "ExpertsAOQuantizable",
]


class MoEFeedForwardAOQuantizable(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        expert_dim: int,
        top_k: int,
        act_fn: Callable[Tensor, Tensor] = F.silu,
        shared_expert: torch.nn.Module = None,
        return_scores: bool = False,
        decompose_grouped_mm: bool = False,
        empty_init: bool = True,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = ExpertsAOQuantizable(
            num_experts,
            hidden_dim,
            expert_dim,
            act_fn,
            decompose_grouped_mm,
            empty_init,
        )
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.return_scores = return_scores

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])  # x: [T, H]
        scores = self.router(x)  # [T, E]
        scores = F.softmax(scores, dim=-1)
        scores, expert_indices = torch.topk(
            scores, self.top_k, dim=-1
        )  # [T, K], [T, K]
        scores /= scores.sum(dim=-1, keepdim=True).to(x.dtype)  # [T, K]

        out = self.experts(x, expert_indices, scores)
        if self.shared_expert:
            out += self.shared_expert(x)

        if self.return_scores:
            return out.view(*orig_shape), scores
        else:
            return out.view(*orig_shape)


class ExpertsAOQuantizable(nn.Module):
    weight_attrs: List[str] = ["up_proj", "down_proj"]

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        expert_dim: int,
        act_fn: Callable[Tensor, Tensor] = F.silu,
        decompose_grouped_mm: bool = False,
        empty_init: bool = True,
    ):
        super().__init__()
        if empty_init:
            # E, 2H, D
            self.up_proj = nn.Parameter(
                torch.empty(num_experts, 2 * expert_dim, hidden_dim)
            )
            # E, D, H
            self.down_proj = nn.Parameter(
                torch.empty(num_experts, hidden_dim, expert_dim)
            )
        else:
            self.up_proj = nn.Parameter(
                torch.randn(num_experts, 2 * expert_dim, hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.randn(num_experts, hidden_dim, expert_dim)
            )

        self.act_fn = act_fn
        self.decompose_grouped_mm = decompose_grouped_mm

    def forward(
        self,
        x: Tensor,  # T, D
        expert_indices: Tensor,  # T, A
        scores: Tensor,  # T, A
    ) -> Tensor:
        if not self.decompose_grouped_mm:
            final_out = self._forward_grouped_mm(
                x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn
            )
        elif x.shape[0] > 1 or "FakeExtraDimTensor" in str(type(self.up_proj)):
            final_out = self._forward_multi_token_linear_decomposition(
                x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn
            )
        else:
            final_out = self._forward_single_token_linear_decomposition(
                x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn
            )

        return final_out

    @staticmethod
    def _forward_grouped_mm(
        x: Tensor,
        expert_indices: Tensor,
        scores: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        act_fn: Callable[Tensor, Tensor],
    ):
        assert hasattr(torch, "_grouped_mm"), (
            "the _grouped_mm op was not found, try installing pytorch nightly or test with: >python -c 'import torch; print(torch._grouped_mm)'"
        )

        # get shapes
        num_experts, hidden_dim, expert_dim = down_proj.shape
        num_tokens, top_k = expert_indices.shape
        num_token_activations = num_tokens * top_k

        # token shuffle
        expert_indices = expert_indices.view(-1)
        ordered_token_activations = expert_indices.argsort(stable=True)
        ordered_token_indices = (
            ordered_token_activations.div(top_k).floor().to(torch.int32)
        )
        indices_for_histc = (
            expert_indices if expert_indices.is_cuda else expert_indices.float()
        )
        num_tokens_per_expert = torch.histc(  # histc doesn't work on cpu for integers
            indices_for_histc,
            bins=num_experts,
            min=0,
            max=num_experts,
        )
        offs = num_tokens_per_expert.cumsum(dim=0).to(torch.int32)
        ordered_inputs = x[ordered_token_indices]
        ordered_scores = scores.view(-1, 1)[ordered_token_activations]

        # calculate outputs
        gate, up = torch._grouped_mm(
            ordered_inputs, up_proj.transpose(-2, -1), offs
        ).chunk(2, dim=1)
        y1 = act_fn(gate) * up
        ordered_outs = torch._grouped_mm(y1, down_proj.transpose(-2, -1), offs)
        ordered_weighted_outs = ordered_scores * ordered_outs

        # un-shuffle output
        final_out = torch.zeros_like(x)
        final_out = final_out.scatter_add(
            dim=0,
            index=ordered_token_indices.unsqueeze(-1)
            .expand(num_token_activations, hidden_dim)
            .to(torch.int64),
            src=ordered_weighted_outs,
        )
        return final_out

    @staticmethod
    def _forward_single_token_linear_decomposition(
        x: Tensor,
        expert_indices: Tensor,
        scores: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        act_fn: Callable[Tensor, Tensor],
    ):
        # get shapes
        assert x.shape[0] == 1 and x.dim() == 2, (
            f"single_token_moe_kernel_linear_decomposition only works with inputs of shape [1, hidden_dim] but got {x.shape}"
        )
        num_activated_experts = expert_indices.numel()
        expert_indices = expert_indices.view(-1)

        # collect only experts that get activated
        cur_up_proj = up_proj[expert_indices]
        cur_down_proj = down_proj[expert_indices]

        # calculate outputs
        outs = []
        for index in range(num_activated_experts):
            gate, up = F.linear(x, cur_up_proj[index]).chunk(2, dim=-1)
            y1 = act_fn(gate) * up
            cur_out = F.linear(y1, cur_down_proj[index])
            outs.append(cur_out)

        # combine output
        out = torch.cat(outs, dim=0)
        final_out = (out * scores.view(-1, 1)).sum(dim=0).unsqueeze(0)
        return final_out

    @staticmethod
    def _forward_multi_token_linear_decomposition(
        x: Tensor,
        expert_indices: Tensor,
        scores: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        act_fn: Callable[Tensor, Tensor],
    ):
        @torch._dynamo.disable()
        def _group_tokens_by_expert(ordered_token_indices, cum_tokens_per_expert):
            num_experts = cum_tokens_per_expert.numel() - 1
            token_indices_per_expert = [
                ordered_token_indices[
                    cum_tokens_per_expert[expert] : cum_tokens_per_expert[expert + 1]
                ]
                for expert in range(num_experts)
                if cum_tokens_per_expert[expert] < cum_tokens_per_expert[expert + 1]
            ]  # [T'(e1)], [T'(e2)] ...
            return token_indices_per_expert

        # get shapes
        num_experts, hidden_dim, expert_dim = down_proj.shape
        num_tokens, top_k = expert_indices.shape
        num_token_activations = num_tokens * top_k

        # token shuffle
        expert_indices = expert_indices.view(-1)
        ordered_token_activations = expert_indices.argsort(stable=True)
        ordered_token_indices = (
            ordered_token_activations.div(top_k).floor().to(torch.int32)
        )
        indices_for_histc = (
            expert_indices if expert_indices.is_cuda else expert_indices.float()
        )
        num_tokens_per_expert = torch.histc(  # histc doesn't work on cpu for integers
            indices_for_histc,
            bins=num_experts + 1,
            min=-1,
            max=num_experts,
        )
        cum_tokens_per_expert = num_tokens_per_expert.cumsum(0).to(torch.int64)
        token_indices_per_expert = _group_tokens_by_expert(
            ordered_token_indices, cum_tokens_per_expert
        )
        tokens_grouped_by_expert = [x[indices] for indices in token_indices_per_expert]

        # calculate outputs for each expert
        outs = []
        for expert, cur_x in enumerate(tokens_grouped_by_expert):
            cur_up_proj = up_proj[expert]
            cur_down_proj = down_proj[expert]

            gate, up = F.linear(cur_x, cur_up_proj).chunk(2, dim=1)
            y1 = act_fn(gate) * up
            cur_out = F.linear(y1, cur_down_proj)
            outs.append(cur_out)

        # weigh outputs
        ordered_outs = torch.cat(outs, dim=0)  # [T*A, D]
        ordered_scores = scores.view(-1, 1)[ordered_token_activations]  # [T*A, 1]
        ordered_weighted_outs = ordered_scores * ordered_outs

        # un-shuffle outputs
        final_out = torch.zeros_like(x)
        final_out = final_out.scatter_add(
            dim=0,
            index=ordered_token_indices.unsqueeze(-1)
            .expand(num_token_activations, hidden_dim)
            .to(torch.int64),
            src=ordered_weighted_outs,
        )
        return final_out
