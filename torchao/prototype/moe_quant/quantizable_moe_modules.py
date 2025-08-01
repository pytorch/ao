import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List

from torchao.prototype.moe_quant.kernels import moe_kernel

__all__ = [
    "MoEFeedForwardAOQuantizable",
    "ConditionalFeedForwardAOQuantizable",
]

class MoEFeedForwardAOQuantizable(nn.Module):
    def __init__(
        self,
        num_experts,
        hidden_dim,
        expert_dim,
        top_k,
        act_fn=F.silu,
        shared_expert=None,
        return_scores=False,
        use_grouped_mm=True,
        empty_init=True,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = ExpertsAOQuantizable(
            num_experts, hidden_dim, expert_dim, act_fn, use_grouped_mm, empty_init
        )
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.return_scores = return_scores

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)  # x: [T, H]

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
            return out.reshape(batch_size, -1, hidden_dim), scores
        else:
            return out.reshape(batch_size, -1, hidden_dim)

class ExpertsAOQuantizable(nn.Module):
    weight_attrs: List[str] = ["up_proj", "down_proj"]

    def __init__(self, num_experts, hidden_dim, expert_dim, act_fn=F.silu, use_grouped_mm=True, empty_init=True):
        super().__init__()
        if empty_init:
             # E, 2H, D
            self.up_proj = nn.Parameter(torch.empty(num_experts, 2*expert_dim, hidden_dim)) 
             # E, D, H
            self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, expert_dim)) 
        else:
            self.up_proj = nn.Parameter(torch.randn(num_experts, 2*expert_dim, hidden_dim))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, expert_dim))
        
        self.act_fn = act_fn
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: Tensor,  # T, D
        expert_indices: Tensor,  # T, A
        scores: Tensor,  # T, A
    ) -> Tensor:

        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if self.use_grouped_mm:
           final_out = self._forward_grouped_mm(x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn)
        elif x.shape[0] == 1:
            final_out = self._forward_single_token_linear_decomposition(x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn)
        else:
            final_out = self._forward_multi_token_linear_decomposition(x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn)
        return final_out.view(orig_shape)

    @staticmethod
    def _forward_grouped_mm(
        x: Tensor,
        expert_indices: Tensor,
        scores: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        act_fn: Callable[Tensor, Tensor],
    ):
        # get shapes
        num_experts, hidden_dim, expert_dim = down_proj.shape
        num_tokens, top_k = expert_indices.shape
        num_token_activations = num_tokens * top_k

        # token shuffle
        expert_indices = expert_indices.view(-1)
        ordered_token_activations = expert_indices.argsort(stable=True)
        ordered_token_indices = ordered_token_activations.div(top_k).floor().to(torch.int32)
        indices_for_histc = expert_indices if expert_indices.is_cuda else expert_indices.float() 
        num_tokens_per_expert = torch.histc(  # histc doesn't work on cpu for integers
            indices_for_histc,
            bins=num_experts, 
            min=0, 
            max=num_experts,
        )
        offs = num_tokens_per_expert.cumsum(dim=0).to(torch.int32)
        ordered_inputs = x[ordered_token_indices]
        ordered_scores = scores.view(-1,1)[ordered_token_activations]

        # calculate outputs
        gate, up = torch._grouped_mm(ordered_inputs, up_proj.transpose(-2, -1), offs).chunk(2, dim=1)
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

    assert x.shape[0] == 1, f"single_token_moe_kernel_linear_decomposition only works with inputs of shape [1, hidden_dim] but got {x.shape}"
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
    num_experts, hidden_dim, expert_dim = down_proj.shape
    num_token_activations = expert_indices.numel()

    # get token_shuffle_ordering
    ordered_token_indices, ordered_token_activations, offs = basic_token_shuffle(expert_indices, num_experts)
    token_indices_per_expert = _group_token_indices_by_expert(ordered_token_indices, offs)
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
    ordered_scores = scores.view(-1, 1)[ordered_token_activations]# [T*A, 1]
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
