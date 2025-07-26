import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List

from torchao.prototype.moe_quant.utils import FakeExtraDimTensor
from torchao.prototype.moe_quant.kernels import moe_kernel

__all__ = [
    "MoEFeedForwardAOQuantizable",
    "ConditionalFeedForwardAOQuantizable",
]

class MoEFeedForwardAOQuantizable(nn.Module):
    def __init__(
        self,
        hidden_dim,
        expert_dim,
        num_experts,
        top_k,
        act_fn=F.silu,
        shared_expert=None,
        return_scores=False,
        empty_init=True,
    ) -> None:
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = ExpertsAOQuantizable(
            num_experts, hidden_dim, expert_dim, act_fn, empty_init
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

    def __init__(self, num_experts, hidden_dim, expert_dim, act_fn, empty_init=True):
        super().__init__()
        if empty_init:
            self.up_proj = nn.Parameter(
                torch.empty(num_experts, hidden_dim, 2*expert_dim)
            )  # E, D, H
            self.down_proj = nn.Parameter(
                torch.empty(num_experts, expert_dim, hidden_dim)
            )  # E, H, D
        else:
            self.up_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, 2*expert_dim))
            self.down_proj = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim))

        self.act_fn = act_fn

    def forward(
        self,
        x: Tensor,  # T, D
        expert_indices: Tensor,  # T, A
        scores: Tensor,  # T, A
    ) -> Tensor:

        return moe_kernel(x, expert_indices, scores, self.up_proj, self.down_proj, self.act_fn)
