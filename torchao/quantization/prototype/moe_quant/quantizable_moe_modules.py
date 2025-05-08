import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchao.quantization.prototype.moe_quant.utils import FakeExtraDimTensor


class MOEFeedForwardAOQuantizable(nn.Module):
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
        self.experts = ConditionalFeedForwardAOQuantizable(
            num_experts, hidden_dim, expert_dim, act_fn, empty_init
        )
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.return_scores = return_scores

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = x.view(-1, self.hidden_dim)  # x: [T, D]
        scores = self.router(x)  # [T, E]
        scores = F.softmax(scores, dim=-1)
        scores, expert_indices = torch.topk(
            scores, self.top_k, dim=-1
        )  # [T, A], [T, A]
        scores /= scores.sum(dim=-1, keepdim=True).to(x.dtype)  # [T, A]

        out = self.experts(x, expert_indices, scores, self.top_k)
        if self.shared_expert:
            out += self.shared_expert(x)

        if self.return_scores:
            return out.reshape(batch_size, -1, self.hidden_dim), scores
        else:
            return out.reshape(batch_size, -1, self.hidden_dim)


class ConditionalFeedForwardAOQuantizable(nn.Module):
    def __init__(self, num_experts, hidden_dim, expert_dim, act_fn, empty_init=True):
        super().__init__()
        if empty_init:
            self.w1 = nn.Parameter(
                torch.empty(num_experts, expert_dim, hidden_dim)
            )  # E, I, D
            self.w2 = nn.Parameter(
                torch.empty(num_experts, hidden_dim, expert_dim)
            )  # E, D, I
            self.w3 = nn.Parameter(
                torch.empty(num_experts, expert_dim, hidden_dim)
            )  # E, I, D
        else:
            self.w1 = nn.Parameter(
                torch.randn(num_experts, expert_dim, hidden_dim)
            )  # E, I, D
            self.w2 = nn.Parameter(
                torch.randn(num_experts, hidden_dim, expert_dim)
            )  # E, D, I
            self.w3 = nn.Parameter(
                torch.randn(num_experts, expert_dim, hidden_dim)
            )  # E, I, D
        self.num_experts = num_experts
        self.act_fn = act_fn
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim

    def forward(
        self,
        x: Tensor,  # T, D
        expert_indices: Tensor,  # T, A
        expert_weights: Tensor,  # T, A
        top_k: int,
    ) -> Tensor:
        num_tokens, _hidden_dim = x.shape
        num_token_activations = num_tokens * top_k

        if x.shape[0] == 1 and not isinstance(
            self.w1, FakeExtraDimTensor
        ):  # only 1 token (can be done without graph breaks when compiled)
            outs = []
            expert_indices = expert_indices.view(top_k)
            # collect used experts
            w1 = self.w1[expert_indices]
            w2 = self.w2[expert_indices]
            w3 = self.w3[expert_indices]
            # run token through each expert
            for index in range(top_k):
                y1 = F.silu(F.linear(x, w1[index]))
                y3 = F.linear(x, w3[index])
                y2 = w2[index]

                cur_out = F.linear(y1 * y3, y2)
                outs.append(cur_out)

            # combine outputs
            final_out = (
                (torch.cat(outs, dim=0) * expert_weights.view(-1, 1))
                .sum(dim=0)
                .reshape(x.shape)
            )
            return final_out
        else:
            expert_list = [x for x in range(self.num_experts)]

            # shuffle tokens into groups for each expert
            ordered_token_activations = expert_indices.view(-1).argsort(
                stable=True
            )  # [A]
            ordered_token_indices = (
                ordered_token_activations.div(top_k).floor().to(torch.int64)
            )  #  [T]
            if not expert_indices.is_cuda:  # histc doesn't work on cpu for integers
                num_tokens_per_expert = torch.bincount(
                    expert_indices.view(-1) + 1, minlength=self.num_experts + 1
                )
            else:
                num_tokens_per_expert = torch.histc(
                    expert_indices,
                    bins=self.num_experts + 1,
                    min=-1,
                    max=self.num_experts,
                )  #  [E+1] (added leading 0 so can be used for indexing)
            cum_tokens_per_expert = num_tokens_per_expert.cumsum(0).to(
                torch.int64
            )  #  [E+1]

            @torch._dynamo.disable()
            def group_tokens_by_expert(
                ordered_token_indices, cum_tokens_per_expert, expert_list
            ):
                token_indices_per_expert = [
                    ordered_token_indices[
                        cum_tokens_per_expert[expert] : cum_tokens_per_expert[
                            expert + 1
                        ]
                    ].to(torch.int64)
                    for expert in expert_list
                ]  # [T'(e1)], [T'(e2)] ...
                return token_indices_per_expert

            token_indices_per_expert = group_tokens_by_expert(
                ordered_token_indices, cum_tokens_per_expert, expert_list
            )
            tokens_grouped_by_expert = [
                x[indices] for indices in token_indices_per_expert
            ]

            # calculate outputs for each expert
            outs = []
            for cur_x, expert in zip(tokens_grouped_by_expert, expert_list):
                w1 = self.w1[expert]  # I, D
                w2 = self.w2[expert]  # D, I
                w3 = self.w3[expert]  # I, D

                y1 = F.silu(F.linear(cur_x, w1))
                y3 = F.linear(cur_x, w3)
                y2 = w2

                cur_out = F.linear(y1 * y3, y2)  # [T'(e), D]
                outs.append(cur_out)

            # weigh outputs
            ordered_outs = torch.cat(outs, dim=0)  # [T*A, D]
            ordered_token_activation_weights = expert_weights.view(-1, 1)[
                ordered_token_activations
            ].view(-1, 1)  # [T*A, 1]
            weighted_ordered_outs = (
                ordered_outs * ordered_token_activation_weights
            )  # [T*A, D]

            # sum weighted token-activation outputs together for each token
            final_out = torch.zeros_like(x)  #  [T, D]
            final_out = final_out.scatter_add(
                dim=0,
                index=ordered_token_indices.unsqueeze(-1)
                .expand(num_token_activations, self.hidden_dim)
                .to(torch.int64),
                src=weighted_ordered_outs,
            )
        return final_out
