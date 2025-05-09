# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torchao.prototype.moe_quant.utils import FakeExtraDimTensor


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    num_experts: int = 8
    num_activated_experts: int = 2

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "Mixtral-8x7B-Instruct-v0.1": dict(
        block_size=32768,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        rope_base=1000000.0,
        num_experts=8,
        num_activated_experts=2,
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.block_sparse_moe = MOEFeedForwardAOQuantizable(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.block_sparse_moe(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


# class ConditionalFeedForward(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.w1 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim))
#         self.w2 = nn.Parameter(torch.empty(config.num_experts, config.dim, config.intermediate_size))
#         self.w3 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim))

#     def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
#         w1_weights = self.w1[expert_indices] # [T, A, D, D]
#         w3_weights = self.w3[expert_indices] # [T, A, D, D]
#         w2_weights = self.w2[expert_indices]  # [T, A, D, D]
#         x1 = F.silu(torch.einsum('ti,taoi -> tao', x, w1_weights))
#         x3 = torch.einsum('ti, taoi -> tao', x, w3_weights)
#         expert_outs =  torch.einsum('tao, taio -> tai', (x1 * x3), w2_weights)
#         return expert_outs


# class MOEFeedForward(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
#         self.cond_ffn = ConditionalFeedForward(config)
#         self.dim = config.dim
#         self.num_activated_experts = config.num_activated_experts
#     def forward(self, x: Tensor) -> Tensor:
#         x = x.view(-1, self.dim)
#         # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
#         # x: [T, D]
#         scores = self.gate(x) # [T, E]
#         expert_weights = F.softmax(scores, dim=-1)
#         expert_weights, expert_indices = torch.topk(expert_weights, self.num_activated_experts, dim=-1) # [T, A], [T, A]
#         expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # [T, A]
#         expert_outs = self.cond_ffn(x, expert_indices)
#         return torch.einsum('tai,ta -> ti', expert_outs, expert_weights)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


# T tokens
# E experts
# D dim
# I intermediate dim
# A activated experts
# T'(e) tokens for expert e


class MOEFeedForwardAOQuantizable(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForwardAOQuantizable(config)
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = x.view(-1, self.dim)  # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True).to(x.dtype)  # [T, A]
        out = self.cond_ffn(
            x, expert_indices, expert_weights, self.num_activated_experts
        )
        return out.reshape(batch_size, -1, self.dim)


class ConditionalFeedForwardAOQuantizable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )  # E, I, D
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.intermediate_size)
        )  # E, D, I
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )  # E, I, D
        self.num_experts = config.num_experts

    def forward(
        self,
        x: Tensor,  # T, D
        expert_indices: Tensor,  # T, A
        expert_weights: Tensor,  # T, A
        num_activated_experts: int,
    ) -> Tensor:
        num_tokens, dim = x.shape
        num_token_activations = num_tokens * num_activated_experts
        if x.shape[0] == 1 and not isinstance(
            self.w1, FakeExtraDimTensor
        ):  # only 1 token (can be done without graph breaks when compiled)
            outs = []
            expert_indices = expert_indices.view(num_activated_experts)
            # collect used experts
            w1 = self.w1[expert_indices]
            w2 = self.w2[expert_indices]
            w3 = self.w3[expert_indices]

            # run token through each expert
            for index in range(num_activated_experts):
                y1 = F.silu(F.linear(x, w1[index]))
                y3 = F.linear(x, w3[index])
                y2 = w2[index]
                cur_out = F.linear(y1 * y3, y2)
                outs.append(cur_out)

            # combine outputs
            final_out = (
                (torch.cat(outs, dim=0) * expert_weights.view(-1, 1))
                .sum(dim=0)
                .unsqueeze(-1)
            )
            return final_out
        else:
            expert_list = [x for x in range(self.num_experts)]

            # shuffle tokens into groups for each expert
            ordered_token_activations = expert_indices.view(-1).argsort(
                stable=True
            )  # [A]
            ordered_token_indices = (
                ordered_token_activations.div(num_activated_experts)
                .floor()
                .to(torch.int64)
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
                    ]
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

                cur_out = F.linear(
                    F.silu(F.linear(cur_x, w1)) * F.linear(cur_x, w3), w2
                )  # [T'(e), D]
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
                .expand(num_token_activations, dim)
                .to(torch.int64),
                src=weighted_ordered_outs,
            )
        return final_out
