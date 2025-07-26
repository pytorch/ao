import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Tuple, Callable

__all__ = [
    "moe_kernel",
    "basic_token_shuffle",
    "moe_calculation",
    "single_token_moe_kernel_linear_decomposition",
]


def moe_kernel(
    x: Tensor,
    expert_indices: Tensor,
    scores: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    act_fn: callable = F.silu
) -> Tensor:

    # return multi_token_expert_calculation_linear_decomposition(x, expert_indices, scores, up_proj, down_proj, act_fn)


    num_experts, expert_dim, hidden_dim = down_proj.shape
    num_token_activations = expert_indices.numel()

    # shuffle tokens
    ordered_token_indices, ordered_token_activations, offs = basic_token_shuffle(expert_indices, num_experts)
    ordered_inputs = x[ordered_token_indices]

    # get outputs
    ordered_outs = expert_calculation(ordered_inputs, up_proj, down_proj, offs, act_fn)

    # weigh outputs by score
    ordered_scores = scores.view(-1, 1)[ordered_token_activations]
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

def basic_token_shuffle(
    expert_indices: Tensor,
    num_experts: int
) -> Tuple[Tensor, Tensor, Tensor]:
    num_tokens, top_k = expert_indices.shape
    num_token_activations = num_tokens * top_k

    expert_indices = expert_indices.view(-1)

    ordered_token_activations = expert_indices.argsort(stable=True)
    ordered_token_indices = (
        ordered_token_activations.div(top_k)
        .floor()
        .to(torch.int32)
    )  #  [T]

    indices_for_histc = expert_indices if expert_indices.is_cuda else expert_indices.float() # histc doesn't work on cpu for integers
    num_tokens_per_expert = torch.histc(
        indices_for_histc,
        bins=num_experts, 
        min=0, 
        max=num_experts,
    )
    offs = num_tokens_per_expert.cumsum(dim=0).to(torch.int32)
    
    return ordered_token_indices, ordered_token_activations, offs

def expert_calculation(
    ordered_inputs: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    offs: Tensor,
    act_fn: Callable[Tensor, Tensor] = F.silu
) -> Tensor:
    x1, x3 = torch._grouped_mm(ordered_inputs, up_proj, offs).chunk(2, dim=1)
    y1 = act_fn(x1) * x3
    ordered_outs = torch._grouped_mm(y1, down_proj, offs)
    return ordered_outs

def decomposed_grouped_mm(
    ordered_inputs: Tensor, # [num_token_activations, in_features]
    proj: Tensor, # [num_experts, in_features, out_features]
    offs: Tensor # [num_experts]
) -> Tensor: # [num_token_activations, out_features]
    num_experts = proj.shape[0]

    new_offs = torch.zeros(offs.numel()+1, dtype=offs.dtype, device=offs.device)
    new_offs[1:] = offs
    inputs = [ordered_inputs[new_offs[i]:new_offs[i+1]] for i in range(num_experts)]

    outs = []
    for expert, cur_input in enumerate(inputs):
        cur_proj = proj[expert].t()
        cur_out = F.linear(cur_input, cur_proj)
        outs.append(cur_out)

    outputs = torch.cat(outs, dim=0)
    return outputs

def single_token_expert_calculation_linear_decomposition(
    x: Tensor,
    expert_indices: Tensor,
    scores: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    act_fn: Callable[Tensor, Tensor]=F.silu,
) -> Tensor:
    x = x.view(-1, x.shape[-1])
    assert x.shape[0] == 1, f"single_token_moe_kernel_linear_decomposition only works with inputs of shape [1, n] but got {x.shape}"
    num_activated_experts = expert_indices.numel()
    expert_indices = expert_indices.view(-1)

    cur_up_proj = up_proj[expert_indices]
    cur_down_proj = down_proj[expert_indices]

    outs = []
    for index in range(num_activated_experts):
        x1, x3 = F.linear(x, cur_up_proj[index]).chunk(2, dim=-1)
        y1 = act_fn(x1) * x3
        cur_out = F.linear(y1, cur_down_proj[index])
        outs.append(cur_out)
    
    out = torch.cat(outs, dim=0)
    final_out = out * scores.view(-1, 1)

    return final_out

@torch._dynamo.disable()
def _group_token_indices_by_expert(
    ordered_token_indices, offs, 
):
    token_indices_per_expert = [ordered_token_indices[:offs[0]]]
    token_indices_per_expert += [ordered_token_indices[start:end] for start,end in zip(offs[:-1],offs[1:])] 
    return token_indices_per_expert

def multi_token_expert_calculation_linear_decomposition(
    x: Tensor,
    expert_indices: Tensor,
    scores: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    act_fn: Callable[Tensor, Tensor],
):
    num_experts, expert_dim, hidden_dim = down_proj.shape
    num_token_activations = expert_indices.numel()

    # get token_shuffle_ordering
    ordered_token_indices, ordered_token_activations, offs = basic_token_shuffle(expert_indices, num_experts)
    token_indices_per_expert = _group_token_indices_by_expert(ordered_token_indices, offs)
    tokens_grouped_by_expert = [x[indices] for indices in token_indices_per_expert]

    # calculate outputs for each expert
    outs = []
    for expert, cur_x in enumerate(tokens_grouped_by_expert):
        cur_up_proj = up_proj[expert].t()
        cur_down_proj = down_proj[expert].t()

        x1, x3 = F.linear(cur_x, cur_up_proj).chunk(2, dim=1)
        y1 = act_fn(x1) * x3
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
