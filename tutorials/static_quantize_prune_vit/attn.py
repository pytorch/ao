from torch import nn
# from einops import rearrange, repeat
from typing import Any, Optional, Callable
import torch


class CompositeMHA(torch.nn.Module):
    def __init__(self, num_heads, in_proj_weight, in_proj_bias, out_proj, batch_first):
        super().__init__()
        # Make it a nn.Linear layer so it can be more easily replaced
        self.in_proj = torch.nn.Linear(8, 8, bias=True)
        self.in_proj.to(device=in_proj_bias.device, dtype=in_proj_bias.dtype)
        self.in_proj.in_features = in_proj_weight.size(1)
        self.in_proj.out_features = in_proj_weight.size(0)
        self.in_proj.weight = in_proj_weight
        self.in_proj.bias = in_proj_bias
        self.out_proj = out_proj
        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        assert self.batch_first
        assert not need_weights
        assert key_padding_mask is None
        assert query.dim() == 3

        batch_size = query.size(0)
        embed_dim = query.size(2)
        head_dim = embed_dim // (self.num_heads)

        assert query is key
        assert key is value

        E = query.size(-1)
        proj = self.in_proj(query)
        proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        query, key, value = proj[0], proj[1], proj[2]

        query = query.view(batch_size, -1, self.num_heads,
                           head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads,
                       head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads,
                           head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * head_dim)
        # Match return signature of nn.MHA
        return self.out_proj(attn), None


def build_composite_mha_from_nn_mha(pt):
    assert pt._qkv_same_embed_dim
    in_proj_weight = pt.in_proj_weight
    assert in_proj_weight is not None
    return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj, pt.batch_first)

def apply_attn(model):
    from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
    _replace_with_custom_fn_if_matches_filter(
        model,
        build_composite_mha_from_nn_mha,
        lambda mod, fqn: isinstance(mod, torch.nn.MultiheadAttention))
