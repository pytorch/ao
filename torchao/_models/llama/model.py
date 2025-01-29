# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torchao.utils import find_multiple


# TODO remove suplerfluous arg
def prepare_inputs_for_model(inps, max_new_tokens=1):
    # this is because input from lm-eval is 2d
    if inps.dim() > 2:
        raise ValueError(f"Expected input to be of dim 1 or 2, but got {inps.dim()}")

    input_pos = torch.arange(0, inps.numel(), device=inps.device)
    return (inps.view(1, -1), input_pos)


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
    use_scaled_rope: bool = False
    tie_word_embeddings: bool = False

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

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(
                config[1]
            ), name  # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": dict(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "Llama-3-8B": dict(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "Llama-3.1-8B": dict(
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        use_scaled_rope=True,
    ),
    "Llama-3.1-70B": dict(
        block_size=131072,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        use_scaled_rope=True,
    ),
    "Llama-3.1-405B": dict(
        block_size=131072,
        n_layer=126,
        n_head=128,
        n_local_heads=8,
        dim=16384,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        use_scaled_rope=True,
    ),
    "Llama-3.2-3B": dict(
        block_size=131072,
        n_layer=28,
        n_head=24,
        n_local_heads=8,
        dim=3072,
        intermediate_size=8192,
        vocab_size=128256,
        rope_base=500000,
        use_scaled_rope=True,
        tie_word_embeddings=True,
    ),
}

# this is a model specific variable that controls whether index_put is used for the kv_cache update,
# it is needed for GPTQ but otherwise attenuates perf so the default is to not use it
use_index_put_for_kv_cache = False


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

        if use_index_put_for_kv_cache:
            k_out = torch.ops.aten.index_put_(
                self.k_cache, [None, None, input_pos], k_val
            )
            v_out = torch.ops.aten.index_put_(
                self.v_cache, [None, None, input_pos], v_val
            )
        else:
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

        return k_out, v_out


from torchao.quantization.utils import quantize_activation_per_token_absmax


class AffineQuantizedKVCache(nn.Module):
    def __init__(
        self,
        max_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        scale_dtype=torch.bfloat16,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        scale_shape = (max_batch_size, n_heads, max_seq_length, 1)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=torch.int8))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=torch.int8))
        self.register_buffer(
            "k_cache_scale", torch.ones(scale_shape, dtype=scale_dtype)
        )
        self.register_buffer(
            "v_cache_scale", torch.ones(scale_shape, dtype=scale_dtype)
        )

    def update(self, input_pos, k_val, v_val):
        # quantize current k_val and store it in the cache
        q_k_val, k_scale = quantize_activation_per_token_absmax(k_val)
        self.k_cache[:, :, input_pos] = q_k_val
        self.k_cache_scale[:, :, input_pos] = k_scale.unsqueeze(-1)
        k_out = self.k_cache * self.k_cache_scale
        k_out[:, :, input_pos] = k_val

        q_v_val, v_scale = quantize_activation_per_token_absmax(v_val)
        self.v_cache[:, :, input_pos] = q_v_val
        self.v_cache_scale[:, :, input_pos] = v_scale.unsqueeze(-1)
        v_out = self.v_cache * self.v_cache_scale
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

    @classmethod
    def from_float(cls, kv_cache):
        cache_shape = kv_cache.k_cache.shape
        max_batch_size, n_heads, max_seq_length, head_dim = cache_shape
        scale_dtype = kv_cache.k_cache.dtype
        return cls(max_batch_size, max_seq_length, n_heads, head_dim, scale_dtype)


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

    def setup_caches(
        self,
        max_batch_size,
        max_seq_length,
        training: bool = False,
        kv_cache_quantization=None,
        linear_causal_mask=False,
        prompt_length=None,
    ):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = None
        # module swaps can cause issues without this
        if hasattr(self.output, "weight"):
            dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype

        self.linear_causal_mask = linear_causal_mask
        if not self.linear_causal_mask:
            self.causal_mask = torch.tril(
                torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
            )
        else:
            assert (
                prompt_length is not None and prompt_length > 1
            ), "need to set prompt_length>1 to use non quadratic causal mask in setup_caches"
            self.causal_mask = torch.zeros(
                1, 1, 1, self.max_seq_length, dtype=torch.bool
            )
            self.causal_mask[:, :, :, :prompt_length] = 1

        if not training:
            for b in self.layers:
                if kv_cache_quantization:
                    with torch.device("meta"):
                        b.attention.kv_cache = KVCache(
                            max_batch_size,
                            max_seq_length,
                            self.config.n_local_heads,
                            head_dim,
                            dtype,
                        )
                    b.attention.kv_cache = AffineQuantizedKVCache.from_float(
                        b.attention.kv_cache
                    )
                else:
                    b.attention.kv_cache = KVCache(
                        max_batch_size,
                        max_seq_length,
                        self.config.n_local_heads,
                        head_dim,
                        dtype,
                    )
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            dtype,
            use_scaled=self.config.use_scaled_rope,
        )

    def reset_caches(self):
        """Reset caches.

        The caches used by training stage and inference stage may be different, reset them before switching.
        """
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the model.

        Args:
            idx  (`torch.LongTensor` of shape `(batch_size, seq_length)`):
                Indices of input sequence tokens in the vocabulary.
            input_pos (`torch.LongTensor` of shape `(batch_size, seq_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
                This argument is optional for training mode but required for
                inference mode(when model.setup_caches(training=False) is used).

        Returns:
            Tensor: The output logits tensor.
        """
        assert self.freqs_cis is not None, "Caches must be initialized first"

        if input_pos is None:
            mask = None
            freqs_cis = self.freqs_cis[: idx.shape[1]]
        else:
            if not self.linear_causal_mask:
                mask = self.causal_mask[None, None, input_pos]
            elif (
                len(input_pos) > 1 and self.linear_causal_mask
            ):  # prefill for linear causal mask
                mask = (
                    torch.tril(
                        torch.ones(
                            len(input_pos),
                            self.max_seq_length,
                            dtype=torch.bool,
                            device=input_pos.device,
                        )
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            else:  # decode_one_token for linear causal mask
                self.causal_mask[0, 0, 0, input_pos] = 1
                mask = self.causal_mask
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
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor],
        freqs_cis: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
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
        mask: Optional[Tensor],
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
        if mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    use_scaled: bool = False,
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


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
