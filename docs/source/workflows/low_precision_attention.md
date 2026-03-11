# Low-Precision Attention (Prototype)

FP8 low-precision attention for inference, built on Flash Attention backends. Currently supports FA3 on Hopper (SM90) architectures, with FA4 for Blackwell coming soon.

> **Requirements:** PyTorch >= 2.11, Hopper GPU (H100/H200), `flash-attn` with FA3 support.

> **Note:** Only the forward pass is supported — backward is not supported by the underlying backends.

## High-Level API

The simplest way to enable low-precision attention is `apply_low_precision_attention`, which wraps your model to replace all `F.scaled_dot_product_attention` calls with FP8 attention:

```python
from torchao.prototype.attention import apply_low_precision_attention

model = ...  # your model
model = apply_low_precision_attention(model)
```

This must be called **before** `torch.compile`.

### Disabling KV Caching

KV caching should be disabled before calling `apply_low_precision_attention` (e.g., `config.use_cache = False` for HuggingFace models).

With KV caching enabled, HuggingFace models materialize an explicit attention mask for causal layers, which blocks Flash Attention from running. The monkey-patch path detects and strips these causal masks automatically, so **FP8 attention still works in eager mode with KV caching enabled**.

However, **RoPE fusion under `torch.compile` requires KV caching to be disabled.** KV caching inserts a `torch.cat` operation between RoPE and SDPA (to concatenate cached and new keys/values), which breaks the pattern matching required for the fusion pass — it expects RoPE to feed directly into SDPA. With KV caching enabled, you will still get FP8 attention but without the RoPE fusion optimization.

### RoPE Fusion with `torch.compile`

If you then `torch.compile` the wrapped model, the compiler will automatically detect RoPE patterns preceding `F.scaled_dot_product_attention` and fuse them into a single `fp8_fa3_rope_sdpa` kernel:

```python
model = apply_low_precision_attention(model)
model = torch.compile(model)  # RoPE fusion happens automatically
```

The fusion pass supports two RoPE patterns:
- **NeoX/LLaMA style** (half-split): `x * cos + rotate_half(x) * sin`
- **Interleaved style** (FLUX): complex rotation via reshape + unbind + stack

> **Warning:** The RoPE fusion pass sets `torch._inductor.config.pre_grad_custom_pass`. This will overwrite any existing custom pass you may have registered.

### Selecting a Backend

By default, `apply_low_precision_attention` auto-detects the best available backend. You can also specify one explicitly:

```python
from torchao.prototype.attention import apply_low_precision_attention, AttentionBackend

model = apply_low_precision_attention(model, backend=AttentionBackend.FP8_FA3)
```

| Backend | Architecture | Status |
|---------|-------------|--------|
| `FP8_FA3` | Hopper (SM90) | Available |
| `FP8_FA4` | Blackwell (SM100) | Coming soon |

## Direct Usage

For finer-grained control, you can use `fp8_fa3_sdpa` and `fp8_fa3_rope_sdpa` directly as drop-in replacements for `F.scaled_dot_product_attention`.

### `fp8_fa3_sdpa` — Drop-in SDPA Replacement

Replaces `F.scaled_dot_product_attention`. Input/output layout is `[B, H, S, D]`.

```python
from torch.nn.attention import activate_flash_attention_impl, restore_flash_attention_impl
from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

activate_flash_attention_impl("FA3")
try:
    out = fp8_fa3_sdpa(query, key, value, is_causal=True)
finally:
    restore_flash_attention_impl()
```

**Parameters:**
- `query`, `key`, `value` — `[B, H, S, D]` tensors
- `is_causal` (`bool`, default `False`) — Whether to apply causal masking
- `scale` (`float | None`, default `None`) — Attention scale factor (defaults to `1/sqrt(D)`)
- `enable_gqa` (`bool`, default `False`) — Enable grouped-query attention

> **Note:** `attn_mask` and `dropout_p` are accepted for signature compatibility but must be `None` and `0.0` respectively.

### `fp8_fa3_rope_sdpa` — Fused RoPE + SDPA

Fuses RoPE application with FP8 attention in a single kernel. Input layout is `[B, S, H, D]` (pre-transpose); output layout is `[B, H, S, D]`.

```python
from torch.nn.attention import activate_flash_attention_impl, restore_flash_attention_impl
from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_rope_sdpa

activate_flash_attention_impl("FA3")
try:
    out = fp8_fa3_rope_sdpa(
        query, key, value, cos, sin,
        is_causal=True,
        rope_interleaved=False,  # NeoX/LLaMA style
    )
finally:
    restore_flash_attention_impl()
```

**Additional parameters:**
- `cos`, `sin` — RoPE frequency tensors, shape `[S, D]`
- `rope_interleaved` (`bool`, default `False`) — `False` for NeoX/LLaMA half-split rotation, `True` for FLUX-style interleaved rotation
