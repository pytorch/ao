"""
Numerical equivalence test: original SAM2 complex RoPE vs new F.apply_rotary_emb.

Verifies that compute_2d_axial_rope_frequencies + F.apply_rotary_emb(interleaved=True)
produces identical results to the original compute_axial_cis + apply_rotary_enc.
"""
import sys
from typing import Tuple

import torch
import torch.nn.functional as F

# Shim F.apply_rotary_emb for PyTorch versions that don't include it yet
if not hasattr(F, "apply_rotary_emb"):
    def _rotate_half(x, interleaved=False):
        if interleaved:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)
        else:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_emb(query, key, cos, sin, seq_dim=1, interleaved=False):
        cos = cos.narrow(0, 0, query.size(seq_dim))
        sin = sin.narrow(0, 0, query.size(seq_dim))
        if seq_dim == 1:
            cos_b = cos.unsqueeze(0).unsqueeze(2)
            sin_b = sin.unsqueeze(0).unsqueeze(2)
        elif seq_dim == 2:
            cos_b = cos.unsqueeze(0).unsqueeze(0)
            sin_b = sin.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"seq_dim must be 1 or 2, got {seq_dim}")
        query_rot = query * cos_b + _rotate_half(query, interleaved) * sin_b
        key_rot = key * cos_b + _rotate_half(key, interleaved) * sin_b
        return query_rot, key_rot

    F.apply_rotary_emb = _apply_rotary_emb

from torchao._models.sam2.modeling.position_encoding import (
    compute_2d_axial_rope_frequencies,
)


# ── Original SAM2 implementations (copied from before our changes) ──

def orig_init_t_xy(end_x, end_y, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def orig_compute_axial_cis(dim, end_x, end_y, theta=10000.0, device=None):
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    t_x, t_y = orig_init_t_xy(end_x, end_y, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def orig_reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def orig_apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = orig_reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# ── Tests ──

def check(name, a, b, atol=1e-5):
    diff = (a - b).abs().max().item()
    ok = diff < atol
    print(f"  {'PASS' if ok else 'FAIL'}: {name} (max diff: {diff:.2e})")
    return ok


def test_basic_attention():
    """SAM2 default: 32x32 features, dim=32."""
    print("=== test_basic_attention (SAM2 decoder) ===")
    device = "cuda"
    B, heads, dim = 2, 8, 32
    L_x, L_y = 32, 32
    L = L_x * L_y

    q = torch.randn(B, heads, L, dim, device=device)
    k = torch.randn(B, heads, L, dim, device=device)

    freqs_cis = orig_compute_axial_cis(dim, L_x, L_y, device=device)
    q_orig, k_orig = orig_apply_rotary_enc(q, k, freqs_cis)

    cos, sin = compute_2d_axial_rope_frequencies(dim, L_x, L_y)
    cos, sin = cos.to(device), sin.to(device)
    q_new, k_new = F.apply_rotary_emb(q, k, cos, sin, seq_dim=2, interleaved=True)

    ok = check("basic q", q_orig, q_new)
    ok &= check("basic k", k_orig, k_new)
    return ok


def test_k_repeat():
    """Cross-attention with rope_k_repeat."""
    print("\n=== test_k_repeat (cross-attention to memories) ===")
    device = "cuda"
    B, heads, dim = 2, 8, 32
    L_x, L_y = 32, 32
    L_q = L_x * L_y
    L_k = L_q * 3

    q = torch.randn(B, heads, L_q, dim, device=device)
    k = torch.randn(B, heads, L_k, dim, device=device)

    freqs_cis = orig_compute_axial_cis(dim, L_x, L_y, device=device)
    q_orig, k_orig = orig_apply_rotary_enc(q, k, freqs_cis, repeat_freqs_k=True)

    cos, sin = compute_2d_axial_rope_frequencies(dim, L_x, L_y)
    cos, sin = cos.to(device), sin.to(device)
    q_new, _ = F.apply_rotary_emb(q, q, cos, sin, seq_dim=2, interleaved=True)
    r = L_k // L_q
    cos_k = cos.repeat(r, 1)
    sin_k = sin.repeat(r, 1)
    _, k_new = F.apply_rotary_emb(k, k, cos_k, sin_k, seq_dim=2, interleaved=True)

    ok = check("k_repeat q", q_orig, q_new)
    ok &= check("k_repeat k", k_orig, k_new)
    return ok


def test_exclude_rope():
    """RoPE with num_k_exclude_rope."""
    print("\n=== test_exclude_rope ===")
    device = "cuda"
    B, heads, dim = 2, 8, 32
    L_x, L_y = 32, 32
    L_q = L_x * L_y
    num_exclude = 4
    L_k = L_q + num_exclude

    q = torch.randn(B, heads, L_q, dim, device=device)
    k = torch.randn(B, heads, L_k, dim, device=device)

    freqs_cis = orig_compute_axial_cis(dim, L_x, L_y, device=device)
    num_k_rope = L_k - num_exclude
    q_orig, k_rope_orig = orig_apply_rotary_enc(q, k[:, :, :num_k_rope], freqs_cis)

    cos, sin = compute_2d_axial_rope_frequencies(dim, L_x, L_y)
    cos, sin = cos.to(device), sin.to(device)
    q_new, k_rope_new = F.apply_rotary_emb(
        q, k[:, :, :num_k_rope], cos, sin, seq_dim=2, interleaved=True
    )

    ok = check("exclude q", q_orig, q_new)
    ok &= check("exclude k_rope", k_rope_orig, k_rope_new)
    return ok


def test_compile_fullgraph():
    """Verify F.apply_rotary_emb compiles fullgraph."""
    print("\n=== test_compile_fullgraph ===")
    device = "cuda"
    B, heads, dim = 2, 8, 32
    L_x, L_y = 32, 32
    L = L_x * L_y

    cos, sin = compute_2d_axial_rope_frequencies(dim, L_x, L_y)
    cos, sin = cos.to(device), sin.to(device)

    def rope_fn(q, k, cos, sin):
        return F.apply_rotary_emb(q, k, cos, sin, seq_dim=2, interleaved=True)

    q = torch.randn(B, heads, L, dim, device=device)
    k = torch.randn(B, heads, L, dim, device=device)

    q_eager, k_eager = rope_fn(q, k, cos, sin)
    compiled_fn = torch.compile(rope_fn, fullgraph=True)
    q_compiled, k_compiled = compiled_fn(q, k, cos, sin)

    ok = check("compile q", q_compiled, q_eager)
    ok &= check("compile k", k_compiled, k_eager)
    return ok


def test_bfloat16():
    """Test with bfloat16 (SAM2's typical inference dtype)."""
    print("\n=== test_bfloat16 ===")
    device = "cuda"
    B, heads, dim = 1, 8, 32
    L_x, L_y = 32, 32
    L = L_x * L_y

    q = torch.randn(B, heads, L, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, heads, L, dim, device=device, dtype=torch.bfloat16)

    freqs_cis = orig_compute_axial_cis(dim, L_x, L_y, device=device)
    q_orig, k_orig = orig_apply_rotary_enc(q, k, freqs_cis)

    cos, sin = compute_2d_axial_rope_frequencies(dim, L_x, L_y)
    cos, sin = cos.to(device), sin.to(device)
    q_new, k_new = F.apply_rotary_emb(
        q.float(), k.float(), cos, sin, seq_dim=2, interleaved=True
    )
    q_new, k_new = q_new.to(torch.bfloat16), k_new.to(torch.bfloat16)

    ok = check("bf16 q", q_orig, q_new, atol=5e-2)
    ok &= check("bf16 k", k_orig, k_new, atol=5e-2)
    return ok


if __name__ == "__main__":
    torch.manual_seed(42)
    all_ok = True
    all_ok &= test_basic_attention()
    all_ok &= test_k_repeat()
    all_ok &= test_exclude_rope()
    all_ok &= test_compile_fullgraph()
    all_ok &= test_bfloat16()

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)
