import torch
import torch.nn as nn
import torch.nn.functional as F

from torchao.prototype.attention import apply_low_precision_attention


# Simple model with attention
class MyModel(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(attn_out.transpose(1, 2).contiguous().view(B, S, -1))


model = MyModel().to(device="cuda", dtype=torch.bfloat16).eval()

# Auto-detect best backend
model = apply_low_precision_attention(model)

# Or specify a backend explicitly
# model = apply_low_precision_attention(model, backend=AttentionBackend.FP8_FA3)

# Optional: torch.compile for RoPE fusion
model = torch.compile(model)
