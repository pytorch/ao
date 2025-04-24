# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import re

import torch
import torch.nn as nn


# TODO: Refactor torchao and tests to use these models
class ToyLinearModel(torch.nn.Module):
    def __init__(self, k=64, n=32, dtype=torch.bfloat16):
        super().__init__()
        self.linear1 = torch.nn.Linear(k, n, bias=False).to(dtype)

    def forward(self, x):
        x = self.linear1(x)
        return x


class LNLinearActivationModel(nn.Module):
    def __init__(self, fc_dim1, fc_dim2, dtype=torch.bfloat16, activation="sigmoid"):
        super().__init__()

        activation = activation.lower()
        activation_map = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU(),
            "relu6": nn.ReLU6(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "hardswish": nn.Hardswish(),
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        self.ln = nn.LayerNorm(fc_dim1, elementwise_affine=False)
        self.fc = nn.Linear(fc_dim1, fc_dim2, bias=False).to(dtype=dtype)
        self.activation = activation_map[activation]

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return self.activation(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=8, mlp_ratio=4, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-attention
        self.qkv = torch.nn.Linear(hidden_dim, 3 * hidden_dim, bias=False).to(dtype)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False).to(dtype)

        # MLP
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp_fc1 = torch.nn.Linear(hidden_dim, self.mlp_hidden_dim, bias=False).to(
            dtype
        )
        self.mlp_fc2 = torch.nn.Linear(self.mlp_hidden_dim, hidden_dim, bias=False).to(
            dtype
        )

        # Layer norms
        self.norm1 = RMSNorm(hidden_dim).to(dtype)
        self.norm2 = RMSNorm(hidden_dim).to(dtype)

        # Activation
        self.activation = torch.nn.GELU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Self-attention
        residual = x
        x = self.norm1(x)

        # Reshape qkv projection for better memory layout
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * hidden_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv  # Each has shape [batch_size, num_heads, seq_len, head_dim]

        # Scaled dot-product attention with proper reshaping
        # Reshape for better memory layout and avoid broadcasting issues
        q = q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(batch_size * self.num_heads, seq_len, self.head_dim)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        x = attn @ v  # [batch_size * num_heads, seq_len, head_dim]

        # Reshape back to original dimensions
        x = x.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        # Project back to hidden dimension
        x = self.proj(x)
        x = residual + x

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = self.activation(x)
        x = self.mlp_fc2(x)
        x = residual + x

        return x


def create_model_and_input_data(
    model_type: str,
    m: int,
    k: int,
    n: int,
    high_precision_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    activation: str = "relu",
):
    """Create a model and input data for benchmarking.

    Args:
        model_type (str): type of the model to be created
        batch_size (int): batch size of the input data
        device (str): device to run the model on
        high_precision_dtype (torch.dtype): data type of the model
        m, k, n (int): dimensions of the model and input data
    """
    if model_type == "linear":
        model = ToyLinearModel(k, n, high_precision_dtype).to(device)
        input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)
    elif "ln_linear" in model_type:
        # Extract activation type from model_type string
        match = re.search(r"ln_linear_?(\w+)?", model_type)
        activation = match.group(1) if match and match.group(1) else "relu"
        model = LNLinearActivationModel(
            k, n, high_precision_dtype, activation=activation
        ).to(device)
        input_data = torch.randn(m, k, device=device, dtype=high_precision_dtype)
    elif model_type == "transformer_block":
        # For transformer block, k is the hidden dimension
        model = TransformerBlock(
            k, num_heads=8, mlp_ratio=4, dtype=high_precision_dtype
        ).to(device)
        # Input shape for transformer is [batch_size, seq_len, hidden_dim]
        input_data = torch.randn(m, 16, k, device=device, dtype=high_precision_dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, input_data
