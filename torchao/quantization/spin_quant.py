"""
SpinQuant implementation (https://arxiv.org/abs/2405.16406)

Based on https://github.com/facebookresearch/SpinQuant
"""

import typing

import torch
from torch import nn

from torchao.quantization.hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix, get_hadK, matmul_hadU
from torchao._models.llama.model import RMSNorm, Transformer, Attention


class HadamardMultiplier(nn.Module):
    """Multiply the input by a Hadamard matrix."""

    def __init__(self, had_K, K, fp32_had=False):
        super().__init__()
        assert had_K is not None, "had_K must be provided"
        self.register_buffer("had_K", had_K)
        self.K = K
        self.fp32_had = fp32_had

    def forward(self, x):
        if self.fp32_had:  # Full Hadamard in FP32
            x = matmul_hadU(x.float(), self.had_K, self.K).to(x.dtype)
        else:  # Full Hadamard in FP16
            x = matmul_hadU(x, self.had_K, self.K)
        return x


def apply_spinquant(model: Transformer):
    """
    Apply SpinQuant to a Transformer model: https://arxiv.org/abs/2405.16406
    
    Currently, this only applies the R1 + R2 + R4 rotation matrices to the model
    (not R3, and no Cayley optimization).
    """
    assert isinstance(model, Transformer), "Only Transformer models are supported"

    original_device = next(model.parameters()).device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)
    torch.manual_seed(0)  # for reproducability of random Hadamard matrices

    fuse_layernorm_weights_into_linear(model)
    apply_spinquant_r1(model, device)
    apply_spinquant_r2(model, device)
    apply_spinquant_r4(model, device)

    model.to(device=original_device)


def apply_spinquant_r1(model, device):
    R1 = random_hadamard_matrix(model.config.dim, device)
    _rotate_model_r1(model, R1)


def apply_spinquant_r2(model, device):
    R2 = random_hadamard_matrix(model.config.head_dim, device)
    _rotate_model_r2(model, R2)


def apply_spinquant_r4(model, device):
    _rotate_model_r4(model)
    _add_activation_wrappers_r4(model)


@torch.inference_mode()
def _fuse_layernorm_linear(layernorm: RMSNorm, linear_layers: typing.Iterable[torch.nn.Linear]):
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W = linear.weight.data.double()
        linear.weight.data = (W * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):  # not true for RMSNorm
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


@torch.inference_mode()
def _rotate_model_r1(model, R1):
    _rotate_embeddings(model, R1)
    _rotate_head(model, R1)

    for layer in model.layers:
        _rotate_attention_inputs(layer, R1)
        _rotate_attention_output(layer, R1)
        _rotate_mlp_input(layer, R1)
        _rotate_mlp_output(layer, R1)


@torch.inference_mode()
def _rotate_model_r2(model, R2):
    """Rotate the W_v and W_o weights of the multi-head self-attention modules."""
    
    # Apply R2 rotation to all multi-head self-attention modules
    for layer in model.layers:
        attn = layer.attention
        head_dim = model.config.head_dim

        # Rotate W_o
        apply_exact_had_to_linear(attn.wo, had_dim=head_dim, output=False, R2=R2)

        # Extract W_v
        kv_size = model.config.n_local_heads * head_dim
        wq, wk, wv = attn.wqkv.weight.data.split([model.config.dim, kv_size, kv_size], dim=0)
        out_features, in_features = wv.shape
        wv_mod = nn.Linear(in_features, out_features, bias=False, device=wq.device, dtype=wq.dtype)
        wv_mod.weight.data = wv

        # Rotate W_v
        apply_exact_had_to_linear(wv_mod, had_dim=head_dim, output=True, R2=R2)

        attn.wqkv.weight.data = torch.cat([wq, wk, wv_mod.weight.data], dim=0)


@torch.inference_mode()
def _rotate_model_r4(model):
    """Rotate the MLP output weights."""

    for layer in model.layers:
        W = layer.feed_forward.w2
        # print(f"Min/max before R4 rotation: {W.weight.data.min().item():.2f}/{W.weight.data.max().item():.2f}")
        apply_exact_had_to_linear(
            W, had_dim=-1, output=False
        )  # apply exact (inverse) hadamard on the weights of mlp output
        # print(f"Min/max *after* R4 rotation: {W.weight.data.min().item():.2f}/{W.weight.data.max().item():.2f}")


def _add_activation_wrappers_r4(model):
    """Modify the forward pass to rotate the activations at specific points."""
    # eval_utils/main.py:36
    fp32_had = False   # default used in SpinQuant
    had_K, K = get_hadK(model.config.intermediate_size)
    # print(f"K: {K}")
    _wrap_r4_layers(model, had_K, K, fp32_had)


def _wrap_r4_layers(module, had_K, K, fp32_had):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name == "w2":  # FeedForward last layer
            new_child = nn.Sequential(HadamardMultiplier(had_K, K, fp32_had), child)
            setattr(module, name, new_child)
        else:
            _wrap_r4_layers(child, had_K, K, fp32_had)


@torch.inference_mode()
def fuse_layernorm_weights_into_linear(model):
    """
    Fuse RMSNorm weights into the subsequent linear layers. 
    
    This is done in the paper specifically to make pre-norm LLMs like LLaMa
    rotation-invariant when quantization is not present. (I must admit I don't
    understand this. It would seem that either location for the RMSNorm weight
    multiplication would lead to the same results, but I might be missing
    something.)
    """
    # Embedding fusion (from utils/fuse_norm_utils.py:43)
    # I currently don't understand why this is necessary, so I'm omitting it (I
    # contacted the authors about it:
    # https://github.com/facebookresearch/SpinQuant/issues/14). It doesn't seem
    # to affect performance (tested on int4wo)
    for W in [model.tok_embeddings]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    for layer in model.layers:
        _fuse_layernorm_linear(layer.ffn_norm, [layer.feed_forward.w1, layer.feed_forward.w3])
        _fuse_layernorm_linear(layer.attention_norm, [layer.attention.wqkv])

        # Set the scale parameters to 1 (identity)
        W_norm = layer.ffn_norm.weight.data
        layer.ffn_norm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.attention_norm.weight.data
        layer.attention_norm.weight.data = torch.ones_like(W_norm)

    # Fuse the output layer
    _fuse_layernorm_linear(model.norm, [model.output])
    W_norm = model.norm.weight.data
    model.norm.weight.data = torch.ones_like(W_norm)


def _rotate_mlp_output(layer, R1):
    W = layer.feed_forward.w2
    dtype = W.weight.dtype
    W_ = W.weight.data.to( dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(dtype=dtype)


def _rotate_mlp_input(layer, R1):
    mlp_inputs = [layer.feed_forward.w1, layer.feed_forward.w3]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def _rotate_attention_output(layer, R1):
    W = layer.attention.wo
    dtype = W.weight.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(dtype=dtype)


def _rotate_attention_inputs(layer, R1):
    W = layer.attention.wqkv
    dtype = W.weight.dtype
    W_ = W.weight.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def _rotate_head(model, R1):
    W = model.output
    dtype = W.weight.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)


def _rotate_embeddings(model, R1):
    W = model.tok_embeddings
    dtype = W.weight.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(dtype=dtype)
