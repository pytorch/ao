"""
SpinQuant implementation (https://arxiv.org/abs/2405.16406)

Based on https://github.com/facebookresearch/SpinQuant
"""

from pathlib import Path
import typing

import torch
from torch import nn

from torchao.prototype.spinquant.hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix, get_hadK, matmul_hadU
from torchao._models.llama.model import RMSNorm, Transformer


class HadamardMultiplier(nn.Module):
    """Multiply the input by a Hadamard transform matrix."""

    def __init__(self, had_K, K, use_fp32=False):
        super().__init__()
        assert had_K is not None, "had_K must be provided"
        self.register_buffer("had_K", had_K)
        self.K = K
        self.use_fp32 = use_fp32

    def forward(self, x):
        if self.use_fp32:
            x = matmul_hadU(x.float(), self.had_K, self.K).to(x.dtype)
        else:
            x = matmul_hadU(x, self.had_K, self.K)
        return x


def apply_spinquant(model: Transformer, use_r1=False, use_r2=False, use_r4=True, pretrained_rotation_path=None):
    """
    Apply SpinQuant to a Transformer model: https://arxiv.org/abs/2405.16406
    
    Currently, the R1, R2, and R4 rotation matrices are implemented, and can be used independently
    from each other. For R1 and R2, random Hadamard matrices are used. The default is to only use R4,
    which appears to show best results in many cases (see https://github.com/pytorch/ao/pull/983).

    Note that the R3 rotation matrix and Cayley optimization for R1/R2 are currently not implemented.
    """
    assert isinstance(model, Transformer), "Only Transformer models are supported"

    original_device = next(model.parameters()).device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)

    # For testing purposes
    # Weights link: https://drive.google.com/drive/folders/1nV9juzE6_OHr10y6Ke5KCyOiGqDr0srX
    # pretrained_rotation_path = "7B_W4A16KV16_lr_1.5_seed_0/R.bin"

    if pretrained_rotation_path is not None:
        assert Path(pretrained_rotation_path).is_file(), "Pretrained rotation path does not exist"
        assert Path(pretrained_rotation_path).suffix == ".bin", "Expected a .bin file."

    if use_r1:
        fuse_layernorm_into_linear(model)
        apply_spinquant_r1(model, device, pretrained_rotation_path)
    if use_r2:
        apply_spinquant_r2(model, device, pretrained_rotation_path)
    if use_r4:
        apply_spinquant_r4(model, device)

    model.to(device=original_device)


def apply_spinquant_r1(model, device, pretrained_rotation_path=None):
    """Apply the SpinQuant R1 rotation matrix to the model."""
    
    if pretrained_rotation_path is not None:
        R1 = torch.load(pretrained_rotation_path)["R1"].to(device).to(torch.float64)
        assert R1.shape == (model.config.dim, model.config.dim), f"{R1.shape} vs {model.config.dim}"
    else:
        R1 = random_hadamard_matrix(model.config.dim, device)

    _rotate_model_r1(model, R1)


def apply_spinquant_r2(model, device, pretrained_rotation_path=None):
    """Apply the SpinQuant R2 rotation matrices to the model."""

    R2s = []  # note that unlike R1, there are multiple R2 matrices (one per layer)
    head_dim = model.config.head_dim
    for i, _ in enumerate(model.layers):
        if pretrained_rotation_path is not None:
            key = f"model.layers.{i}.self_attn.R2"
            R2s_ = torch.load(pretrained_rotation_path)
            R2 = R2s_[key].to(device).to(torch.float64)
            assert R2.shape == (head_dim, head_dim), f"{R2.shape} != ({head_dim}, {head_dim})"
        else:
            R2 = random_hadamard_matrix(head_dim, device)
        R2s.append(R2)

    _rotate_model_r2(model, R2s)


def apply_spinquant_r4(model, device):
    """Apply the SpinQuant R4 rotation matrix to the model."""
    _rotate_model_r4(model)
    _add_activation_wrappers_r4(model)


@torch.inference_mode()
def _fuse_layernorm_into_linear(layernorm: RMSNorm, linear_layers: typing.Iterable[torch.nn.Linear]):
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculate new weight and bias values
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

    # Set the original layernorm scale parameters to 1 (identity transform)
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)


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
def _rotate_model_r2(model, R2s):
    """Rotate the W_v and W_o weights of the multi-head self-attention modules."""

    head_dim = model.config.head_dim

    # Apply R2 rotation to all multi-head self-attention modules
    for idx, layer in enumerate(model.layers):
        attn = layer.attention

        R2 = R2s[idx]
    
        # Rotate W_o
        apply_exact_had_to_linear(attn.wo, had_dim=head_dim, output=False, R2=R2)

        # Extract W_v
        kv_size = model.config.n_local_heads * head_dim
        wq, wk, wv = attn.wqkv.weight.data.split([model.config.dim, kv_size, kv_size], dim=0)
        out_features, in_features = wv.shape
        wv_mod = nn.Linear(in_features, out_features, bias=attn.wqkv.bias is not None,
                           device=wv.device, dtype=wv.dtype)
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
        )  # apply exact (inverse) hadamard
        # print(f"Min/max *after* R4 rotation: {W.weight.data.min().item():.2f}/{W.weight.data.max().item():.2f}")


def _add_activation_wrappers_r4(model):
    """Modify the forward pass to rotate the activations at specific points."""
    # eval_utils/main.py:36
    had_K, K = get_hadK(model.config.intermediate_size)
    # print(f"K: {K}")
    for layer in model.layers:
        layer.feed_forward.w2 = nn.Sequential(
            HadamardMultiplier(had_K, K, use_fp32=False),
            layer.feed_forward.w2
        )


@torch.inference_mode()
def fuse_layernorm_into_linear(model):
    """
    Fuse RMSNorm weights into the subsequent linear layers. 
    
    This is done in the paper specifically to make pre-norm LLMs like LLaMa
    rotation-invariant when quantization is not present.
    """
    # Embedding fusion (from SpinQuant repo: utils/fuse_norm_utils.py:43)
    # I currently don't understand why this is necessary, so I contacted the
    # authors about it: https://github.com/facebookresearch/SpinQuant/issues/14
    W = model.tok_embeddings
    W_ = W.weight.data.double()
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    for layer in model.layers:
        _fuse_layernorm_into_linear(layer.ffn_norm, [layer.feed_forward.w1, layer.feed_forward.w3])
        _fuse_layernorm_into_linear(layer.attention_norm, [layer.attention.wqkv])

    _fuse_layernorm_into_linear(model.norm, [model.output])


def _rotate_mlp_output(layer, R1):
    mod = layer.feed_forward.w2
    _rotate_mod_weight_left(mod, R1)
    if mod.bias is not None:
        b = mod.bias.data.to(dtype=torch.float64)
        mod.bias.data = torch.matmul(R1.T, b).to(dtype=mod.weight.dtype)


def _rotate_mlp_input(layer, R1):
    _rotate_mod_weight_right(layer.feed_forward.w1, R1)
    _rotate_mod_weight_right(layer.feed_forward.w3, R1)


def _rotate_attention_output(layer, R1):
    mod = layer.attention.wo
    _rotate_mod_weight_left(mod, R1)
    if mod.bias is not None:
        b = mod.bias.data.to(dtype=torch.float64)
        mod.bias.data = torch.matmul(R1.T, b).to(dtype=mod.weight.dtype)


def _rotate_attention_inputs(layer, R1):
    _rotate_mod_weight_right(layer.attention.wqkv, R1)


def _rotate_head(model, R1):
    _rotate_mod_weight_right(model.output, R1)


def _rotate_embeddings(model, R1):
    _rotate_mod_weight_right(model.tok_embeddings, R1)


def _rotate_mod_weight_right(mod, R):
    dtype = mod.weight.dtype
    W = mod.weight.data.to(dtype=torch.float64)
    mod.weight.data = torch.matmul(W, R).to(dtype=dtype)


def _rotate_mod_weight_left(mod, R):
    dtype = mod.weight.dtype
    W = mod.weight.data.to(dtype=torch.float64)
    mod.weight.data = torch.matmul(R.T, W).to(dtype=dtype)

