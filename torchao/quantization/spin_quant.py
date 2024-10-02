"""Based on https://github.com/facebookresearch/SpinQuant"""

import torch
from torch import nn

from torchao.quantization.hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix, get_hadK, matmul_hadU_cuda
from torchao._models.llama.model import Transformer, Attention


class HadamardMultiplier(nn.Module):
    """Wrapper that multiplies input by Hadamard matrix before feeding it to a module."""

    def __init__(self, module, had_K, K, fp32_had=False):
        super().__init__()
        self.module = module
        self.register_buffer("had_K", had_K)
        self.K = K
        self.fp32_had = fp32_had

    def forward(self, x):
        x_dtype = x.dtype

        if self.fp32_had:  # Full Hadamard in FP32
            x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
        else:  # Full Hadamard in FP16
            x = matmul_hadU_cuda(x, self.had_K, self.K)

        x = self.module(x)

        return x


@torch.inference_mode()
def _rotate_model_r2(model):
    """Rotate the W_v and W_o weights of the multi-head self-attention modules."""

    # Note: using a random Hadamard matrix for R2 is a substitute for the Caley
    # optimization they use in the paper. It's not clear that it adds value to
    # use a random Hadamard matrix.
    torch.manual_seed(0)
    R2 = random_hadamard_matrix(model.config.head_dim, "cuda")
    
    # Apply R2 rotation to all multi-head self-attention modules
    for m in model.modules():
        if isinstance(m, Attention):
            head_dim = model.config.head_dim

            # Rotate W_o
            apply_exact_had_to_linear(m.wo, had_dim=head_dim, output=False, R2=R2)

            # Extract W_v
            kv_size = model.config.n_local_heads * head_dim
            wq, wk, wv = m.wqkv.weight.data.split([model.config.dim, kv_size, kv_size], dim=0)
            out_features, in_features = wv.shape
            wv_mod = nn.Linear(in_features, out_features, bias=False, device=wq.device, dtype=wq.dtype)
            wv_mod.weight.data = wv

            # Rotate W_v
            apply_exact_had_to_linear(wv_mod, had_dim=head_dim, output=True, R2=R2)

            m.wqkv.weight.data = torch.cat([wq, wk, wv_mod.weight.data], dim=0)


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


def _wrap_r4_layers(module, had_K, K, fp32_had):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name == "w2":  # FeedForward last layer
            setattr(module, name, HadamardMultiplier(child, had_K, K, fp32_had))
        else:
            _wrap_r4_layers(child, had_K, K, fp32_had)


def _add_activation_wrappers_r3_r4(model):
    """Modify the forward pass to rotate the activations at specific points."""
    
    # R3
    # TODO

    # R4
    # eval_utils/main.py:36
    fp32_had = False   # default used in SpinQuant
    had_K, K = get_hadK(model.config.intermediate_size)
    _wrap_r4_layers(model, had_K, K, fp32_had)


def apply_spinquant_to_llama(model: Transformer):
    """
    Apply SpinQuant to Llama: https://arxiv.org/abs/2405.16406
    
    Currently, this only applies the R2 + R3 + R4 rotation matrices to the model (not R1).
    """
    _rotate_model_r2(model)
    _rotate_model_r4(model)
    _add_activation_wrappers_r3_r4(model)
